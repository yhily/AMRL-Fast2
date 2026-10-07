#!/usr/bin/env python3
"""Audit TorchScript as an execution-only PPO-update candidate.

The probe does not alter the experiment runner.  It traces only SB3's
``MlpExtractor`` and verifies that its replacement preserves real
``evaluate_actions`` outputs, gradients and a single Adam parameter update on
the production B=64 minibatch shape before measuring forward+backward cost.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrl_fast.agent import PPOBaseAgent, set_global_seed
from amrl_fast.envs import make_env


class PPOCategoricalLossCore(torch.nn.Module):
    """Static discrete PPO loss path for the currently frozen SB3 policy."""

    def __init__(self, policy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, observations, actions, old_log_probability, advantages, returns):
        features = self.policy.extract_features(observations)
        latent_policy, latent_value = self.policy.mlp_extractor(features)
        logits = self.policy.action_net(latent_policy)
        distribution = torch.distributions.Categorical(logits=logits)
        log_probability = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.policy.value_net(latent_value).flatten()
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ratio = torch.exp(log_probability - old_log_probability)
        policy_loss = -torch.minimum(
            normalized_advantages * ratio,
            normalized_advantages * torch.clamp(ratio, 0.8, 1.2),
        ).mean()
        value_loss = torch.nn.functional.mse_loss(returns, values)
        entropy_loss = -entropy.mean()
        return values, log_probability, entropy, policy_loss + 0.01 * entropy_loss + 0.5 * value_loss


def sync() -> None:
    torch.cuda.synchronize()


def metric(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    a = reference.detach().float().reshape(-1)
    b = candidate.detach().float().reshape(-1)
    difference = a - b
    centered_a, centered_b = a - a.mean(), b - b.mean()
    norm = torch.linalg.vector_norm(centered_a) * torch.linalg.vector_norm(centered_b)
    if norm <= 1e-12:
        correlation = 1.0 if torch.allclose(a, b) else 0.0
    else:
        correlation = float(torch.dot(centered_a, centered_b).div(norm).item())
    return {
        "correlation": correlation,
        "relative_l2": float(
            torch.linalg.vector_norm(difference).div(torch.linalg.vector_norm(a).clamp_min(1e-12)).item()
        ),
        "max_absolute": float(difference.abs().max().item()),
    }


def objective(policy, observations, actions, old_log_probability, advantages, returns):
    values, log_probability, entropy = policy.evaluate_actions(observations, actions)
    values = values.flatten()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    ratio = torch.exp(log_probability - old_log_probability)
    policy_loss = -torch.minimum(
        advantages * ratio, advantages * torch.clamp(ratio, 0.8, 1.2)
    ).mean()
    value_loss = torch.nn.functional.mse_loss(returns, values)
    entropy_loss = -entropy.mean() if entropy is not None else torch.zeros((), device=values.device)
    return values, log_probability, entropy, policy_loss + 0.5 * value_loss + 0.01 * entropy_loss


def objective_with_core(core, observations, actions, old_log_probability, advantages, returns):
    return core(observations, actions, old_log_probability, advantages, returns)


def all_gradients(policy) -> torch.Tensor:
    values = [parameter.grad.detach().reshape(-1) for parameter in policy.parameters() if parameter.grad is not None]
    return torch.cat(values) if values else torch.empty(0, device="cuda")


def latency(fn, policy, observations, actions, old_log_probability, advantages, returns, *, repeats: int = 100) -> float:
    for _ in range(20):
        policy.optimizer.zero_grad(set_to_none=True)
        _, _, _, loss = fn(policy, observations, actions, old_log_probability, advantages, returns)
        loss.backward()
    sync()
    started = time.perf_counter()
    for _ in range(repeats):
        policy.optimizer.zero_grad(set_to_none=True)
        _, _, _, loss = fn(policy, observations, actions, old_log_probability, advantages, returns)
        loss.backward()
    sync()
    return (time.perf_counter() - started) * 1e6 / repeats


def passes(values: dict[str, float]) -> bool:
    return values["correlation"] >= 0.999 and values["relative_l2"] <= 5e-3 and values["max_absolute"] <= 1e-1


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    set_global_seed(0)
    env, resolved = make_env("CartPole-v1", seed=0)
    try:
        agent = PPOBaseAgent(env, device=torch.device("cuda"), seed=0, rollout_steps=10_000)
        standard = copy.deepcopy(agent.model.policy).cuda()
        candidate = copy.deepcopy(agent.model.policy).cuda()
        observations = torch.randn(64, 4, device="cuda")
        actions = torch.randint(0, 2, (64,), device="cuda")
        old_log_probability = torch.randn(64, device="cuda")
        advantages = torch.randn(64, device="cuda")
        returns = torch.randn(64, device="cuda")
        traced = torch.jit.trace(candidate.mlp_extractor, observations, check_trace=True)
        candidate.mlp_extractor = traced

        standard.optimizer.zero_grad(set_to_none=True)
        candidate.optimizer.zero_grad(set_to_none=True)
        std_value, std_log, std_entropy, std_loss = objective(
            standard, observations, actions, old_log_probability, advantages, returns
        )
        opt_value, opt_log, opt_entropy, opt_loss = objective(
            candidate, observations, actions, old_log_probability, advantages, returns
        )
        std_loss.backward()
        opt_loss.backward()
        std_gradient = all_gradients(standard)
        opt_gradient = all_gradients(candidate)
        output_metrics = {
            "value": metric(std_value, opt_value),
            "log_probability": metric(std_log, opt_log),
            "entropy": metric(std_entropy, opt_entropy),
            "loss": metric(std_loss.reshape(1), opt_loss.reshape(1)),
            "gradient": metric(std_gradient, opt_gradient),
        }
        standard.optimizer.step()
        candidate.optimizer.step()
        parameter_metrics = metric(
            torch.cat([x.detach().reshape(-1) for x in standard.parameters()]),
            torch.cat([x.detach().reshape(-1) for x in candidate.parameters()]),
        )
        standard_us = latency(objective, standard, observations, actions, old_log_probability, advantages, returns)
        candidate_us = latency(objective, candidate, observations, actions, old_log_probability, advantages, returns)
        speedup = standard_us / candidate_us
        admitted = speedup >= 1.02 and all(passes(item) for item in output_metrics.values()) and passes(parameter_metrics)
        full_standard = copy.deepcopy(agent.model.policy).cuda()
        full_candidate = copy.deepcopy(agent.model.policy).cuda()
        standard_core = PPOCategoricalLossCore(full_standard).cuda()
        traced_core = torch.jit.trace(
            PPOCategoricalLossCore(full_candidate).cuda(),
            (observations, actions, old_log_probability, advantages, returns),
            check_trace=True,
        )
        full_standard.optimizer.zero_grad(set_to_none=True)
        full_candidate.optimizer.zero_grad(set_to_none=True)
        full_standard_values = objective(
            full_standard, observations, actions, old_log_probability, advantages, returns
        )
        full_candidate_values = objective_with_core(
            traced_core, observations, actions, old_log_probability, advantages, returns
        )
        full_standard_values[-1].backward()
        full_candidate_values[-1].backward()
        full_metrics = {
            "value": metric(full_standard_values[0], full_candidate_values[0]),
            "log_probability": metric(full_standard_values[1], full_candidate_values[1]),
            "entropy": metric(full_standard_values[2], full_candidate_values[2]),
            "loss": metric(full_standard_values[3].reshape(1), full_candidate_values[3].reshape(1)),
            "gradient": metric(all_gradients(full_standard), all_gradients(full_candidate)),
        }
        full_standard.optimizer.step()
        full_candidate.optimizer.step()
        full_parameter_metrics = metric(
            torch.cat([x.detach().reshape(-1) for x in full_standard.parameters()]),
            torch.cat([x.detach().reshape(-1) for x in full_candidate.parameters()]),
        )
        full_standard_us = latency(
            objective, full_standard, observations, actions, old_log_probability, advantages, returns
        )
        full_candidate_us = latency(
            lambda _policy, *items: objective_with_core(traced_core, *items),
            full_candidate, observations, actions, old_log_probability, advantages, returns,
        )
        full_speedup = full_standard_us / full_candidate_us
        full_admitted = full_speedup >= 1.02 and all(passes(item) for item in full_metrics.values()) and passes(full_parameter_metrics)
        payload = {
            "development_only": True,
            "paper_result_eligible": False,
            "environment": "CartPole-v1",
            "resolved_environment_id": resolved,
            "candidate": "TorchScript traced SB3 MlpExtractor inside real PPO evaluate_actions",
            "shape": [64, 4],
            "standard_forward_backward_us": standard_us,
            "candidate_forward_backward_us": candidate_us,
            "speedup": speedup,
            "metrics": output_metrics,
            "parameter_update": parameter_metrics,
            "admitted": admitted,
            "full_ppo_loss_core": {
                "standard_forward_backward_us": full_standard_us,
                "candidate_forward_backward_us": full_candidate_us,
                "speedup": full_speedup,
                "metrics": full_metrics,
                "parameter_update": full_parameter_metrics,
                "admitted": full_admitted,
            },
        }
    finally:
        env.close()
    destination = ROOT / "artifacts" / "base_agent_hotspots" / "torchscript_ppo_probe.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
