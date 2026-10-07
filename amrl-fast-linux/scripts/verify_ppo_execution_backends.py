#!/usr/bin/env python3
"""Audit PPO execution-only candidates before a full training run.

The audit uses the real SB3 ``ActorCriticPolicy.evaluate_actions`` and the
same clipped PPO loss algebra at the two minibatch sizes seen in this project.
It compares full parameter updates from identical initial states; it does not
change the learner or enter the experiment result directory.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrl_fast.agent import PPOBaseAgent, set_global_seed


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _max_metrics(reference: list[torch.Tensor], candidate: list[torch.Tensor]) -> dict[str, float]:
    ref = torch.cat([item.detach().reshape(-1).float().cpu() for item in reference])
    got = torch.cat([item.detach().reshape(-1).float().cpu() for item in candidate])
    difference = got - ref
    relative_l2 = float(difference.norm() / ref.norm().clamp_min(1e-12))
    maximum = float(difference.abs().max())
    if ref.numel() < 2 or float(ref.std()) == 0.0 or float(got.std()) == 0.0:
        correlation = 1.0 if maximum == 0.0 else 0.0
    else:
        correlation = float(torch.corrcoef(torch.stack((ref, got)))[0, 1])
    return {"correlation": correlation, "relative_l2": relative_l2, "max_absolute": maximum}


def _loss(policy, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    values, log_probability, entropy = policy.evaluate_actions(batch["obs"], batch["actions"])
    values = values.flatten()
    ratio = torch.exp(log_probability - batch["old_log_probability"])
    surrogate_a = batch["advantages"] * ratio
    surrogate_b = batch["advantages"] * torch.clamp(ratio, 0.8, 1.2)
    policy_loss = -torch.minimum(surrogate_a, surrogate_b).mean()
    value_loss = F.mse_loss(batch["returns"], values)
    entropy_loss = -entropy.mean()
    return policy_loss + 0.5 * value_loss + 0.01 * entropy_loss


def _update(policy, optimizer, batch: dict[str, torch.Tensor]) -> tuple[float, list[torch.Tensor]]:
    optimizer.zero_grad(set_to_none=True)
    total_loss = _loss(policy, batch)
    total_loss.backward()
    gradients = [parameter.grad.detach().clone() for parameter in policy.parameters() if parameter.grad is not None]
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return float(total_loss.detach()), gradients


def _new_policy(state_dict: dict[str, torch.Tensor], *, device: torch.device):
    """Rebuild from a state dict instead of deep-copying SB3 runtime caches."""
    environment = gym.make("CartPole-v1")
    agent = PPOBaseAgent(environment, device=device, seed=0, rollout_steps=5_000)
    policy = agent.model.policy
    policy.load_state_dict(state_dict)
    return agent, policy


def _time_updates(state_dict, batch, *, optimizer_keyword: dict, repeats: int, device: torch.device) -> float:
    # A fresh real SB3 policy avoids copying non-leaf policy distribution caches.
    timed_agent, timed_policy = _new_policy(state_dict, device=device)
    timed_optimizer = torch.optim.Adam(timed_policy.parameters(), **optimizer_keyword)
    _sync(device)
    start = time.perf_counter_ns()
    for _ in range(repeats):
        _update(timed_policy, timed_optimizer, batch)
    _sync(device)
    elapsed = (time.perf_counter_ns() - start) / repeats / 1_000.0
    timed_agent.env.close()
    return elapsed


def _audit_variant(reference_policy, batch, *, variant: str, device: torch.device) -> dict:
    state_dict = {key: value.detach().clone() for key, value in reference_policy.state_dict().items()}
    standard_agent, standard = _new_policy(state_dict, device=device)
    candidate_agent, candidate = _new_policy(state_dict, device=device)
    standard_keyword = {"lr": 3e-4, "eps": 1e-5, "foreach": False}
    standard_optimizer = torch.optim.Adam(standard.parameters(), **standard_keyword)
    keyword = {"lr": 3e-4, "eps": 1e-5}
    if variant == "foreach":
        keyword["foreach"] = True
    elif variant == "fused":
        keyword["fused"] = True
    else:
        raise ValueError(variant)
    candidate_optimizer = torch.optim.Adam(candidate.parameters(), **keyword)
    _sync(device)
    standard_loss, standard_gradients = _update(standard, standard_optimizer, batch)
    _sync(device)
    candidate_loss, candidate_gradients = _update(candidate, candidate_optimizer, batch)
    _sync(device)
    parameter_metrics = _max_metrics(list(standard.parameters()), list(candidate.parameters()))
    gradient_metrics = _max_metrics(standard_gradients, candidate_gradients)
    # More than one step is timed to make CUDA launch noise less dominant.
    standard_us = _time_updates(state_dict, batch, optimizer_keyword=standard_keyword, repeats=64, device=device)
    candidate_us = _time_updates(state_dict, batch, optimizer_keyword=keyword, repeats=64, device=device)
    admitted = (
        parameter_metrics["correlation"] >= 0.999
        and gradient_metrics["correlation"] >= 0.99
        and parameter_metrics["relative_l2"] <= 5e-3
        and gradient_metrics["relative_l2"] <= 5e-3
        and standard_us / candidate_us >= 1.02
    )
    result = {
        "variant": variant,
        "loss_absolute_difference": abs(standard_loss - candidate_loss),
        "parameter_update": parameter_metrics,
        "gradient": gradient_metrics,
        "standard_update_us": standard_us,
        "candidate_update_us": candidate_us,
        "optimizer_speedup": standard_us / candidate_us,
        "admitted_by_local_fidelity_and_runtime_screen": admitted,
    }
    standard_agent.env.close()
    candidate_agent.env.close()
    return result


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This audit must run on the target CUDA training stack.")
    device = torch.device("cuda")
    set_global_seed(0)
    environment = gym.make("CartPole-v1")
    agent = PPOBaseAgent(environment, device=device, seed=0, rollout_steps=5_000)
    policy = agent.model.policy
    capability = torch.cuda.get_device_capability(device)
    payload = {
        "development_only": True,
        "device": torch.cuda.get_device_name(device),
        "cuda_capability": list(capability),
        "torch_compile": {
            "admitted": False,
            "reason": "Rejected on this stack: TorchInductor/Triton CUDA code generation requires compute capability >= 7.0; Titan Xp is 6.1.",
        },
        "batches": {},
    }
    for batch_size in (8, 64):
        generator = torch.Generator(device=device).manual_seed(10_000 + batch_size)
        batch = {
            "obs": torch.randn(batch_size, 4, generator=generator, device=device),
            "actions": torch.randint(0, 2, (batch_size, 1), generator=generator, device=device),
            "old_log_probability": torch.randn(batch_size, generator=generator, device=device),
            "advantages": torch.randn(batch_size, generator=generator, device=device),
            "returns": torch.randn(batch_size, generator=generator, device=device),
        }
        results = {}
        for variant in ("foreach", "fused"):
            try:
                results[variant] = _audit_variant(policy, batch, variant=variant, device=device)
            except Exception as error:
                results[variant] = {"variant": variant, "admitted_by_local_fidelity_and_runtime_screen": False,
                                    "error": f"{type(error).__name__}: {error}"}
        payload["batches"][str(batch_size)] = results
    target = ROOT / "artifacts" / "base_agent_hotspots" / "ppo_execution_backend_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
