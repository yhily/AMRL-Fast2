#!/usr/bin/env python3
"""Screen a cached fused actor/critic MLP rollout backend.

This is an execution-only candidate for the real SB3 rollout hot path.  The
policy and value MLPs are algebraically packed into two larger linear calls:
two first-layer linears become one vertically packed linear, and two
second-layer linears become one block-diagonal linear.  The candidate is used
only for inference-shaped rollout inputs; PPO training, distributions, action
sampling, losses, gradients, and optimizer remain untouched.

The script intentionally performs admission measurements before any integration
change: output and gradient equivalence are checked with the source parameters,
and cached runtime is compared on the actual B=1 rollout shape.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrl_fast.agent import PPOBaseAgent, set_global_seed
from amrl_fast.envs import make_env


ENVIRONMENTS = ("CartPole-v1", "MountainCar-v0", "BSuite Memory")


def _sync() -> None:
    torch.cuda.synchronize()


def _metric(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.detach().float().reshape(-1)
    candidate = candidate.detach().float().reshape(-1)
    difference = reference - candidate
    centered_ref = reference - reference.mean()
    centered_candidate = candidate - candidate.mean()
    denom = torch.linalg.vector_norm(centered_ref) * torch.linalg.vector_norm(centered_candidate)
    correlation = 1.0 if denom <= 1e-12 and torch.allclose(reference, candidate) else (
        0.0 if denom <= 1e-12 else float(torch.dot(centered_ref, centered_candidate).div(denom).item())
    )
    return {
        "correlation": correlation,
        "relative_l2": float(torch.linalg.vector_norm(difference).div(torch.linalg.vector_norm(reference).clamp_min(1e-12)).item()),
        "max_absolute": float(difference.abs().max().item()),
    }


def _linear_layers(network: nn.Sequential) -> tuple[nn.Linear, nn.Linear]:
    linears = [module for module in network if isinstance(module, nn.Linear)]
    activations = [module for module in network if isinstance(module, nn.Tanh)]
    if len(linears) != 2 or len(activations) != 2:
        raise RuntimeError("Candidate currently supports the public two-layer Tanh MlpExtractor only.")
    return linears[0], linears[1]


def fused_dynamic(policy_net: nn.Sequential, value_net: nn.Sequential, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable algebraic form used only for gradient admission."""
    policy_first, policy_second = _linear_layers(policy_net)
    value_first, value_second = _linear_layers(value_net)
    first_weight = torch.cat((policy_first.weight, value_first.weight), dim=0)
    first_bias = torch.cat((policy_first.bias, value_first.bias), dim=0)
    second_weight = torch.block_diag(policy_second.weight, value_second.weight)
    second_bias = torch.cat((policy_second.bias, value_second.bias), dim=0)
    hidden = torch.tanh(F.linear(features, first_weight, first_bias))
    latent = torch.tanh(F.linear(hidden, second_weight, second_bias))
    policy_width = policy_second.out_features
    return latent[:, :policy_width], latent[:, policy_width:]


class CachedFusedMlp:
    """Detached fixed-weight pack for a single rollout between PPO updates."""

    def __init__(self, policy_net: nn.Sequential, value_net: nn.Sequential) -> None:
        self.policy_net = policy_net
        self.value_net = value_net
        self.policy_width = 0
        self.first_weight: torch.Tensor
        self.first_bias: torch.Tensor
        self.second_weight: torch.Tensor
        self.second_bias: torch.Tensor
        self.refresh()

    @torch.no_grad()
    def refresh(self) -> None:
        policy_first, policy_second = _linear_layers(self.policy_net)
        value_first, value_second = _linear_layers(self.value_net)
        self.policy_width = policy_second.out_features
        self.first_weight = torch.cat((policy_first.weight, value_first.weight), dim=0).contiguous()
        self.first_bias = torch.cat((policy_first.bias, value_first.bias), dim=0).contiguous()
        self.second_weight = torch.block_diag(policy_second.weight, value_second.weight).contiguous()
        self.second_bias = torch.cat((policy_second.bias, value_second.bias), dim=0).contiguous()

    @torch.no_grad()
    def __call__(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.tanh(F.linear(features, self.first_weight, self.first_bias))
        latent = torch.tanh(F.linear(hidden, self.second_weight, self.second_bias))
        return latent[:, :self.policy_width], latent[:, self.policy_width:]


def _time(callable_, *, warmup: int = 200, repeats: int = 4_000) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            callable_()
        _sync()
        started = time.perf_counter()
        for _ in range(repeats):
            callable_()
        _sync()
    return (time.perf_counter() - started) * 1e6 / repeats


def _check_environment(environment: str) -> dict:
    set_global_seed(0)
    env, resolved = make_env(environment, seed=0, allow_version_fallback=False)
    try:
        agent = PPOBaseAgent(env, device=torch.device("cuda"), seed=0, rollout_steps=5_000)
        extractor = agent.model.policy.mlp_extractor.cuda()
        policy_net, value_net = extractor.policy_net, extractor.value_net
        input_dim = _linear_layers(policy_net)[0].in_features
        source_input = torch.randn(1, input_dim, device="cuda", requires_grad=True)
        dynamic_input = source_input.detach().clone().requires_grad_(True)

        reference_policy, reference_value = extractor(source_input)
        dynamic_policy, dynamic_value = fused_dynamic(policy_net, value_net, dynamic_input)
        reference_loss = reference_policy.square().mean() + reference_value.square().mean()
        dynamic_loss = dynamic_policy.square().mean() + dynamic_value.square().mean()
        reference_loss.backward()
        reference_grads = torch.cat([parameter.grad.detach().reshape(-1) for parameter in extractor.parameters()])
        extractor.zero_grad(set_to_none=True)
        dynamic_loss.backward()
        dynamic_grads = torch.cat([parameter.grad.detach().reshape(-1) for parameter in extractor.parameters()])
        extractor.zero_grad(set_to_none=True)

        cached = CachedFusedMlp(policy_net, value_net)
        with torch.no_grad():
            cached_policy, cached_value = cached(source_input.detach())
            standard_us = _time(lambda: extractor(source_input.detach()))
            cached_us = _time(lambda: cached(source_input.detach()))
        outputs = (_metric(reference_policy, cached_policy), _metric(reference_value, cached_value))
        gradients = (_metric(source_input.grad, dynamic_input.grad), _metric(reference_grads, dynamic_grads))
        passed_fidelity = all(
            measure["correlation"] >= 0.999
            and measure["relative_l2"] <= 5e-3
            and measure["max_absolute"] <= 1e-1
            for measure in (*outputs, *gradients)
        )
        speedup = standard_us / cached_us
        return {
            "environment": environment,
            "resolved_environment_id": resolved,
            "candidate": "cached-fused-dual-mlp-rollout",
            "real_rollout_feature_shape": [1, input_dim],
            "standard_mlp_forward_us": standard_us,
            "candidate_mlp_forward_us": cached_us,
            "speedup": speedup,
            "policy_output": outputs[0],
            "value_output": outputs[1],
            "input_gradient": gradients[0],
            "parameter_gradient": gradients[1],
            "admitted": bool(passed_fidelity and speedup >= 1.02),
            "reason": "Passed output/gradient fidelity and >=1.02x B=1 rollout MLP speed screen."
            if passed_fidelity and speedup >= 1.02
            else "Rejected by the common fidelity or runtime admission criterion.",
        }
    finally:
        env.close()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    report = {"development_only": True, "results": [_check_environment(item) for item in ENVIRONMENTS]}
    target = ROOT / "artifacts" / "base_agent_hotspots" / "fused_rollout_mlp_probe.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
