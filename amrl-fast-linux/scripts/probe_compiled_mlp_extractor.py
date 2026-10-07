#!/usr/bin/env python3
"""Fidelity and latency probe for compiling the pure SB3 MLP extractor.

Distribution construction, action sampling, PPO loss and optimizer remain on
the standard code path.  Only the deterministic actor/critic MLP extractor is
compiled, so this probe does not disable Categorical validation or alter RL
semantics.
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


def _sync() -> None:
    torch.cuda.synchronize()


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference, candidate = reference.detach().float().reshape(-1), candidate.detach().float().reshape(-1)
    diff = reference - candidate
    a, b = reference - reference.mean(), candidate - candidate.mean()
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    corr = 1.0 if denominator <= 1e-12 and torch.allclose(reference, candidate) else (
        0.0 if denominator <= 1e-12 else float((torch.dot(a, b) / denominator).item())
    )
    return {
        "correlation": corr,
        "relative_l2": float((torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(reference).clamp_min(1e-12)).item()),
        "max_absolute": float(diff.abs().max().item()),
    }


def _time(fn, *, warmup: int = 50, repeats: int = 1000) -> float:
    for _ in range(warmup): fn()
    _sync(); started = time.perf_counter()
    for _ in range(repeats): fn()
    _sync()
    return (time.perf_counter() - started) * 1e6 / repeats


def _probe_shape(reference, compiled, batch: int) -> dict:
    x_ref = torch.randn(batch, 4, device="cuda", requires_grad=True)
    x_opt = x_ref.detach().clone().requires_grad_(True)
    ref_pi, ref_vf = reference(x_ref)
    opt_pi, opt_vf = compiled(x_opt)
    ref_loss = ref_pi.square().mean() + ref_vf.square().mean()
    opt_loss = opt_pi.square().mean() + opt_vf.square().mean()
    ref_loss.backward(); opt_loss.backward()
    ref_grads = torch.cat([parameter.grad.detach().reshape(-1) for parameter in reference.parameters()])
    opt_grads = torch.cat([parameter.grad.detach().reshape(-1) for parameter in compiled.parameters()])
    reference.zero_grad(set_to_none=True); compiled.zero_grad(set_to_none=True)
    with torch.no_grad():
        standard_us = _time(lambda: reference(x_ref.detach()))
        compiled_us = _time(lambda: compiled(x_opt.detach()))
    return {
        "batch": batch,
        "policy_output": _metrics(ref_pi, opt_pi),
        "value_output": _metrics(ref_vf, opt_vf),
        "input_gradient": _metrics(x_ref.grad, x_opt.grad),
        "parameter_gradient": _metrics(ref_grads, opt_grads),
        "standard_forward_us": standard_us,
        "compiled_forward_us": compiled_us,
        "speedup": standard_us / compiled_us,
    }


def main() -> None:
    if not torch.cuda.is_available(): raise SystemExit("CUDA required")
    if not hasattr(torch, "compile"): raise SystemExit("torch.compile unavailable")
    set_global_seed(0)
    env, resolved = make_env("CartPole-v1", seed=0, allow_version_fallback=False)
    agent = PPOBaseAgent(env, device=torch.device("cuda"), seed=0, rollout_steps=5_000)
    try:
        reference = copy.deepcopy(agent.model.policy.mlp_extractor).cuda()
        candidate = copy.deepcopy(agent.model.policy.mlp_extractor).cuda()
        compiled = torch.compile(candidate, mode="reduce-overhead", fullgraph=True, dynamic=False)
        results = [_probe_shape(reference, compiled, batch) for batch in (1, 64)]
        accepted = all(
            item["speedup"] >= 1.02
            and all(metric["correlation"] >= 0.999 and metric["relative_l2"] <= 5e-3 and metric["max_absolute"] <= 1e-1
                    for metric in (item["policy_output"], item["value_output"], item["input_gradient"], item["parameter_gradient"]))
            for item in results
        )
        payload = {
            "development_only": True,
            "environment": "CartPole-v1",
            "resolved_environment_id": resolved,
            "candidate": "torch.compile(MlpExtractor, mode=reduce-overhead)",
            "input_shapes": {"rollout": [1, 4], "ppo_minibatch": [64, 4]},
            "results": results,
            "admitted": accepted,
            "reason": "Both real input shapes meet fidelity and >=1.02x speed criteria." if accepted else "Rejected by fidelity or speed screen.",
        }
    finally:
        env.close()
    target = ROOT / "artifacts" / "base_agent_hotspots" / "cartpole_compiled_mlp_probe.json"
    target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(payload, indent=2, sort_keys=True),encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__": main()
