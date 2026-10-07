#!/usr/bin/env python3
"""Feasibility probe for a fixed-shape CUDA-graph SB3 policy forward path.

This probe does not patch training.  It captures the exact existing
``ActorCriticPolicy.forward`` call used during rollout collection, validates
graph outputs against the standard policy conditional on the sampled action,
and benchmarks standard forward versus graph replay for CartPole's real
batch-one observation shape.
"""
from __future__ import annotations

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


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    raw_a, raw_b = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    a, b = raw_a - raw_a.mean(), raw_b - raw_b.mean()
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if denominator <= 1e-12:
        return 1.0 if torch.allclose(raw_a, raw_b) else 0.0
    return float((torch.dot(a, b) / denominator).item())


def _time(fn, *, warmup: int = 100, repeats: int = 1000) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync()
    return (time.perf_counter() - started) * 1e6 / repeats


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    set_global_seed(0)
    env, resolved = make_env("CartPole-v1", seed=0, allow_version_fallback=False)
    agent = PPOBaseAgent(env, device=torch.device("cuda"), seed=0, rollout_steps=5_000)
    policy = agent.model.policy
    was_training = policy.training
    policy.set_training_mode(False)
    obs = torch.zeros((1, *agent.model.observation_space.shape), dtype=torch.float32, device="cuda")
    try:
        with torch.no_grad():
            for _ in range(20):
                policy(obs)
        _sync()
        static_obs = obs.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_actions, static_values, static_log_prob = policy(static_obs)
        _sync()

        def graph_forward():
            static_obs.copy_(obs)
            graph.replay()
            return static_actions, static_values, static_log_prob

        actions, values, log_prob = graph_forward()
        _sync()
        with torch.no_grad():
            reference_values, reference_log_prob, _ = policy.evaluate_actions(obs, actions)
        value_diff = reference_values - values
        log_prob_diff = reference_log_prob - log_prob
        standard_us = _time(lambda: policy(obs))
        graph_us = _time(graph_forward)
        payload = {
            "development_only": True,
            "environment": "CartPole-v1",
            "resolved_environment_id": resolved,
            "shape": list(obs.shape),
            "capture_succeeded": True,
            "value_correlation": _corr(reference_values, values),
            "value_relative_l2": float((torch.linalg.vector_norm(value_diff) / torch.linalg.vector_norm(reference_values).clamp_min(1e-12)).item()),
            "value_max_absolute": float(value_diff.abs().max().item()),
            "log_probability_correlation": _corr(reference_log_prob, log_prob),
            "log_probability_relative_l2": float((torch.linalg.vector_norm(log_prob_diff) / torch.linalg.vector_norm(reference_log_prob).clamp_min(1e-12)).item()),
            "log_probability_max_absolute": float(log_prob_diff.abs().max().item()),
            "action_finite": bool(torch.isfinite(actions).all().item()),
            "standard_forward_us": standard_us,
            "cuda_graph_forward_us": graph_us,
            "speedup": standard_us / graph_us,
        }
    finally:
        policy.set_training_mode(was_training)
        env.close()
    target = ROOT / "artifacts" / "base_agent_hotspots" / "cartpole_cuda_graph_policy_probe.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
