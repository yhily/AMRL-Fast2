#!/usr/bin/env python3
"""Run a short real-training integration check for the rebuilt backend.

This is deliberately a development smoke test, not a manuscript experiment:
it records the shapes observed in an actual AMRL training chain before the
five-seed Linux runs are started.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrl_fast.agent import AMRLAgent
from amrl_fast.envs import make_env
from amrl_fast.rebuilt_lstm import RebuiltLSTMBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="CartPole-v1")
    parser.add_argument("--timesteps", type=int, default=1024)
    parser.add_argument("--adaptation-interval", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-version-fallback", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    backend = RebuiltLSTMBackend()
    admission = backend.calibrate(device)
    if not admission.admitted:
        raise SystemExit(f"Rebuilt backend was not admitted: {admission.reason}")
    env, resolved_environment = make_env(
        args.environment,
        seed=args.seed,
        allow_version_fallback=args.allow_version_fallback,
    )
    try:
        agent = AMRLAgent(
            env,
            device=device,
            seed=args.seed,
            use_fast=True,
            backend=backend,
            adaptation_interval=args.adaptation_interval,
            warmup_steps=0,
            transition_steps=0,
        )
        history = agent.train(args.timesteps)
        result = {
            "kernel_development_only": True,
            "version": "integration-v1-real-training-chain",
            "requested_environment": args.environment,
            "resolved_environment": resolved_environment,
            "timesteps": args.timesteps,
            "adaptation_interval": args.adaptation_interval,
            "seed": args.seed,
            "backend": backend.to_dict(),
            "controller_output_count": len(history.controller_outputs),
            "reward_delta_count": len(history.reward_deltas),
            "timing": agent.telemetry.timing.to_dict(),
            "stability": agent.telemetry.stability.to_dict(),
        }
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
