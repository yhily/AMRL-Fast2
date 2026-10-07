#!/usr/bin/env python3
"""Short non-reportable training smoke test for the shared AMRL pipeline."""

from __future__ import annotations

import argparse
import json

import torch

from amrl_fast.agent import AMRLAgent
from amrl_fast.envs import make_env
from amrl_fast.rebuilt_lstm import RebuiltLSTMBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="Hopper-v2")
    parser.add_argument("--timesteps", type=int, default=10_240)
    parser.add_argument("--adaptation-interval", type=int, default=2_048)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env, resolved = make_env(
        args.environment, seed=0, allow_version_fallback=True
    )
    try:
        agent = AMRLAgent(
            env,
            device=device,
            seed=0,
            use_fast=False,
            backend=RebuiltLSTMBackend(),
            adaptation_interval=args.adaptation_interval,
            warmup_steps=args.adaptation_interval,
            transition_steps=args.adaptation_interval,
        )
        history = agent.train(args.timesteps)
        final_return, final_return_std = agent.evaluate(n_episodes=1)
        print(json.dumps({
            "status": "smoke_only_not_reportable",
            "requested_environment": args.environment,
            "resolved_environment": resolved,
            "timesteps": args.timesteps,
            "paired_meta_windows": len(history.reward_deltas),
            "outer_ppo_updates_exercised": len(history.reward_deltas) >= 4,
            "evaluation_return": final_return,
            "evaluation_return_std": final_return_std,
            "timing": agent.telemetry.timing.to_dict(),
        }, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
