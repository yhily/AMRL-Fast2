#!/usr/bin/env python3
"""Create every continuous-control benchmark once before a training budget."""

from __future__ import annotations

import json

from amrl_fast.envs import make_env


def main() -> None:
    for requested in ("HalfCheetah-v2", "Hopper-v2", "Walker2d-v2"):
        env, resolved = make_env(requested, seed=0, allow_version_fallback=True)
        try:
            observation, _ = env.reset(seed=0)
            action = env.action_space.sample()
            next_observation, reward, terminated, truncated, _ = env.step(action)
            print(json.dumps({
                "requested": requested,
                "resolved": resolved,
                "observation_shape": list(observation.shape),
                "next_observation_shape": list(next_observation.shape),
                "action_shape": list(action.shape),
                "finite_reward": bool(float("-inf") < float(reward) < float("inf")),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }, sort_keys=True))
        finally:
            env.close()


if __name__ == "__main__":
    main()
