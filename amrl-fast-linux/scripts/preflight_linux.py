#!/usr/bin/env python3
"""Validate the Linux server before consuming a five-seed experiment budget."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrl_fast.envs import make_env
from amrl_fast.experiment import ExperimentConfig
from amrl_fast.rebuilt_lstm import RebuiltLSTMBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--allow-version-fallback", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = ExperimentConfig.from_mapping(yaml.safe_load(handle))
    if args.allow_version_fallback:
        config.allow_version_fallback = True
    if args.device is not None:
        config.device = args.device
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    env, resolved = make_env(
        config.environment,
        seed=config.seeds[0],
        allow_version_fallback=config.allow_version_fallback,
    )
    try:
        observation, _ = env.reset()
        print(f"requested_environment={config.environment}")
        print(f"resolved_environment={resolved}")
        print(f"observation_shape={getattr(env.observation_space, 'shape', None)}")
        print(f"action_space={env.action_space}")
        print(f"device={device}; cuda_available={torch.cuda.is_available()}")
    finally:
        env.close()
    backend = RebuiltLSTMBackend()
    report = backend.calibrate(device)
    print("backend_report=" + json.dumps(backend.to_dict(), sort_keys=True))
    print(f"optimized_backend_admitted={report.admitted}")
    print(f"backend_reason={report.reason}")
    if config.require_optimized_backend and not report.admitted:
        raise SystemExit("Preflight failed: Fast result cannot be reported without an admitted backend.")


if __name__ == "__main__":
    main()
