#!/usr/bin/env python3
"""Run paired AMRL / AMRL-Fast seeds and write manuscript-compatible JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrl_fast.experiment import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument(
        "--allow-version-fallback",
        action="store_true",
        help="Allow v2 request to resolve to Gymnasium's v4 MuJoCo ID; JSON records it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = ExperimentConfig.from_mapping(yaml.safe_load(handle))
    if args.output_dir is not None:
        config.output_dir = str(args.output_dir)
    if args.device is not None:
        config.device = args.device
    if args.allow_version_fallback:
        config.allow_version_fallback = True
    result = run_experiment(config)
    print(
        f"Completed {config.environment}. "
        f"speedup={result['runtime']['speedup']:.4f}; "
        f"results={Path(config.output_dir) / config.environment}"
    )


if __name__ == "__main__":
    main()
