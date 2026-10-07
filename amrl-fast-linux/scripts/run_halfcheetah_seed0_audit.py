#!/usr/bin/env python3
"""Run the frozen, strictly paired HalfCheetah-v4 seed-0 audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrl_fast.experiment import ExperimentConfig
from amrl_fast.halfcheetah_audit import run_halfcheetah_seed0_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = ExperimentConfig.from_mapping(yaml.safe_load(handle))
    if args.device is not None:
        config.device = args.device
    result = run_halfcheetah_seed0_audit(config, args.output_dir)
    print(
        "HalfCheetah seed-0 audit complete: "
        f"train+eval speedup={result['paired_efficiency']['training_evaluation_speedup']:.4%}; "
        f"acceptance={result['acceptance']['all_pass']}"
    )


if __name__ == "__main__":
    main()

