#!/usr/bin/env python3
"""Run only the paired AMRL-Fast arm of the frozen heavy CartPole protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amrl_fast.paper_constrained_heavy import run_cartpole_seed0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/paper_constrained_heavy_cartpole_seed0_fast")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--paired-initialisation", required=True)
    args = parser.parse_args()
    result = run_cartpole_seed0(
        Path(args.output),
        seed=0,
        device_name=args.device,
        use_fast_execution=True,
        paired_initialisation_path=args.paired_initialisation,
        paired_initialisation_mode="load",
    )
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "method": result["method"],
        "main_env_steps": result["counters"]["main_env_steps"],
        "auxiliary_probe_training_steps": result["counters"]["auxiliary_probe_training_steps"],
        "main_final_return": result["main_final_return"],
        "training_evaluation_wall_clock": result["timing_boundaries"]["training_evaluation_wall_clock"],
        "lifecycle_wall_clock": result["timing_boundaries"]["lifecycle_wall_clock"],
    }, indent=2))


if __name__ == "__main__":
    main()
