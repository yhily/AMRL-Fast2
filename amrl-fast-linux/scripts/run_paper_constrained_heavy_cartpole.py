#!/usr/bin/env python3
"""Run the single seed=0 paper-constrained heavy AMRL baseline audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Allow direct execution from ``scripts/`` without requiring installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amrl_fast.paper_constrained_heavy import run_cartpole_seed0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/paper_constrained_heavy_cartpole_seed0")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--paired-initialisation", default=None)
    parser.add_argument("--paired-mode", default="none", choices=("none", "write", "load"))
    args = parser.parse_args()
    result = run_cartpole_seed0(
        Path(args.output), seed=0, device_name=args.device,
        paired_initialisation_path=args.paired_initialisation,
        paired_initialisation_mode=args.paired_mode,
    )
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "main_env_steps": result["counters"]["main_env_steps"],
        "auxiliary_probe_training_steps": result["counters"]["auxiliary_probe_training_steps"],
        "outer_optimizer_step_count": result["counters"]["outer_optimizer_step_count"],
        "main_final_return": result["main_final_return"],
        "pure_meta_compute_share": result["shares"]["pure_meta_compute_share"],
        "outer_loop_total_share": result["shares"]["outer_loop_total_share"],
    }, indent=2))


if __name__ == "__main__":
    main()
