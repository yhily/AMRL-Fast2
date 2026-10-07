#!/usr/bin/env python3
"""Run the frozen Hopper/Walker protocol with complete per-seed audit logs.

The experiment implementation and Fast backend are unchanged.  This runner
only targets a new output directory so it cannot overwrite prior aggregate
results and can resume after every completed paired seed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrl_fast.experiment import ExperimentConfig, run_experiment


def main() -> None:
    configs = (
        ROOT / "configs" / "hopper-v4-audited-r3.yaml",
        ROOT / "configs" / "walker2d-v4-audited-r3.yaml",
    )
    completed = []
    for path in configs:
        config = ExperimentConfig.from_mapping(yaml.safe_load(path.read_text(encoding="utf-8")))
        result = run_experiment(config)
        completed.append({
            "environment": config.environment,
            "output_dir": config.output_dir,
            "training_evaluation_speedup": result["runtime"]["training_evaluation_speedup"],
        })
        print(json.dumps(completed[-1], sort_keys=True), flush=True)
    marker = ROOT / "results" / "audited_continuous_r3" / "formal_v4_audited_r3_complete.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"completed": completed}, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
