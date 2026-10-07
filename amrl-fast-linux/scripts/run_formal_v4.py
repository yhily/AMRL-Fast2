#!/usr/bin/env python3
"""Run the approved v4 formal extension sequentially on one GPU.

The two environments are deliberately sequential: overlapping them would
change the wall-clock resource budget used by the manuscript's timing metric.
``run_experiment`` checkpoints every completed paired seed in
``results/<environment>/seed_trials.json`` and can resume after interruption.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrl_fast.experiment import ExperimentConfig, run_experiment


def _load(path: Path) -> ExperimentConfig:
    with path.open("r", encoding="utf-8") as handle:
        return ExperimentConfig.from_mapping(yaml.safe_load(handle))


def main() -> None:
    configs = (ROOT / "configs" / "hopper-v4.yaml", ROOT / "configs" / "walker2d-v4.yaml")
    summaries = []
    for path in configs:
        config = _load(path)
        if config.development_only:
            raise RuntimeError(f"Formal runner refuses development configuration: {path}")
        result = run_experiment(config)
        summaries.append({
            "environment": config.environment,
            "results_directory": str(Path(config.output_dir) / config.environment),
            "primary_total_lifecycle_speedup": result["runtime"]["speedup"],
        })
        print(json.dumps(summaries[-1], sort_keys=True), flush=True)
    (ROOT / "results" / "formal_v4_run_complete.json").write_text(
        json.dumps({"completed": summaries}, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
