#!/usr/bin/env python3
"""Single-seed reproduction screen using the public 5k controller trigger."""

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


def _meta(trial: dict) -> float:
    timing = trial["timing"]
    return sum(timing[key] for key in (
        "meta_learner_forward_time",
        "meta_learner_backward_time",
        "hyperparameter_update_time",
    ))


def _summary(result: dict, config: ExperimentConfig) -> dict:
    runtime = result["runtime"]
    trials = {item["method"]: item for item in runtime["seed_trials"]}
    base, fast = trials["amrl"], trials["amrl_fast"]
    base_meta, fast_meta = _meta(base), _meta(fast)
    base_training = base["training_evaluation_wall_clock_time"]
    fast_training = fast["training_evaluation_wall_clock_time"]
    report = fast["backend"]
    return {
        "environment": config.environment,
        "resolved_environment_ids": runtime["resolved_environment_ids"],
        "screening_only": True,
        "seed": 0,
        "total_timesteps": config.total_timesteps,
        "adaptation_interval": config.adaptation_interval,
        "controller_start_after_steps": config.controller_start_after_steps,
        "AMRL_training_wall_clock": base_training,
        "AMRL_Fast_training_wall_clock": fast_training,
        "training_speedup": (base_training - fast_training) / base_training,
        "offline_calibration_time": fast["offline_kernel_calibration_admission_time"],
        "AMRL_total_lifecycle_wall_clock": base["total_lifecycle_wall_clock_time"],
        "AMRL_Fast_total_lifecycle_wall_clock": fast["total_lifecycle_wall_clock_time"],
        "AMRL_meta_execution_time": base_meta,
        "AMRL_Fast_meta_execution_time": fast_meta,
        "AMRL_meta_runtime_share": base_meta / base_training,
        "AMRL_Fast_meta_runtime_share": fast_meta / fast_training,
        "kernel_forward_backward_speedup": report["measured_speedup"],
        "kernel_admitted": report["admitted"],
        "output_correlation": report["output_correlation"],
        "gradient_correlation": report["gradient_correlation"],
        "fallback_count": fast["stability"]["fallback_count"],
        "fallback_ratio": fast["stability"]["fallback_ratio"],
    }


def main() -> None:
    paths = (
        ROOT / "configs" / "cartpole-original-screening.yaml",
        ROOT / "configs" / "mountaincar-original-screening.yaml",
        ROOT / "configs" / "bsuite-memory-original-screening.yaml",
    )
    summaries = []
    for path in paths:
        config = _load(path)
        if config.seeds != [0] or not config.development_only:
            raise RuntimeError(f"Expected one-seed development configuration: {path}")
        result = run_experiment(config)
        summary = _summary(result, config)
        root = Path(config.output_dir) / config.environment
        (root / "original_benchmark_screening_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    (ROOT / "results_original_screening" / "complete.json").write_text(
        json.dumps({"screening_only": True, "results": summaries}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
