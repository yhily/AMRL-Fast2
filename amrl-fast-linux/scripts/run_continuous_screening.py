#!/usr/bin/env python3
"""One-seed, 100k-step Box2D profiling screen for AMRL-Fast suitability.

This runner deliberately retains the existing formal adaptation cadence and
does not label its single-seed output as a manuscript result.  It writes the
standard result JSON files and an additional concise screening summary.
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


def _screen_summary(result: dict) -> dict:
    runtime = result["runtime"]
    profiling = result["profiling"]
    report = runtime["AMRL_Fast_backend_reports"][0]
    meta = (
        profiling["meta_learner_forward_time"]
        + profiling["meta_learner_backward_time"]
        + profiling["hyperparameter_update_time"]
    )
    total = runtime["AMRL_training_evaluation_wall_clock_time"]
    share = meta / total if total else float("nan")
    kernel_speedup = report["measured_speedup"]
    # Amdahl upper bound assumes that every timed meta operation receives the
    # calibrated kernel speedup; it is intentionally optimistic.
    max_speedup = 1.0 / ((1.0 - share) + share / kernel_speedup)
    return {
        "environment": runtime["environment"],
        "resolved_environment_ids": runtime["resolved_environment_ids"],
        "screening_only": True,
        "total_training_evaluation_wall_clock_time": total,
        "meta_learner_forward_time": profiling["meta_learner_forward_time"],
        "meta_learner_backward_time": profiling["meta_learner_backward_time"],
        "hyperparameter_update_time": profiling["hyperparameter_update_time"],
        "meta_execution_time": meta,
        "meta_runtime_share": share,
        "native_kernel_forward_backward_speedup": kernel_speedup,
        "theoretical_max_end_to_end_speedup": max_speedup - 1.0,
        "passes_meta_share_screen": share >= 0.02,
        "admission": {
            "backend": report["backend_name"],
            "admitted": report["admitted"],
            "output_correlation": report["output_correlation"],
            "gradient_correlation": report["gradient_correlation"],
            "relative_l2_error": report["relative_l2_error"],
            "gradient_relative_l2_error": report["gradient_relative_l2_error"],
        },
    }


def main() -> None:
    paths = (
        ROOT / "configs" / "lunarlandercontinuous-v3-screening.yaml",
        ROOT / "configs" / "bipedalwalker-v3-screening.yaml",
    )
    summaries = []
    for path in paths:
        config = _load(path)
        if not config.development_only or config.seeds != [0]:
            raise RuntimeError(f"Screening configuration must be one-seed development-only: {path}")
        result = run_experiment(config)
        summary = _screen_summary(result)
        root = Path(config.output_dir) / config.environment
        (root / "screening_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    (ROOT / "results_screening" / "continuous_screening_complete.json").write_text(
        json.dumps({"screening_only": True, "results": summaries}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
