#!/usr/bin/env python3
"""Screen adaptive-execution intensity without changing AMRL objectives.

This is explicitly a workload stress experiment, not a replacement for the
paper's fixed-cadence benchmark.  Only the controller trigger interval varies.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrl_fast.experiment import ExperimentConfig, run_experiment

INTERVALS = (128, 256, 512, 1000, 5000, 10000)
ENVIRONMENTS = ("CartPole-v1", "MountainCar-v0", "BSuite Memory")
TOTAL_TIMESTEPS = 100_000


def _meta(trial: dict) -> float:
    t = trial["timing"]
    return t["meta_learner_forward_time"] + t["meta_learner_backward_time"] + t["hyperparameter_update_time"]


def _counts(interval: int) -> tuple[int, int]:
    # Trigger is t > interval and t % interval == 0.  The first action occurs
    # at 2*interval.  One controller PPO update occurs per four delayed pairs.
    actions = sum(1 for t in range(interval, TOTAL_TIMESTEPS + 1, interval) if t > interval)
    return actions, (actions - 1) // 4


def main() -> None:
    all_results = []
    for environment in ENVIRONMENTS:
        for interval in INTERVALS:
            config = ExperimentConfig(
                environment=environment,
                seeds=[0],
                total_timesteps=TOTAL_TIMESTEPS,
                adaptation_interval=interval,
                controller_start_after_steps=interval,
                # Fixed guard schedule; only adaptation workload varies.
                warmup_steps=5_000,
                transition_steps=5_000,
                evaluation_episodes=20,
                device="auto",
                output_dir=str(ROOT / "results_adaptive_execution_sweep" / f"interval_{interval}"),
                development_only=True,
                require_optimized_backend=True,
            )
            result = run_experiment(config)
            runtime = result["runtime"]
            trials = {x["method"]: x for x in runtime["seed_trials"]}
            base, fast = trials["amrl"], trials["amrl_fast"]
            base_meta, fast_meta = _meta(base), _meta(fast)
            actions, optimizer_updates = _counts(interval)
            report = fast["backend"]
            row = {
                "stress_experiment": True,
                "environment": environment,
                "resolved_environment_ids": runtime["resolved_environment_ids"],
                "seed": 0,
                "total_timesteps": TOTAL_TIMESTEPS,
                "adaptation_interval": interval,
                "controller_start_after_steps": interval,
                "controller_action_count": actions,
                "controller_optimizer_update_count": optimizer_updates,
                "AMRL_training_time": base["training_evaluation_wall_clock_time"],
                "AMRL_Fast_training_time": fast["training_evaluation_wall_clock_time"],
                "training_speedup": (base["training_evaluation_wall_clock_time"] - fast["training_evaluation_wall_clock_time"]) / base["training_evaluation_wall_clock_time"],
                "AMRL_meta_execution_time": base_meta,
                "AMRL_Fast_meta_execution_time": fast_meta,
                "AMRL_meta_runtime_share": base_meta / base["training_evaluation_wall_clock_time"],
                "AMRL_Fast_meta_runtime_share": fast_meta / fast["training_evaluation_wall_clock_time"],
                "kernel_forward_backward_speedup": report["measured_speedup"],
                "kernel_admitted": report["admitted"],
                "output_correlation": report["output_correlation"],
                "gradient_correlation": report["gradient_correlation"],
                "fallback_count": fast["stability"]["fallback_count"],
                "final_return_AMRL": base["final_return"],
                "final_return_AMRL_Fast": fast["final_return"],
                "offline_calibration_time": fast["offline_kernel_calibration_admission_time"],
                "total_lifecycle_time_AMRL": base["total_lifecycle_wall_clock_time"],
                "total_lifecycle_time_AMRL_Fast": fast["total_lifecycle_wall_clock_time"],
            }
            target = Path(config.output_dir) / environment / "stress_screening_summary.json"
            target.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
            all_results.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    root = ROOT / "results_adaptive_execution_sweep"
    root.mkdir(parents=True, exist_ok=True)
    (root / "complete.json").write_text(json.dumps({"stress_experiment": True, "results": all_results}, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
