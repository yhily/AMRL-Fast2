#!/usr/bin/env python3
"""Screen controller workload size under the original public trigger cadence.

This development-only experiment changes only the meta-controller LSTM hidden
size.  The environment, 100k-step training budget, controller objective, PPO
update rule, 5,000-step trigger condition, kernel admission criteria, and
fallback policy remain unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrl_fast.experiment import ExperimentConfig, run_experiment


ENVIRONMENTS = ("CartPole-v1", "MountainCar-v0", "BSuite Memory")
HIDDEN_SIZES = (64, 128, 256)
TOTAL_TIMESTEPS = 100_000
ADAPTATION_INTERVAL = 5_000


def _meta_components(trial: dict) -> tuple[float, float, float, float]:
    timing = trial["timing"]
    forward = timing["meta_learner_forward_time"]
    backward = timing["meta_learner_backward_time"]
    update = timing["hyperparameter_update_time"]
    return forward, backward, update, forward + backward + update


def main() -> None:
    rows: list[dict] = []
    for environment in ENVIRONMENTS:
        for hidden_size in HIDDEN_SIZES:
            config = ExperimentConfig(
                environment=environment,
                seeds=[0],
                total_timesteps=TOTAL_TIMESTEPS,
                adaptation_interval=ADAPTATION_INTERVAL,
                controller_start_after_steps=ADAPTATION_INTERVAL,
                controller_hidden_size=hidden_size,
                warmup_steps=5_000,
                transition_steps=5_000,
                evaluation_episodes=20,
                device="auto",
                output_dir=str(
                    ROOT / "results_controller_workload_sweep" / f"hidden_{hidden_size}"
                ),
                development_only=True,
                require_optimized_backend=True,
            )
            result = run_experiment(config)
            runtime = result["runtime"]
            trials = {trial["method"]: trial for trial in runtime["seed_trials"]}
            baseline, fast = trials["amrl"], trials["amrl_fast"]
            base_forward, base_backward, base_update, base_meta = _meta_components(baseline)
            fast_forward, fast_backward, fast_update, fast_meta = _meta_components(fast)
            report = fast["backend"]
            row = {
                "stress_experiment": True,
                "stress_factor": "meta_controller_hidden_size",
                "environment": environment,
                "resolved_environment_ids": runtime["resolved_environment_ids"],
                "seed": 0,
                "total_timesteps": TOTAL_TIMESTEPS,
                "adaptation_interval": ADAPTATION_INTERVAL,
                "controller_start_after_steps": ADAPTATION_INTERVAL,
                "controller_hidden_size": hidden_size,
                "AMRL_meta_forward_time": base_forward,
                "AMRL_meta_backward_time": base_backward,
                "AMRL_hyperparameter_update_time": base_update,
                "AMRL_meta_execution_time": base_meta,
                "AMRL_meta_runtime_share": base_meta / baseline["training_evaluation_wall_clock_time"],
                "AMRL_Fast_meta_forward_time": fast_forward,
                "AMRL_Fast_meta_backward_time": fast_backward,
                "AMRL_Fast_hyperparameter_update_time": fast_update,
                "AMRL_Fast_meta_execution_time": fast_meta,
                "AMRL_Fast_meta_runtime_share": fast_meta / fast["training_evaluation_wall_clock_time"],
                "kernel_admitted": report["admitted"],
                "kernel_forward_backward_speedup": report["measured_speedup"],
                "output_correlation": report["output_correlation"],
                "gradient_correlation": report["gradient_correlation"],
                "fallback_count": fast["stability"]["fallback_count"],
                "AMRL_training_time": baseline["training_evaluation_wall_clock_time"],
                "AMRL_Fast_training_time": fast["training_evaluation_wall_clock_time"],
                "training_speedup": (
                    baseline["training_evaluation_wall_clock_time"]
                    - fast["training_evaluation_wall_clock_time"]
                ) / baseline["training_evaluation_wall_clock_time"],
                "final_return_AMRL": baseline["final_return"],
                "final_return_AMRL_Fast": fast["final_return"],
                "offline_calibration_time": fast["offline_kernel_calibration_admission_time"],
            }
            target = Path(config.output_dir) / environment / "controller_workload_summary.json"
            target.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    root = ROOT / "results_controller_workload_sweep"
    root.mkdir(parents=True, exist_ok=True)
    (root / "complete.json").write_text(
        json.dumps({"stress_experiment": True, "results": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
