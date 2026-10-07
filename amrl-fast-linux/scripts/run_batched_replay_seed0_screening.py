#!/usr/bin/env python3
"""Seed-zero training screen for the semantically verified batched replay path.

AMRL retains the original sequential outer replay.  AMRL-Fast uses the same
four transitions, four PPO epochs, loss, optimizer, and trigger cadence, but
replays independent stored states as B=4 once per epoch.  This is explicitly
development-only and never starts a five-seed experiment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrl_fast.experiment import ExperimentConfig, run_seed_with_history


ENVIRONMENTS = ("CartPole-v1", "MountainCar-v0", "BSuite Memory")
TOTAL_TIMESTEPS = 100_000
ADAPTATION_INTERVAL = 5_000


def _config(environment: str, *, batched: bool) -> ExperimentConfig:
    return ExperimentConfig(
        environment=environment,
        seeds=[0],
        total_timesteps=TOTAL_TIMESTEPS,
        adaptation_interval=ADAPTATION_INTERVAL,
        controller_start_after_steps=ADAPTATION_INTERVAL,
        controller_hidden_size=256,
        controller_batched_replay=batched,
        warmup_steps=5_000,
        transition_steps=5_000,
        evaluation_episodes=20,
        device="auto",
        output_dir=str(ROOT / "results_batched_replay_seed0"),
        development_only=True,
        require_optimized_backend=True,
    )


def _meta(trial) -> dict:
    timing = trial.timing
    return {
        "meta_forward_time": timing["meta_learner_forward_time"],
        "meta_backward_time": timing["meta_learner_backward_time"],
        "hyperparameter_update_time": timing["hyperparameter_update_time"],
        "meta_execution_time": (
            timing["meta_learner_forward_time"]
            + timing["meta_learner_backward_time"]
            + timing["hyperparameter_update_time"]
        ),
    }


def main() -> None:
    summaries = []
    for environment in ENVIRONMENTS:
        baseline, resolved_base, history_base = run_seed_with_history(
            _config(environment, batched=False), "amrl", seed=0
        )
        fast, resolved_fast, history_fast = run_seed_with_history(
            _config(environment, batched=True), "amrl_fast", seed=0
        )
        if resolved_base != resolved_fast:
            raise RuntimeError("AMRL and AMRL-Fast resolved different environments.")
        base_meta, fast_meta = _meta(baseline), _meta(fast)
        row = {
            "development_only": True,
            "paper_result_eligible": False,
            "environment": environment,
            "resolved_environment_id": resolved_base,
            "seed": 0,
            "total_timesteps": TOTAL_TIMESTEPS,
            "adaptation_interval": ADAPTATION_INTERVAL,
            "controller_start_after_steps": ADAPTATION_INTERVAL,
            "controller_hidden_size": 256,
            "AMRL_replay_execution": "sequential",
            "AMRL_Fast_replay_execution": "batched",
            "stored_transitions_per_outer_update": 4,
            "outer_ppo_epochs": 4,
            "AMRL_controller_replay_calls_per_outer_update": 16,
            "AMRL_Fast_controller_replay_calls_per_outer_update": 4,
            "AMRL_training_evaluation_time": baseline.training_evaluation_wall_clock_time,
            "AMRL_Fast_training_evaluation_time": fast.training_evaluation_wall_clock_time,
            "training_evaluation_speedup": (
                baseline.training_evaluation_wall_clock_time - fast.training_evaluation_wall_clock_time
            ) / baseline.training_evaluation_wall_clock_time,
            "AMRL_total_lifecycle_time": baseline.total_lifecycle_wall_clock_time,
            "AMRL_Fast_total_lifecycle_time": fast.total_lifecycle_wall_clock_time,
            "AMRL_Fast_offline_calibration_time": fast.offline_kernel_calibration_admission_time,
            "AMRL": {
                **_meta(baseline),
                "meta_runtime_share": base_meta["meta_execution_time"] / baseline.training_evaluation_wall_clock_time,
                "final_return": baseline.final_return,
                "discrete_mi_bits": baseline.discrete_mi_bits,
                "timing": baseline.timing,
                "stability": baseline.stability,
            },
            "AMRL_Fast": {
                **_meta(fast),
                "meta_runtime_share": fast_meta["meta_execution_time"] / fast.training_evaluation_wall_clock_time,
                "final_return": fast.final_return,
                "discrete_mi_bits": fast.discrete_mi_bits,
                "timing": fast.timing,
                "stability": fast.stability,
                "kernel": fast.backend,
            },
            "reward_curve_AMRL": history_base.mean_reward_curve,
            "reward_curve_AMRL_Fast": history_fast.mean_reward_curve,
        }
        target = ROOT / "results_batched_replay_seed0" / environment / "batched_replay_screening_summary.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
        summaries.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    root = ROOT / "results_batched_replay_seed0"
    root.mkdir(parents=True, exist_ok=True)
    (root / "complete.json").write_text(json.dumps({"results": summaries}, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
