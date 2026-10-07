#!/usr/bin/env python3
"""One-seed 100k extension screen using the already frozen AMRL-Fast stack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrl_fast.experiment import ExperimentConfig, run_seed_with_history


def config_for(environment: str) -> ExperimentConfig:
    return ExperimentConfig(
        environment=environment,
        seeds=[0],
        total_timesteps=100_000,
        adaptation_interval=10_000,
        controller_start_after_steps=0,
        warmup_steps=10_000,
        transition_steps=10_000,
        checkpoint_interval=10_000,
        controller_hidden_size=256,
        controller_batched_replay=True,
        evaluation_episodes=20,
        device="auto",
        development_only=True,
        require_optimized_backend=True,
        enable_rollout_backend=True,
    )


def payload_for(trial, history) -> dict:
    return {
        "total_lifecycle_wall_clock": trial.total_lifecycle_wall_clock_time,
        "training_evaluation_wall_clock": trial.training_evaluation_wall_clock_time,
        "offline_calibration_time": trial.offline_kernel_calibration_admission_time,
        "final_return": trial.final_return,
        "final_return_std": trial.final_return_std,
        "discrete_mi_bits": trial.discrete_mi_bits,
        "timing": trial.timing,
        "stability": trial.stability,
        "backend": trial.backend,
        "checkpoint_evaluations": history.checkpoint_evaluations,
        "reward_curve": history.mean_reward_curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True, choices=("Hopper-v4", "Walker2d-v4"))
    parser.add_argument("--label", default="frozen_stack_screen_r1")
    args = parser.parse_args()
    config = config_for(args.environment)
    baseline, baseline_id, baseline_history = run_seed_with_history(config, "amrl", 0)
    fast, fast_id, fast_history = run_seed_with_history(config, "amrl_fast", 0)
    if baseline_id != fast_id:
        raise RuntimeError(f"Environment mismatch: {baseline_id} != {fast_id}")
    result = {
        "development_only": True,
        "paper_result_eligible": False,
        "environment": args.environment,
        "resolved_environment_id": baseline_id,
        "seed": 0,
        "protocol": {
            "total_timesteps": config.total_timesteps,
            "adaptation_interval": config.adaptation_interval,
            "checkpoint_interval": config.checkpoint_interval,
            "evaluation_episodes_per_checkpoint": config.evaluation_episodes,
            "controller_hidden_size": config.controller_hidden_size,
            "controller_batched_replay": config.controller_batched_replay,
            "adam_backend": "standard",
            "fast_execution_stack": "frozen",
            "primary_speedup_scope": "training_evaluation_wall_clock",
        },
        "AMRL": payload_for(baseline, baseline_history),
        "AMRL_Fast": payload_for(fast, fast_history),
        "training_evaluation_speedup": (
            baseline.training_evaluation_wall_clock_time - fast.training_evaluation_wall_clock_time
        ) / baseline.training_evaluation_wall_clock_time,
        "total_lifecycle_speedup": (
            baseline.total_lifecycle_wall_clock_time - fast.total_lifecycle_wall_clock_time
        ) / baseline.total_lifecycle_wall_clock_time,
    }
    destination = ROOT / "artifacts" / "extension_screening" / (
        f"{args.environment.lower().replace('-', '_')}_{args.label}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
