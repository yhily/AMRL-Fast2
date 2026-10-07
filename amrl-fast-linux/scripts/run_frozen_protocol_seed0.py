#!/usr/bin/env python3
"""Run one paired AMRL/AMRL-Fast seed under the frozen 100k protocol.

The protocol intentionally records total lifecycle, training+evaluation,
offline calibration, checkpoint evaluation, and logging separately.  It is a
development validation runner, not a five-seed manuscript-result generator.
"""

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
    )


def trial_payload(trial, history) -> dict:
    return {
        "total_lifecycle_wall_clock": trial.total_lifecycle_wall_clock_time,
        "training_evaluation_wall_clock": trial.training_evaluation_wall_clock_time,
        "offline_calibration_time": trial.offline_kernel_calibration_admission_time,
        "timing": trial.timing,
        "final_return": trial.final_return,
        "final_return_std": trial.final_return_std,
        "discrete_mi_bits": trial.discrete_mi_bits,
        "stability": trial.stability,
        "backend": trial.backend,
        "checkpoint_evaluations": history.checkpoint_evaluations,
        "reward_curve": history.mean_reward_curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", choices=("CartPole-v1", "MountainCar-v0"), required=True)
    parser.add_argument("--label", default="frozen_protocol_r1")
    args = parser.parse_args()
    config = config_for(args.environment)
    baseline, baseline_id, baseline_history = run_seed_with_history(config, "amrl", 0)
    fast, fast_id, fast_history = run_seed_with_history(
        ExperimentConfig(**{**config.__dict__, "enable_rollout_backend": True}),
        "amrl_fast",
        0,
    )
    if baseline_id != fast_id:
        raise RuntimeError(f"Environment mismatch: {baseline_id} != {fast_id}")
    payload = {
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
            "primary_speedup_scope": "total_lifecycle_wall_clock",
        },
        "AMRL": trial_payload(baseline, baseline_history),
        "AMRL_Fast": trial_payload(fast, fast_history),
        "total_lifecycle_speedup": (
            baseline.total_lifecycle_wall_clock_time - fast.total_lifecycle_wall_clock_time
        ) / baseline.total_lifecycle_wall_clock_time,
        "training_evaluation_speedup": (
            baseline.training_evaluation_wall_clock_time - fast.training_evaluation_wall_clock_time
        ) / baseline.training_evaluation_wall_clock_time,
    }
    target = ROOT / "artifacts" / "frozen_protocol" / f"{args.environment.lower().replace('-', '_')}_{args.label}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
