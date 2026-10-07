#!/usr/bin/env python3
"""Single-seed, full-budget CartPole screen for the admitted base-agent backend.

This is deliberately development-only.  It compares the unchanged AMRL path
with AMRL-Fast using the already admitted native LSTM and the newly admitted
cached fused rollout MLP.  The result records calibration separately from the
identical training+evaluation boundary and does not start a five-seed study.
"""
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrl_fast.experiment import ExperimentConfig, run_seed_with_history


def _config(environment: str, *, use_base_backend: bool, adam_backend: str) -> ExperimentConfig:
    return ExperimentConfig(
        environment=environment,
        seeds=[0],
        total_timesteps=100_000,
        # Public train_meta.py trigger: t > 5000 and t % 5000 == 0.
        adaptation_interval=5_000,
        controller_start_after_steps=5_000,
        warmup_steps=5_000,
        transition_steps=5_000,
        controller_hidden_size=256,
        controller_batched_replay=False,
        enable_rollout_backend=use_base_backend,
        enable_foreach_adam=(adam_backend == "foreach"),
        enable_fused_adam=(adam_backend == "fused"),
        evaluation_episodes=20,
        device="auto",
        development_only=True,
        require_optimized_backend=True,
    )


def _meta_execution(trial) -> float:
    return sum(
        trial.timing[key]
        for key in ("meta_learner_forward_time", "meta_learner_backward_time", "hyperparameter_update_time")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="CartPole-v1")
    parser.add_argument("--adam-backend", choices=("standard", "foreach", "fused"), default="standard")
    parser.add_argument("--label", default="cuda_graph_screening")
    arguments = parser.parse_args()
    environment = arguments.environment
    baseline, baseline_id, baseline_history = run_seed_with_history(
        _config(environment, use_base_backend=False, adam_backend="standard"), "amrl", 0
    )
    fast, fast_id, fast_history = run_seed_with_history(
        _config(environment, use_base_backend=True, adam_backend=arguments.adam_backend), "amrl_fast", 0
    )
    if baseline_id != fast_id:
        raise RuntimeError(f"Environment mismatch: {baseline_id} != {fast_id}")
    baseline_meta = _meta_execution(baseline)
    fast_meta = _meta_execution(fast)
    payload = {
        "development_only": True,
        "paper_result_eligible": False,
        "environment": environment,
        "resolved_environment_id": baseline_id,
        "seed": 0,
        "total_timesteps": 100_000,
        "adaptation_interval": 5_000,
        "controller_start_after_steps": 5_000,
        "fast_adam_backend": arguments.adam_backend,
        "training_evaluation_speedup": (
            baseline.training_evaluation_wall_clock_time - fast.training_evaluation_wall_clock_time
        ) / baseline.training_evaluation_wall_clock_time,
        "AMRL": {
            "training_evaluation_wall_clock": baseline.training_evaluation_wall_clock_time,
            "total_lifecycle_wall_clock": baseline.total_lifecycle_wall_clock_time,
            "offline_calibration_time": baseline.offline_kernel_calibration_admission_time,
            "base_agent_execution_time": baseline.timing["base_agent_training_time"],
            "meta_execution_time": baseline_meta,
            "final_return": baseline.final_return,
            "discrete_mi_bits": baseline.discrete_mi_bits,
            "stability": baseline.stability,
            "reward_curve": baseline_history.mean_reward_curve,
        },
        "AMRL_Fast": {
            "training_evaluation_wall_clock": fast.training_evaluation_wall_clock_time,
            "total_lifecycle_wall_clock": fast.total_lifecycle_wall_clock_time,
            "offline_calibration_time": fast.offline_kernel_calibration_admission_time,
            "base_agent_execution_time": fast.timing["base_agent_training_time"],
            "meta_execution_time": fast_meta,
            "final_return": fast.final_return,
            "discrete_mi_bits": fast.discrete_mi_bits,
            "stability": fast.stability,
            "meta_backend": {key: value for key, value in fast.backend.items() if key not in {"base_agent_rollout", "base_agent_rollout_runtime"}},
            "base_agent_backend_admission": fast.backend["base_agent_rollout"],
            "base_agent_backend_runtime": fast.backend["base_agent_rollout_runtime"],
            "reward_curve": fast_history.mean_reward_curve,
        },
    }
    safe_name = environment.lower().replace(" ", "_").replace("-", "_")
    label = arguments.label.strip().replace(" ", "_").replace("/", "_")
    target = ROOT / "artifacts" / "base_agent_hotspots" / f"{safe_name}_seed0_{label}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
