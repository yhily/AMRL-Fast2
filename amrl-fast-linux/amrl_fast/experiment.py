"""Experiment runner that writes exactly the manuscript's existing metrics."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .agent import AMRLAgent, TrainingHistory
from .envs import get_environment_spec, make_env
from .rebuilt_lstm import RebuiltLSTMBackend
from .statistics import (
    SeedTrial,
    bootstrap_mean_ci,
    discrete_mutual_information,
    information_diagnostics,
    paired_effect_size,
)
from .telemetry import synchronized_time


@dataclass
class ExperimentConfig:
    environment: str
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    total_timesteps: int = 100_000
    adaptation_interval: int = 10_000
    warmup_steps: int = 10_000
    transition_steps: int = 10_000
    checkpoint_interval: int = 10_000
    controller_start_after_steps: int = 0
    controller_hidden_size: int = 256
    controller_batched_replay: bool = False
    enable_rollout_backend: bool = False
    enable_foreach_adam: bool = False
    enable_fused_adam: bool = False
    evaluation_episodes: int = 20
    device: str = "auto"
    output_dir: str = "results"
    allow_version_fallback: bool = False
    development_only: bool = False
    fast_backend_name: str = "AMRL-Fast-Rebuilt-Kernel"
    require_optimized_backend: bool = True

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "ExperimentConfig":
        environment = mapping.get("environment", {})
        training = mapping.get("training", {})
        evaluation = mapping.get("evaluation", {})
        kernel = mapping.get("fast_backend", {})
        return cls(
            environment=environment["env_name"],
            seeds=list(mapping.get("seeds", [0, 1, 2, 3, 4])),
            total_timesteps=int(training.get("total_timesteps", 100_000)),
            adaptation_interval=int(training.get("adaptation_interval", 10_000)),
            warmup_steps=int(training.get("warmup_steps", 10_000)),
            transition_steps=int(training.get("transition_steps", 10_000)),
            checkpoint_interval=int(training.get("checkpoint_interval", 10_000)),
            controller_start_after_steps=int(training.get("controller_start_after_steps", 0)),
            controller_hidden_size=int(training.get("controller_hidden_size", 256)),
            controller_batched_replay=bool(training.get("controller_batched_replay", False)),
            enable_rollout_backend=bool(training.get("enable_rollout_backend", False)),
            enable_foreach_adam=bool(training.get("enable_foreach_adam", False)),
            enable_fused_adam=bool(training.get("enable_fused_adam", False)),
            evaluation_episodes=int(evaluation.get("episodes", 20)),
            device=str(mapping.get("device", "auto")),
            output_dir=str(mapping.get("output_dir", "results")),
            allow_version_fallback=bool(environment.get("allow_version_fallback", False)),
            development_only=bool(mapping.get("development_only", False)),
            fast_backend_name=str(kernel.get("name", "AMRL-Fast-Rebuilt-Kernel")),
            require_optimized_backend=bool(kernel.get("require_optimized_backend", True)),
        )


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _mean(items: list[float]) -> float:
    return float(np.mean(items)) if items else float("nan")


def _aggregate_timing(trials: list[SeedTrial]) -> dict:
    fields = [
        "base_agent_training_time", "meta_learner_forward_time",
        "meta_learner_backward_time", "hyperparameter_update_time",
        "evaluation_time", "logging_time",
    ]
    return {name: _mean([trial.timing[name] for trial in trials]) for name in fields}


def _aggregate_stability(trials: list[SeedTrial]) -> dict:
    count = int(sum(trial.stability["fallback_count"] for trial in trials))
    eligible = int(sum(trial.stability["eligible_dispatch_count"] for trial in trials))
    return {
        "fallback_count": count,
        "fallback_ratio": count / eligible if eligible else 0.0,
        "nan_inf_count": int(sum(trial.stability["nan_inf_count"] for trial in trials)),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


TIMING_BOUNDARY = {
    "start": "Immediately after the seeded environment has been constructed and before AMRLAgent construction.",
    "end": "After training, all scheduled checkpoint evaluations, and final evaluation complete, following CUDA synchronization when CUDA is used.",
    "common_scope": "AMRLAgent construction, training, and evaluation are timed for both methods.",
    "fast_only_scope": "The Fast path additionally includes per-seed kernel calibration/admission before agent construction.",
}

# The manuscript's end-to-end wall-clock claim is a lifecycle claim: the Fast
# path does not get to omit its per-seed calibration/admission work.  The
# separate fields below make that cost auditable without changing the primary
# speedup denominator or numerator after the fact.
TIMING_ACCOUNTING = {
    "primary_speedup_metric": "total_lifecycle_wall_clock_time",
    "primary_speedup_formula": "(T_AMRL_total_lifecycle - T_AMRL_Fast_total_lifecycle) / T_AMRL_total_lifecycle",
    "offline_kernel_calibration_admission_time": "Per-seed candidate timing, fidelity screening, and admission decision before AMRLAgent construction; zero for AMRL.",
    "training_evaluation_wall_clock_time": "AMRLAgent construction, training, and evaluation after any Fast calibration/admission step.",
    "total_lifecycle_wall_clock_time": "Offline kernel calibration/admission plus training/evaluation; this is the manuscript-compatible primary wall-clock scope.",
}


def run_seed_with_history(config: ExperimentConfig, method: str, seed: int) -> tuple[SeedTrial, str, TrainingHistory]:
    """Execute exactly one seed and retain raw history for development audits.

    ``run_seed`` below discards this history for formal aggregate JSON, while
    the preflight validator consumes it to verify the MI pairing and reward
    curve without creating a second training implementation.
    """
    device = _device(config.device)
    env, resolved_env_id = make_env(
        config.environment,
        seed=seed,
        allow_version_fallback=config.allow_version_fallback,
    )
    evaluation_env, resolved_evaluation_env_id = make_env(
        config.environment,
        seed=seed + 1_000_000,
        allow_version_fallback=config.allow_version_fallback,
    )
    if resolved_evaluation_env_id != resolved_env_id:
        env.close()
        evaluation_env.close()
        raise RuntimeError("Training and evaluation environments resolved different IDs.")
    backend = RebuiltLSTMBackend()
    # The Fast wall clock starts before calibration.  This prevents offline
    # selection/compilation work from being silently excluded from an
    # end-to-end speed claim.
    start = time.perf_counter()
    calibration_time = 0.0
    if method == "amrl_fast":
        calibration_start = time.perf_counter()
        report = backend.calibrate(device, hidden_size=config.controller_hidden_size)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        calibration_time = time.perf_counter() - calibration_start
        if config.require_optimized_backend and not report.admitted:
            env.close()
            evaluation_env.close()
            raise RuntimeError(
                "AMRL-Fast kernel was not admitted. Refusing to label a standard "
                f"fallback run as a speed experiment: {report.reason}"
            )
    try:
        agent = AMRLAgent(
            env,
            evaluation_env=evaluation_env,
            device=device,
            seed=seed,
            use_fast=(method == "amrl_fast"),
            backend=backend,
            adaptation_interval=config.adaptation_interval,
            warmup_steps=config.warmup_steps,
            transition_steps=config.transition_steps,
            controller_start_after_steps=config.controller_start_after_steps,
            controller_hidden_size=config.controller_hidden_size,
            controller_batched_replay=config.controller_batched_replay,
            enable_rollout_backend=(method == "amrl_fast" and config.enable_rollout_backend),
            enable_foreach_adam=(method == "amrl_fast" and config.enable_foreach_adam),
            enable_fused_adam=(method == "amrl_fast" and config.enable_fused_adam),
        )
        # The base-agent candidate is an offline execution-layer admission
        # decision, just like the LSTM candidate.  It is calibrated once after
        # model construction and before the timed training/evaluation phase.
        if method == "amrl_fast" and config.enable_rollout_backend:
            base_calibration_start = time.perf_counter()
            agent.base.admit_rollout_backend()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            calibration_time += time.perf_counter() - base_calibration_start
        checkpoint_count = 0

        def evaluate_checkpoint(step: int) -> tuple | None:
            nonlocal checkpoint_count
            if config.checkpoint_interval <= 0 or step % config.checkpoint_interval != 0:
                return None
            with synchronized_time(device) as timer:
                value = agent.evaluate(config.evaluation_episodes, return_episode_returns=True)
            agent.telemetry.timing.evaluation_time += timer.seconds
            # Store the fixed-protocol checkpoint record independently from
            # evaluation timing.  No I/O or synchronization occurs here.
            started = time.perf_counter()
            checkpoint_count += 1
            agent.telemetry.timing.logging_time += time.perf_counter() - started
            return value

        training_start = time.perf_counter()
        history = agent.train(config.total_timesteps, checkpoint_callback=evaluate_checkpoint)
        if history.checkpoint_evaluations:
            final_return = history.checkpoint_evaluations[-1]["mean_return"]
            final_return_std = history.checkpoint_evaluations[-1]["std_return"]
        else:
            with synchronized_time(device) as timer:
                final_return, final_return_std, episode_returns = agent.evaluate(
                    config.evaluation_episodes, return_episode_returns=True
                )
            agent.telemetry.timing.evaluation_time += timer.seconds
            history.checkpoint_evaluations.append({
                "step": float(config.total_timesteps),
                "mean_return": float(final_return),
                "std_return": float(final_return_std),
                "episode_returns": [float(value) for value in episode_returns],
            })
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_evaluation_time = time.perf_counter() - training_start
        wall_time = time.perf_counter() - start
        trial = SeedTrial(
            method=method,
            seed=seed,
            wall_clock_time=wall_time,
            offline_kernel_calibration_admission_time=calibration_time,
            calibration_time=calibration_time,
            training_evaluation_wall_clock_time=training_evaluation_time,
            total_lifecycle_wall_clock_time=wall_time,
            final_return=final_return,
            final_return_std=final_return_std,
            discrete_mi_bits=discrete_mutual_information(
                history.controller_outputs, history.reward_deltas
            ),
            timing=agent.telemetry.timing.to_dict(),
            stability=agent.telemetry.stability.to_dict(),
            backend={
                **backend.to_dict(),
                "base_agent_rollout": agent.base.rollout_backend_audit,
                "base_agent_rollout_runtime": (
                    None if agent.base.rollout_backend is None else {
                        "backend": agent.base.rollout_backend.name,
                        "dispatch_count": agent.base.rollout_backend.dispatch_count,
                        "fallback_count": agent.base.rollout_backend.fallback_count,
                        "nan_inf_count": agent.base.rollout_backend.nan_inf_count,
                        "fallback_reasons": dict(agent.base.rollout_backend.fallback_reasons),
                    }
                ),
            },
            audit={
                "checkpoint_evaluations": history.checkpoint_evaluations,
                "learning_curve": history.mean_reward_curve,
                "information_raw_samples": {
                    "controller_outputs": history.controller_outputs,
                    "reward_deltas": history.reward_deltas,
                },
                "information_metrics": information_diagnostics(
                    history.controller_outputs,
                    history.reward_deltas,
                    wall_clock_seconds=training_evaluation_time,
                ),
                "base_agent_execution": agent.base.execution_audit(),
            },
        )
        return trial, resolved_env_id, history
    finally:
        env.close()
        evaluation_env.close()


def run_seed(config: ExperimentConfig, method: str, seed: int) -> tuple[SeedTrial, str]:
    trial, resolved_env_id, _ = run_seed_with_history(config, method, seed)
    return trial, resolved_env_id


def run_experiment(config: ExperimentConfig) -> dict:
    """Run AMRL and AMRL-Fast across paired seeds and write five JSON files."""
    spec = get_environment_spec(config.environment)
    trials: dict[str, list[SeedTrial]] = {"amrl": [], "amrl_fast": []}
    resolved_ids: set[str] = set()
    root = Path(config.output_dir) / spec.name
    checkpoint_path = root / "seed_trials.json"
    completed_seeds: set[int] = set()
    if checkpoint_path.is_file():
        with checkpoint_path.open("r", encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        if checkpoint.get("environment") != spec.name:
            raise RuntimeError(f"Checkpoint environment mismatch at {checkpoint_path}.")
        for pair in checkpoint.get("paired_seed_trials", []):
            seed = int(pair["seed"])
            trials["amrl"].append(SeedTrial(**pair["AMRL"]))
            trials["amrl_fast"].append(SeedTrial(**pair["AMRL_Fast"]))
            resolved_ids.add(str(pair["resolved_environment_id"]))
            completed_seeds.add(seed)
    for seed in config.seeds:
        if seed in completed_seeds:
            continue
        baseline, resolved = run_seed(config, "amrl", seed)
        fast, resolved_fast = run_seed(config, "amrl_fast", seed)
        if resolved != resolved_fast:
            raise RuntimeError("AMRL and AMRL-Fast resolved different environment versions.")
        resolved_ids.add(resolved)
        trials["amrl"].append(baseline)
        trials["amrl_fast"].append(fast)
        checkpoint = {
            "environment": spec.name,
            "requested_seeds": config.seeds,
            "timing_accounting": TIMING_ACCOUNTING,
            "paired_seed_trials": [
                {
                    # Use the seed owned by this checkpointed pair.  The
                    # outer ``baseline`` variable refers to the last loop
                    # iteration and previously mislabeled every saved pair
                    # with that final seed despite the inner trials being
                    # correctly seeded.
                    "seed": item_baseline.seed,
                    "resolved_environment_id": resolved,
                    "AMRL": item_baseline.to_dict(),
                    "AMRL_Fast": item_fast.to_dict(),
                }
                for item_baseline, item_fast in zip(trials["amrl"], trials["amrl_fast"])
            ],
        }
        _write_json(checkpoint_path, checkpoint)

    baseline, fast = trials["amrl"], trials["amrl_fast"]
    base_times = [item.total_lifecycle_wall_clock_time for item in baseline]
    fast_times = [item.total_lifecycle_wall_clock_time for item in fast]
    base_returns = [item.final_return for item in baseline]
    fast_returns = [item.final_return for item in fast]
    base_mi = [item.discrete_mi_bits for item in baseline]
    fast_mi = [item.discrete_mi_bits for item in fast]
    mi_p_value, mi_effect_size = paired_effect_size(base_mi, fast_mi)
    fast_reports = [item.backend for item in fast]
    reportable_fast = all(bool(report.get("admitted")) for report in fast_reports)
    common = {
        "environment": spec.name,
        "resolved_environment_ids": sorted(resolved_ids),
        "seeds": config.seeds,
        "evaluation_episodes_per_seed": config.evaluation_episodes,
        "checkpoint_interval": config.checkpoint_interval,
        "statistical_unit": "random seed",
        "development_only": config.development_only,
        "paper_result_eligible": not config.development_only,
        "end_to_end_timing_boundary": TIMING_BOUNDARY,
        "timing_accounting": TIMING_ACCOUNTING,
        "AMRL_Fast_reportable": reportable_fast,
        "AMRL_Fast_backend_name": config.fast_backend_name,
        "AMRL_Fast_backend_reports": fast_reports,
    }
    runtime = {
        **common,
        "AMRL_total_wall_clock_time": _mean(base_times),
        "AMRL_Fast_total_wall_clock_time": _mean(fast_times),
        "AMRL_total_lifecycle_wall_clock_time": _mean(base_times),
        "AMRL_Fast_total_lifecycle_wall_clock_time": _mean(fast_times),
        "speedup": (_mean(base_times) - _mean(fast_times)) / _mean(base_times),
        "speedup_reportable": reportable_fast and not config.development_only,
        "AMRL_mean_offline_kernel_calibration_admission_time": _mean([item.offline_kernel_calibration_admission_time for item in baseline]),
        "AMRL_Fast_mean_offline_kernel_calibration_admission_time": _mean([item.offline_kernel_calibration_admission_time for item in fast]),
        # Compatibility aliases retained for existing analysis notebooks.
        "AMRL_mean_calibration_time": _mean([item.calibration_time for item in baseline]),
        "AMRL_Fast_mean_calibration_time": _mean([item.calibration_time for item in fast]),
        "AMRL_training_evaluation_wall_clock_time": _mean([item.training_evaluation_wall_clock_time for item in baseline]),
        "AMRL_Fast_training_evaluation_wall_clock_time": _mean([item.training_evaluation_wall_clock_time for item in fast]),
        "training_evaluation_speedup": (
            _mean([item.training_evaluation_wall_clock_time for item in baseline])
            - _mean([item.training_evaluation_wall_clock_time for item in fast])
        ) / _mean([item.training_evaluation_wall_clock_time for item in baseline]),
        "seed_trials": [item.to_dict() for item in baseline + fast],
    }
    performance = {
        **common,
        "AMRL_final_return": _mean(base_returns),
        "AMRL_Fast_final_return": _mean(fast_returns),
        "return_difference": _mean(fast_returns) - _mean(base_returns),
        "seed_trials": [item.to_dict() for item in baseline + fast],
    }
    adaptation = {
        **common,
        "MI_estimator": "equal-frequency joint plug-in mutual information (bits)",
        "MI_AMRL": _mean(base_mi),
        "MI_AMRL_95_bootstrap_CI": bootstrap_mean_ci(base_mi),
        "MI_AMRL_Fast": _mean(fast_mi),
        "MI_AMRL_Fast_95_bootstrap_CI": bootstrap_mean_ci(fast_mi),
        "p_value": mi_p_value,
        "effect_size": mi_effect_size,
        "seed_level_MI_AMRL": base_mi,
        "seed_level_MI_AMRL_Fast": fast_mi,
    }
    profiling = {**common, **_aggregate_timing(baseline)}
    stability = {**common, **_aggregate_stability(fast)}
    _write_json(root / "runtime.json", runtime)
    _write_json(root / "performance.json", performance)
    _write_json(root / "adaptation.json", adaptation)
    _write_json(root / "profiling.json", profiling)
    _write_json(root / "stability.json", stability)
    return {"runtime": runtime, "performance": performance, "adaptation": adaptation, "profiling": profiling, "stability": stability}
