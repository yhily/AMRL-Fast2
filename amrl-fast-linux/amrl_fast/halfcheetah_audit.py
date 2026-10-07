"""Paired, paper-facing HalfCheetah seed audit.

This module deliberately does not reuse the development-only heavy CartPole
runner.  It exercises the normal AMRL training protocol and writes enough raw
state, timing and information data to decide whether a five-seed run is
permitted.  AMRL and AMRL-Fast are started from one serialized initialization
snapshot; only their admitted execution backends differ.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .agent import AMRLAgent, TrainingHistory
from .envs import make_env
from .experiment import ExperimentConfig, _device
from .rebuilt_lstm import RebuiltLSTMBackend
from .statistics import discrete_mutual_information, information_diagnostics
from .telemetry import synchronized_time


AUDIT_TIMING_BOUNDARY = {
    "training_evaluation_start": "Immediately before AMRLAgent.train(), after identical agent construction and paired initialization restoration.",
    "training_evaluation_end": "After the 100k main training budget and all scheduled 20-episode checkpoint evaluations have completed, with CUDA synchronized.",
    "offline_calibration": "Fast candidate fidelity/timing admission before training; reported separately and excluded from the primary training+evaluation speedup.",
    "lifecycle": "offline calibration plus the identically bounded training+evaluation phase.",
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)


def _capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _physics_state(env) -> dict[str, Any]:
    """Capture a fresh, seeded MuJoCo state for the paired-start audit."""
    raw = env.unwrapped
    payload: dict[str, Any] = {
        "class": type(raw).__name__,
        "np_random_state": copy.deepcopy(raw.np_random.bit_generator.state)
        if getattr(raw, "np_random", None) is not None else None,
    }
    data = getattr(raw, "data", None)
    if data is not None and hasattr(data, "qpos") and hasattr(data, "qvel"):
        payload["qpos"] = np.asarray(data.qpos).copy()
        payload["qvel"] = np.asarray(data.qvel).copy()
    return payload


def _restore_physics_state(env, state: dict[str, Any]) -> None:
    raw = env.unwrapped
    if state.get("np_random_state") is not None and getattr(raw, "np_random", None) is not None:
        raw.np_random.bit_generator.state = copy.deepcopy(state["np_random_state"])
    if "qpos" in state and "qvel" in state and hasattr(raw, "set_state"):
        raw.set_state(np.asarray(state["qpos"]), np.asarray(state["qvel"]))


def _state_dict_equal(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if first.keys() != second.keys():
        return False
    for key in first:
        left, right = first[key], second[key]
        if isinstance(left, torch.Tensor):
            if not isinstance(right, torch.Tensor) or not torch.equal(left.detach().cpu(), right.detach().cpu()):
                return False
        elif left != right:
            return False
    return True


def _state_digest(payload: dict[str, Any]) -> str:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _initial_snapshot(agent: AMRLAgent, *, seed: int, environment_id: str) -> dict[str, Any]:
    return {
        "schema": "amrl-fast-paired-initialization-r1",
        "seed": int(seed),
        "resolved_environment_id": environment_id,
        "policy_state": copy.deepcopy(agent.base.model.policy.state_dict()),
        "policy_optimizer_state": copy.deepcopy(agent.base.model.policy.optimizer.state_dict()),
        "controller_state": copy.deepcopy(agent.controller.state_dict()),
        "controller_optimizer_state": copy.deepcopy(agent.controller.optimizer.state_dict()),
        "base_hyperparameters": copy.deepcopy(agent.base.hyperparameters),
        "rng": _capture_rng(),
        "training_environment": _physics_state(agent.base.env),
        "evaluation_environment": _physics_state(agent.evaluation_env),
    }


def _restore_initial_snapshot(agent: AMRLAgent, snapshot: dict[str, Any]) -> dict[str, bool]:
    agent.base.model.policy.load_state_dict(snapshot["policy_state"], strict=True)
    agent.base.model.policy.optimizer.load_state_dict(snapshot["policy_optimizer_state"])
    agent.controller.load_state_dict(snapshot["controller_state"], strict=True)
    agent.controller.optimizer.load_state_dict(snapshot["controller_optimizer_state"])
    agent.base.set_hyperparameters(copy.deepcopy(snapshot["base_hyperparameters"]))
    _restore_physics_state(agent.base.env, snapshot["training_environment"])
    _restore_physics_state(agent.evaluation_env, snapshot["evaluation_environment"])
    _restore_rng(snapshot["rng"])
    return {
        "policy_state_exact": _state_dict_equal(snapshot["policy_state"], agent.base.model.policy.state_dict()),
        "controller_state_exact": _state_dict_equal(snapshot["controller_state"], agent.controller.state_dict()),
        "policy_optimizer_restored": bool(agent.base.model.policy.optimizer.state_dict()),
        "controller_optimizer_restored": bool(agent.controller.optimizer.state_dict()),
    }


def _t90(checkpoints: list[dict]) -> dict[str, float | None]:
    if not checkpoints:
        return {"threshold": None, "timestep": None}
    start = float(checkpoints[0]["mean_return"])
    finish = float(checkpoints[-1]["mean_return"])
    threshold = start + 0.9 * (finish - start)
    rising = finish >= start
    for item in checkpoints:
        value = float(item["mean_return"])
        if (rising and value >= threshold) or (not rising and value <= threshold):
            return {"threshold": threshold, "timestep": float(item["step"])}
    return {"threshold": threshold, "timestep": None}


def _calibrate_meta(backend: RebuiltLSTMBackend, device: torch.device, hidden_size: int) -> tuple[float, dict]:
    before = _capture_rng()
    started = time.perf_counter()
    try:
        report = backend.calibrate(device, hidden_size=hidden_size)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        _restore_rng(before)
    elapsed = time.perf_counter() - started
    if not report.admitted:
        raise RuntimeError(f"Meta backend admission failed: {report.reason}")
    return elapsed, backend.to_dict()


def _run_training(agent: AMRLAgent, config: ExperimentConfig, device: torch.device) -> tuple[TrainingHistory, float]:
    def checkpoint_callback(step: int):
        if step % config.checkpoint_interval != 0:
            return None
        with synchronized_time(device) as timer:
            evaluation = agent.evaluate(config.evaluation_episodes, return_episode_returns=True)
        agent.telemetry.timing.evaluation_time += timer.seconds
        return evaluation

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    history = agent.train(config.total_timesteps, checkpoint_callback=checkpoint_callback)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return history, time.perf_counter() - started


def _stability(agent: AMRLAgent) -> dict[str, Any]:
    meta = agent.telemetry.stability.to_dict()
    post = agent.stability_audit()
    rollout = agent.base.rollout_backend
    rollout_counts = {
        "eligible_dispatch_count": 0,
        "successful_graph_dispatches": 0,
        "fallback_count": 0,
        "nan_inf_count": 0,
        "fallback_reasons": {},
    }
    if rollout is not None:
        rollout_counts = {
            "eligible_dispatch_count": int(rollout.dispatch_count + rollout.fallback_count),
            "successful_graph_dispatches": int(rollout.dispatch_count),
            "fallback_count": int(rollout.fallback_count),
            "nan_inf_count": int(rollout.nan_inf_count),
            "fallback_reasons": dict(rollout.fallback_reasons),
        }
    fallback_count = int(meta["fallback_count"] + rollout_counts["fallback_count"])
    eligible = int(meta["eligible_dispatch_count"] + rollout_counts["eligible_dispatch_count"])
    return {
        "meta_backend": meta,
        "rollout_backend": rollout_counts,
        "eligible_dispatch_count": eligible,
        "successful_dispatches": int(meta["eligible_dispatch_count"] - meta["fallback_count"] + rollout_counts["successful_graph_dispatches"]),
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / eligible if eligible else 0.0,
        "post_transition_fallback_rate": post["post_transition_fallback_rate"],
        "post_transition_fallback_count": post["post_transition_fallback_count"],
        "post_transition_eligible_dispatch_count": post["post_transition_eligible_dispatch_count"],
        "nan_inf_count": int(meta["nan_inf_count"] + rollout_counts["nan_inf_count"]),
        "nan_inf_share": (
            int(meta["nan_inf_count"] + rollout_counts["nan_inf_count"]) / eligible if eligible else 0.0
        ),
    }


def _method_record(
    *, method: str, agent: AMRLAgent, history: TrainingHistory, execution_seconds: float,
    calibration_seconds: float, meta_backend: dict, rollout_backend: dict | None,
) -> dict[str, Any]:
    checkpoints = history.checkpoint_evaluations
    final = checkpoints[-1] if checkpoints else {"mean_return": float("nan"), "std_return": float("nan")}
    info = information_diagnostics(
        history.controller_outputs, history.reward_deltas, wall_clock_seconds=execution_seconds, k=5
    )
    timing = agent.telemetry.timing.to_dict()
    base_execution = agent.base.execution_audit()
    record = {
        "method": method,
        "performance": {
            "checkpoints": checkpoints,
            "final_return": float(final["mean_return"]),
            "final_return_std": float(final["std_return"]),
            "T90": _t90(checkpoints),
        },
        "timing_seconds": {
            "training_evaluation_wall_clock": execution_seconds,
            "offline_calibration": calibration_seconds,
            "lifecycle_wall_clock": execution_seconds + calibration_seconds,
            "inner_base_runtime": timing["base_agent_training_time"],
            "main_rollout_collection": base_execution["rollout_collection_time"],
            "main_ppo_update": base_execution["ppo_update_time"],
            "meta_controller_runtime": timing["meta_learner_forward_time"] + timing["meta_learner_backward_time"],
            "meta_controller_forward": timing["meta_learner_forward_time"],
            "meta_controller_backward": timing["meta_learner_backward_time"],
            "hyperparameter_application": timing["hyperparameter_update_time"],
            "evaluation": timing["evaluation_time"],
            "logging": timing["logging_time"],
        },
        "information_utility": {
            "U_definition": "bounded 3D controller Gaussian mean tanh(head) * [0.30, 0.02, 0.05]",
            "delta_R_definition": "next adaptation-window mean training reward minus the preceding window mean reward",
            "pairing": "one-cycle delayed: U_t is paired only with DeltaR_(t+1); final unpaired controller output is excluded",
            "raw_U": history.controller_outputs,
            "raw_delta_R": history.reward_deltas,
            "metrics": info,
        },
        "stability": _stability(agent),
        "fidelity": {
            "meta_backend": meta_backend,
            "rollout_backend": rollout_backend,
        },
        "training_audit": base_execution,
        "hyperparameters_after_each_adaptation": history.hyperparameters,
        "learning_curve_training_window_mean_reward": history.mean_reward_curve,
    }
    return record


def run_halfcheetah_seed0_audit(config: ExperimentConfig, output_dir: str | Path) -> dict[str, Any]:
    """Run one strictly paired AMRL/AMRL-Fast HalfCheetah audit.

    ``config.seeds`` must contain exactly seed 0.  The caller is responsible
    for a prior environment smoke test; this function refuses aliases so the
    JSON environment ID is the ID actually executed.
    """
    if config.environment != "HalfCheetah-v4" or config.allow_version_fallback:
        raise ValueError("Formal audit requires exact HalfCheetah-v4 with version fallback disabled.")
    if config.seeds != [0]:
        raise ValueError("Seed-0 preflight requires config.seeds == [0].")
    if config.total_timesteps != 100_000 or config.adaptation_interval != 10_000:
        raise ValueError("Formal audit requires 100000 main steps and adaptation_interval=10000.")
    if config.evaluation_episodes != 20 or config.checkpoint_interval != 10_000:
        raise ValueError("Formal audit requires 20 episodes at each 10000-step checkpoint.")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    device = _device(config.device)
    seed = 0

    base_env, resolved = make_env(config.environment, seed=seed, allow_version_fallback=False)
    base_eval, resolved_eval = make_env(config.environment, seed=seed + 1_000_000, allow_version_fallback=False)
    if resolved != "HalfCheetah-v4" or resolved_eval != resolved:
        raise RuntimeError(f"Resolved environment mismatch: train={resolved}, evaluation={resolved_eval}")
    baseline_backend = RebuiltLSTMBackend()
    baseline = AMRLAgent(
        base_env, evaluation_env=base_eval, device=device, seed=seed, use_fast=False,
        backend=baseline_backend, adaptation_interval=config.adaptation_interval,
        warmup_steps=config.warmup_steps, transition_steps=config.transition_steps,
        controller_start_after_steps=config.controller_start_after_steps,
        controller_hidden_size=config.controller_hidden_size,
        controller_batched_replay=config.controller_batched_replay,
    )
    snapshot = _initial_snapshot(baseline, seed=seed, environment_id=resolved)
    snapshot_path = root / "paired_initialization_seed0.pt"
    torch.save(snapshot, snapshot_path)
    snapshot_digest = _state_digest(snapshot)
    try:
        baseline_history, baseline_execution = _run_training(baseline, config, device)
        baseline_record = _method_record(
            method="AMRL", agent=baseline, history=baseline_history,
            execution_seconds=baseline_execution, calibration_seconds=0.0,
            meta_backend=baseline_backend.to_dict(), rollout_backend=None,
        )
    finally:
        base_env.close()
        base_eval.close()

    fast_env, fast_resolved = make_env(config.environment, seed=seed, allow_version_fallback=False)
    fast_eval, fast_eval_resolved = make_env(config.environment, seed=seed + 1_000_000, allow_version_fallback=False)
    if fast_resolved != resolved or fast_eval_resolved != resolved:
        fast_env.close(); fast_eval.close()
        raise RuntimeError("Fast environment did not resolve to the paired AMRL ID.")
    fast_backend = RebuiltLSTMBackend()
    calibration_seconds, meta_report = _calibrate_meta(fast_backend, device, config.controller_hidden_size)
    fast = AMRLAgent(
        fast_env, evaluation_env=fast_eval, device=device, seed=seed, use_fast=True,
        backend=fast_backend, adaptation_interval=config.adaptation_interval,
        warmup_steps=config.warmup_steps, transition_steps=config.transition_steps,
        controller_start_after_steps=config.controller_start_after_steps,
        controller_hidden_size=config.controller_hidden_size,
        controller_batched_replay=config.controller_batched_replay,
        enable_rollout_backend=True,
    )
    try:
        restoration = _restore_initial_snapshot(fast, snapshot)
        rollout_started = time.perf_counter()
        rollout_report = fast.base.admit_rollout_backend()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        calibration_seconds += time.perf_counter() - rollout_started
        if rollout_report is None or not rollout_report.get("admitted", False):
            raise RuntimeError(f"Continuous rollout backend admission failed: {rollout_report}")
        # Admission must not alter the common initial weights, optimisers,
        # environment state or RNG stream.  Reload to make this invariant
        # explicit even if a future admission implementation changes.
        post_admission_restoration = _restore_initial_snapshot(fast, snapshot)
        fast_history, fast_execution = _run_training(fast, config, device)
        fast_record = _method_record(
            method="AMRL-Fast", agent=fast, history=fast_history,
            execution_seconds=fast_execution, calibration_seconds=calibration_seconds,
            meta_backend=meta_report, rollout_backend=rollout_report,
        )
    finally:
        fast_env.close()
        fast_eval.close()

    base_time = baseline_record["timing_seconds"]["training_evaluation_wall_clock"]
    fast_time = fast_record["timing_seconds"]["training_evaluation_wall_clock"]
    base_lifecycle = baseline_record["timing_seconds"]["lifecycle_wall_clock"]
    fast_lifecycle = fast_record["timing_seconds"]["lifecycle_wall_clock"]
    fidelity = fast_record["fidelity"]
    meta_admitted = bool(fidelity["meta_backend"].get("admitted", False))
    rollout_admitted = bool((fidelity["rollout_backend"] or {}).get("admitted", False))
    nonfinite = fast_record["stability"]["nan_inf_count"] + baseline_record["stability"]["nan_inf_count"]
    acceptance = {
        "main_timesteps_complete": baseline_record["training_audit"]["actual_environment_steps"] == 100_000
        and fast_record["training_audit"]["actual_environment_steps"] == 100_000,
        "no_nan_inf": nonfinite == 0,
        "fast_fidelity_passed": meta_admitted and rollout_admitted,
        "information_metrics_computable": all(
            baseline_record["information_utility"]["metrics"][key] is not None
            and fast_record["information_utility"]["metrics"][key] is not None
            for key in ("discrete_mi_bits", "knn_mi_bits", "spearman_rho")
        ),
        "paired_initialization_restored": all(restoration.values()) and all(post_admission_restoration.values()),
        "positive_training_evaluation_speedup": fast_time < base_time,
        "returns_same_order_of_magnitude": abs(fast_record["performance"]["final_return"] - baseline_record["performance"]["final_return"])
        <= max(100.0, 0.5 * abs(baseline_record["performance"]["final_return"])),
    }
    output = {
        "schema": "halfcheetah-amrl-amrl-fast-seed0-audit-r1",
        "status": "seed0_preflight_not_five_seed_result",
        "paper_result_eligible": False,
        "environment": {"requested": config.environment, "resolved": resolved},
        "protocol": {
            "main_timesteps": config.total_timesteps,
            "adaptation_interval": config.adaptation_interval,
            "adaptation_events": config.total_timesteps // config.adaptation_interval,
            "checkpoint_interval": config.checkpoint_interval,
            "evaluation_episodes_per_checkpoint": config.evaluation_episodes,
            "controller_input_size": 5,
            "controller_hidden_size": config.controller_hidden_size,
            "controller_output_size": 3,
            "controller_action_std": 0.10,
            "same_algorithm_invariant": "AMRL and Fast share PPO/controller/loss/optimizer/action mapping/budget/evaluation; Fast differs only by admitted native-LSTM and continuous rollout CUDA-graph execution backends.",
        },
        "timing_boundary": AUDIT_TIMING_BOUNDARY,
        "paired_initialization": {
            "snapshot_file": snapshot_path.name,
            "sha256": snapshot_digest,
            "saved_fields": ["policy weights", "policy optimizer", "controller weights", "controller optimizer", "Python/NumPy/Torch CPU/CUDA RNG", "fresh seeded training/evaluation MuJoCo state"],
            "fast_restoration_before_admission": restoration,
            "fast_restoration_after_admission": post_admission_restoration,
        },
        "AMRL": baseline_record,
        "AMRL_Fast": fast_record,
        "paired_efficiency": {
            "training_evaluation_speedup": (base_time - fast_time) / base_time,
            "lifecycle_speedup": (base_lifecycle - fast_lifecycle) / base_lifecycle,
            "training_evaluation_seconds_saved": base_time - fast_time,
            "lifecycle_seconds_saved": base_lifecycle - fast_lifecycle,
        },
        "acceptance": {**acceptance, "all_pass": all(acceptance.values())},
    }
    _write_json(root / "halfcheetah_v4_seed0_paired_audit.json", output)
    return output

