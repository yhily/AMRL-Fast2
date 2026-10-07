#!/usr/bin/env python3
"""Formal-experiment preflight for Hopper and Walker2d.

This runner deliberately reuses ``run_seed_with_history`` from the formal
experiment module.  It shortens only the development horizon and adaptation
window so that one paired seed yields multiple delayed ``U_t, DeltaR_t``
pairs.  It must never be used as a paper-result generator.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrl_fast.experiment import (
    TIMING_BOUNDARY,
    ExperimentConfig,
    _device,
    run_seed_with_history,
)
from amrl_fast.rebuilt_lstm import RebuiltLSTMBackend, public_manual_lstm_cell
from amrl_fast.statistics import discrete_mutual_information


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _paired_records(history) -> list[dict]:
    if len(history.controller_outputs) != len(history.reward_deltas):
        raise RuntimeError("MI pairing mismatch: controller outputs and reward deltas differ in length.")
    return [
        {"U_t": output, "DeltaR_t": delta}
        for output, delta in zip(history.controller_outputs, history.reward_deltas)
    ]


def _fallback_probe(device: torch.device) -> dict:
    """Exercise normal dispatch plus exception and NaN/Inf fallback branches.

    This probe is isolated from the AMRL trials and is labeled as forced-path
    validation.  It neither changes their parameters nor contributes to any
    reported fallback statistic.
    """
    torch.manual_seed(911)
    backend = RebuiltLSTMBackend()
    admission = backend.calibrate(device, warmup=10, repeats=60)
    if not admission.admitted:
        return {"backend_admitted": False, "reason": admission.reason}
    inputs = torch.randn(1, 5, device=device)
    hidden = torch.randn(1, 256, device=device)
    cell = torch.randn(1, 256, device=device)
    weight_ih = torch.randn(1024, 5, device=device)
    weight_hh = torch.randn(1024, 256, device=device)
    bias = torch.randn(1024, device=device)
    bias_hh = torch.zeros(1024, device=device)
    standard = public_manual_lstm_cell(inputs, hidden, cell, weight_ih, weight_hh, bias)
    optimized = backend.run(inputs, hidden, cell, weight_ih, weight_hh, bias, bias_hh, activation_weight=1.0)
    normal_dispatch = backend.dispatch_count == 1 and backend.fallback_count == 0
    normal_agreement = bool(
        torch.allclose(optimized[0], standard[0], rtol=5e-5, atol=5e-6)
        and torch.allclose(optimized[1], standard[1], rtol=5e-5, atol=5e-6)
    )
    selected = backend._implementation
    backend._implementation = "forced-candidate-exception"
    before_exception = backend.fallback_count
    exception_output = backend.run(inputs, hidden, cell, weight_ih, weight_hh, bias, bias_hh, activation_weight=1.0)
    candidate_exception_fallback = bool(
        backend.fallback_count == before_exception + 1
        and torch.equal(exception_output[0], standard[0])
        and torch.equal(exception_output[1], standard[1])
    )
    backend._implementation = selected
    nan_inputs = torch.full_like(inputs, float("nan"))
    before_nan_fallback = backend.fallback_count
    before_nan_count = backend.nan_inf_count
    backend.run(
        nan_inputs, hidden, cell, weight_ih, weight_hh, bias, bias_hh,
        activation_weight=1.0, validate_outputs=True,
    )
    nan_inf_fallback = bool(
        backend.fallback_count == before_nan_fallback + 1
        and backend.nan_inf_count == before_nan_count + 1
    )
    rejected = RebuiltLSTMBackend(minimum_speedup=float("inf"))
    rejected_report = rejected.calibrate(device, warmup=3, repeats=10)
    rejected_output = rejected.run(inputs, hidden, cell, weight_ih, weight_hh, bias, bias_hh, activation_weight=1.0)
    rejected_candidate_standard = bool(
        not rejected_report.admitted
        and rejected.dispatch_count == 0
        and torch.equal(rejected_output[0], standard[0])
        and torch.equal(rejected_output[1], standard[1])
    )
    return {
        "backend_admitted": True,
        "normal_dispatch_uses_optimized_backend": normal_dispatch,
        "normal_output_matches_standard": normal_agreement,
        "forced_candidate_exception_falls_back_to_standard": candidate_exception_fallback,
        "forced_nan_inf_triggers_fallback": nan_inf_fallback,
        "rejected_candidate_uses_standard_backend": rejected_candidate_standard,
        "forced_path_final_counters": {
            "dispatch_count": backend.dispatch_count,
            "fallback_count": backend.fallback_count,
            "nan_inf_count": backend.nan_inf_count,
            "fallback_ratio": backend.fallback_count / backend.dispatch_count if backend.dispatch_count else 0.0,
        },
    }


def _admission_pass(report: dict) -> bool:
    return bool(
        report.get("admitted")
        and report.get("output_correlation", -float("inf")) >= 0.999
        and report.get("gradient_correlation", -float("inf")) >= 0.99
        and report.get("relative_l2_error", float("inf")) <= 5e-3
        and report.get("max_absolute_error", float("inf")) <= 1e-1
        and report.get("gradient_relative_l2_error", float("inf")) <= 5e-3
        and report.get("output_finite")
        and report.get("gradient_finite")
        and report.get("measured_speedup", 0.0) >= 1.02
    )


def _run_environment(args, environment: str) -> dict:
    config = ExperimentConfig(
        environment=environment,
        seeds=[args.seed],
        total_timesteps=args.timesteps,
        adaptation_interval=args.adaptation_interval,
        warmup_steps=0,
        transition_steps=0,
        evaluation_episodes=args.evaluation_episodes,
        device=args.device,
        output_dir=str(args.output_dir),
        allow_version_fallback=args.allow_version_fallback,
        development_only=True,
    )
    baseline, baseline_id, baseline_history = run_seed_with_history(config, "amrl", args.seed)
    fast, fast_id, fast_history = run_seed_with_history(config, "amrl_fast", args.seed)
    if baseline_id != fast_id:
        raise RuntimeError("Paired preflight resolved different environment IDs.")
    base_pairs = _paired_records(baseline_history)
    fast_pairs = _paired_records(fast_history)
    base_mi = discrete_mutual_information(baseline_history.controller_outputs, baseline_history.reward_deltas)
    fast_mi = discrete_mutual_information(fast_history.controller_outputs, fast_history.reward_deltas)
    if not math.isclose(base_mi, baseline.discrete_mi_bits, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("AMRL MI recomputation differs from the formal seed result.")
    if not math.isclose(fast_mi, fast.discrete_mi_bits, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("AMRL-Fast MI recomputation differs from the formal seed result.")
    directory = args.output_dir / environment
    common = {
        "development_only": True,
        "paper_result_eligible": False,
        "environment": environment,
        "resolved_environment_id": baseline_id,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "adaptation_interval": args.adaptation_interval,
        "evaluation_episodes": args.evaluation_episodes,
        "statistical_unit": "random seed",
        "end_to_end_timing_boundary": TIMING_BOUNDARY,
    }
    runtime = {
        **common,
        "AMRL_wall_clock": baseline.wall_clock_time,
        "AMRL_Fast_wall_clock": fast.wall_clock_time,
        "end_to_end_speedup": (baseline.wall_clock_time - fast.wall_clock_time) / baseline.wall_clock_time,
        "fast_calibration_included": True,
        "AMRL_calibration_time": baseline.calibration_time,
        "AMRL_Fast_calibration_time": fast.calibration_time,
        "AMRL_training_evaluation_wall_clock": baseline.training_evaluation_wall_clock_time,
        "AMRL_Fast_training_evaluation_wall_clock": fast.training_evaluation_wall_clock_time,
        "training_evaluation_speedup": (
            (baseline.training_evaluation_wall_clock_time - fast.training_evaluation_wall_clock_time)
            / baseline.training_evaluation_wall_clock_time
        ),
    }
    performance = {
        **common,
        "AMRL_final_return": baseline.final_return,
        "AMRL_final_return_std": baseline.final_return_std,
        "AMRL_Fast_final_return": fast.final_return,
        "AMRL_Fast_final_return_std": fast.final_return_std,
        "AMRL_reward_curve": baseline_history.mean_reward_curve,
        "AMRL_Fast_reward_curve": fast_history.mean_reward_curve,
    }
    adaptation = {
        **common,
        "pairing_rule": "U_t is paired with DeltaR in the next adaptation window.",
        "MI_estimator": "equal-frequency joint plug-in mutual information (bits)",
        "AMRL_pairs": base_pairs,
        "AMRL_Fast_pairs": fast_pairs,
        "MI_AMRL": base_mi,
        "MI_AMRL_Fast": fast_mi,
        "same_estimator_for_both_methods": True,
    }
    profiling = {
        **common,
        "AMRL": baseline.timing,
        "AMRL_Fast": fast.timing,
        "AMRL_meta_execution_time": baseline.timing["meta_learner_forward_time"] + baseline.timing["meta_learner_backward_time"],
        "AMRL_Fast_meta_execution_time": fast.timing["meta_learner_forward_time"] + fast.timing["meta_learner_backward_time"],
        "kernel_forward_backward_speedup": fast.backend.get("measured_speedup"),
    }
    profiling["meta_execution_reduction_fraction"] = (
        (profiling["AMRL_meta_execution_time"] - profiling["AMRL_Fast_meta_execution_time"])
        / profiling["AMRL_meta_execution_time"]
        if profiling["AMRL_meta_execution_time"] > 0
        else 0.0
    )
    profiling["short_run_meta_execution_reduced"] = bool(
        profiling["AMRL_Fast_meta_execution_time"] <= profiling["AMRL_meta_execution_time"]
    )
    profiling["kernel_meta_execution_reduced"] = bool(
        fast.backend.get("measured_speedup", 0.0) >= 1.02
    )
    fallback = _fallback_probe(_device(args.device))
    stability = {
        **common,
        "AMRL": baseline.stability,
        "AMRL_Fast": fast.stability,
        "fallback_ratio_matches_counter": math.isclose(
            fast.stability["fallback_ratio"],
            fast.stability["fallback_count"] / fast.stability["eligible_dispatch_count"] if fast.stability["eligible_dispatch_count"] else 0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "forced_fallback_validation": fallback,
    }
    fidelity = {**common, "AMRL_Fast_backend": fast.backend}
    payloads = {
        "runtime.json": runtime,
        "performance.json": performance,
        "adaptation.json": adaptation,
        "profiling.json": profiling,
        "stability.json": stability,
        "fidelity.json": fidelity,
    }
    for filename, payload in payloads.items():
        _write_json(directory / filename, payload)
    checks = {
        "kernel_admission": _admission_pass(fast.backend),
        "end_to_end_timing": all(_finite(value) and value > 0 for value in (baseline.wall_clock_time, fast.wall_clock_time)),
        "learning_performance": all(_finite(value) for value in (baseline.final_return, fast.final_return)) and bool(baseline_history.mean_reward_curve) and bool(fast_history.mean_reward_curve),
        "mi_pipeline": len(base_pairs) >= 2 and len(fast_pairs) >= 2 and len(base_pairs) == len(baseline_history.reward_deltas) and len(fast_pairs) == len(fast_history.reward_deltas),
        # The rebuilt backend is eligible only if it reduces both the exact
        # synchronized F+B calibration path and the integrated meta path
        # measured by this paired short run.
        "profiling": all(_finite(value) for trial in (baseline, fast) for value in trial.timing.values()) and profiling["kernel_meta_execution_reduced"] and profiling["short_run_meta_execution_reduced"],
        "fallback": all(bool(fallback.get(key)) for key in (
            "normal_dispatch_uses_optimized_backend",
            "normal_output_matches_standard",
            "forced_candidate_exception_falls_back_to_standard",
            "forced_nan_inf_triggers_fallback",
            "rejected_candidate_uses_standard_backend",
        )) and stability["fallback_ratio_matches_counter"],
        "result_files": all((directory / filename).is_file() for filename in payloads),
    }
    return {
        "environment": environment,
        "resolved_environment_id": baseline_id,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "kernel": {
            "backend": fast.backend.get("backend_name"),
            "implementation": fast.backend.get("implementation"),
            "speedup": fast.backend.get("measured_speedup"),
            "output_correlation": fast.backend.get("output_correlation"),
            "gradient_correlation": fast.backend.get("gradient_correlation"),
        },
        "runtime": runtime,
        "performance": {
            "AMRL_final_return": baseline.final_return,
            "AMRL_Fast_final_return": fast.final_return,
        },
        "mi": {"AMRL": base_mi, "AMRL_Fast": fast_mi, "pair_count": {"AMRL": len(base_pairs), "AMRL_Fast": len(fast_pairs)}},
        "profiling": profiling,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environments", nargs="+", default=["Hopper-v2", "Walker2d-v2"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=1024)
    parser.add_argument("--adaptation-interval", type=int, default=128)
    parser.add_argument("--evaluation-episodes", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-version-fallback", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/preflight"))
    args = parser.parse_args()
    if args.timesteps < args.adaptation_interval * 6:
        raise SystemExit("Use at least six adaptation windows to validate delayed MI pairing and an outer PPO update.")
    summary = {
        "development_only": True,
        "paper_result_eligible": False,
        "purpose": "Formal experiment preflight; not a five-seed manuscript result.",
        "results": [_run_environment(args, environment) for environment in args.environments],
    }
    summary["all_checks_pass"] = all(result["all_checks_pass"] for result in summary["results"])
    _write_json(args.output_dir / "preflight_acceptance.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["all_checks_pass"]:
        raise SystemExit("Preflight validation failed; do not start formal experiments.")


if __name__ == "__main__":
    main()
