#!/usr/bin/env python3
"""Validate and profile sequential versus batched production outer replay.

The test uses the exact MetaController PPO update from the training path.  It
does not change the objective, reward, number of epochs, optimizer, or number
of optimizer steps; it only batches independent saved replay states within an
epoch.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrl_fast.agent import MetaController
from amrl_fast.rebuilt_lstm import RebuiltLSTMBackend
from amrl_fast.telemetry import SynchronizedSegmentTimer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 256
THRESHOLDS = {"correlation": 0.999, "relative_l2": 5e-3, "max_absolute": 1e-1}


def _sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def _compare(first: torch.Tensor, second: torch.Tensor) -> dict:
    first = first.detach().float().reshape(-1)
    second = second.detach().float().reshape(-1)
    diff = first - second
    centred_first, centred_second = first - first.mean(), second - second.mean()
    denominator = torch.linalg.vector_norm(centred_first) * torch.linalg.vector_norm(centred_second)
    if denominator <= 1e-12:
        correlation = 1.0 if torch.allclose(first, second) else 0.0
    else:
        correlation = float((torch.dot(centred_first, centred_second) / denominator).clamp(-1, 1).item())
    relative_l2 = float((torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(first).clamp_min(1e-12)).item())
    max_absolute = float(diff.abs().max().item()) if diff.numel() else 0.0
    return {
        "correlation": correlation,
        "relative_l2": relative_l2,
        "max_absolute": max_absolute,
        "passes_existing_fidelity_thresholds": bool(
            correlation >= THRESHOLDS["correlation"]
            and relative_l2 <= THRESHOLDS["relative_l2"]
            and max_absolute <= THRESHOLDS["max_absolute"]
        ),
    }


def _make_controller(use_fast: bool) -> tuple[MetaController, RebuiltLSTMBackend]:
    backend = RebuiltLSTMBackend()
    if use_fast:
        report = backend.calibrate(DEVICE, hidden_size=HIDDEN_SIZE)
        if not report.admitted:
            raise RuntimeError(f"Rebuilt backend not admitted: {report.reason}")
    torch.manual_seed(411)
    return MetaController(
        device=DEVICE,
        backend=backend,
        use_fast=use_fast,
        hidden_size=HIDDEN_SIZE,
    ), backend


def _make_transitions(controller: MetaController) -> list:
    transitions = []
    diagnostics = (
        (10.0, 1.0, 0.5, 0.1, 0.2),
        (11.0, 1.1, 0.6, 0.11, 0.19),
        (12.0, 0.9, 0.4, 0.12, 0.21),
        (13.0, 1.2, 0.7, 0.09, 0.18),
    )
    for index, values in enumerate(diagnostics):
        feature = torch.tensor([values], dtype=torch.float32, device=DEVICE)
        _, transition = controller.act(feature, activation_weight=1.0)
        transition.reward = (index + 1) / 10.0
        transitions.append(transition)
    return transitions


def _replay_policy_value(controller: MetaController, transitions: list, *, batched: bool) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        if not batched:
            means, values = [], []
            for item in transitions:
                distribution, value, _ = controller._evaluate(
                    item.features, (item.h_before, item.c_before), item.activation_weight
                )
                means.append(distribution.mean)
                values.append(value)
            return torch.cat(means, dim=0), torch.cat(values, dim=0)
        features = torch.cat([item.features for item in transitions], dim=0)
        hidden = torch.cat([item.h_before for item in transitions], dim=0)
        cell = torch.cat([item.c_before for item in transitions], dim=0)
        distribution, values, _ = controller._evaluate(features, (hidden, cell), 1.0)
        return distribution.mean, values


def _clone_controller(state: dict, *, use_fast: bool) -> tuple[MetaController, RebuiltLSTMBackend]:
    controller, backend = _make_controller(use_fast)
    controller.load_state_dict(state)
    return controller, backend


def _parameters(controller: MetaController) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in controller.parameters()])


def _gradients(controller: MetaController) -> torch.Tensor:
    return torch.cat([
        (parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).detach().reshape(-1)
        for parameter in controller.parameters()
    ])


def _equivalence(use_fast: bool) -> dict:
    source, _ = _make_controller(use_fast)
    transitions = _make_transitions(source)
    state = copy.deepcopy(source.state_dict())
    sequential, _ = _clone_controller(state, use_fast=use_fast)
    batched, _ = _clone_controller(state, use_fast=use_fast)
    sequential_mean, sequential_value = _replay_policy_value(sequential, transitions, batched=False)
    batched_mean, batched_value = _replay_policy_value(batched, transitions, batched=True)
    sequential_debug: dict = {}
    batched_debug: dict = {}
    sequential.update(transitions, batched_replay=False, debug=sequential_debug)
    batched.update(transitions, batched_replay=True, debug=batched_debug)
    losses = _compare(
        torch.tensor(sequential_debug["epoch_losses"], device=DEVICE),
        torch.tensor(batched_debug["epoch_losses"], device=DEVICE),
    )
    return {
        "path": "AMRL-Fast" if use_fast else "AMRL",
        "policy_mean": _compare(sequential_mean, batched_mean),
        "value": _compare(sequential_value, batched_value),
        "total_loss_per_epoch": losses,
        "gradients_after_final_epoch": _compare(_gradients(sequential), _gradients(batched)),
        "parameters_after_optimizer_updates": _compare(_parameters(sequential), _parameters(batched)),
        "sequential_debug": sequential_debug,
        "batched_debug": batched_debug,
        "all_checks_pass": all(
            item["passes_existing_fidelity_thresholds"]
            for item in (
                _compare(sequential_mean, batched_mean),
                _compare(sequential_value, batched_value),
                losses,
                _compare(_gradients(sequential), _gradients(batched)),
                _compare(_parameters(sequential), _parameters(batched)),
            )
        ),
    }


def _profile(use_fast: bool, batched_replay: bool) -> dict:
    controller, backend = _make_controller(use_fast)
    # Warm each path independently so the timed update includes production
    # work but not first-use allocator/Adam-state costs.
    controller.update(_make_transitions(controller), batched_replay=batched_replay)
    _sync()
    transitions = _make_transitions(controller)
    segments = SynchronizedSegmentTimer(DEVICE)
    _sync()
    started = time.perf_counter()
    controller.update(transitions, batched_replay=batched_replay, segments=segments)
    _sync()
    elapsed_us = (time.perf_counter() - started) * 1e6
    # Profile separately: profiler overhead is excluded from the timing above.
    kernel_event_count = None
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
            controller.update(_make_transitions(controller), batched_replay=batched_replay)
        _sync()
        kernel_event_count = sum(
            1 for event in profiler.events()
            if str(getattr(event, "device_type", "")).endswith("CUDA")
        )
    except Exception as exc:  # diagnostic data only; retain failure detail
        kernel_event_count = {"unavailable": f"{type(exc).__name__}: {exc}"}
    segment_us = segments.microseconds()
    lstm_key = "native_lstm_cell_forward" if use_fast else "standard_lstm_forward"
    return {
        "path": "AMRL-Fast" if use_fast else "AMRL",
        "replay_execution": "batched" if batched_replay else "sequential",
        "complete_outer_ppo_update_us": elapsed_us,
        "lstm_forward_us": segment_us.get(lstm_key, {}).get("total_us", 0.0),
        "policy_value_head_us": segment_us.get("policy_value_head", {}).get("total_us", 0.0),
        "loss_us": segment_us.get("loss", {}).get("total_us", 0.0),
        "backward_us": segment_us.get("backward", {}).get("total_us", 0.0),
        "optimizer_us": segment_us.get("optimizer_step", {}).get("total_us", 0.0),
        "segments": segment_us,
        "controller_replay_invocations": 4 if batched_replay else 16,
        "native_lstm_dispatches": backend.dispatch_count if use_fast else 0,
        "cuda_kernel_event_count": kernel_event_count,
        "kernel_report": backend.to_dict() if use_fast else None,
    }


def main() -> None:
    if DEVICE.type != "cuda":
        raise SystemExit("CUDA is required for the replay-throughput diagnostic.")
    equivalence = {"AMRL": _equivalence(False), "AMRL_Fast": _equivalence(True)}
    profiles = {
        "AMRL_sequential": _profile(False, False),
        "AMRL_batched": _profile(False, True),
        "AMRL_Fast_sequential": _profile(True, False),
        "AMRL_Fast_batched": _profile(True, True),
    }
    baseline = profiles["AMRL_sequential"]["complete_outer_ppo_update_us"]
    fast_batched = profiles["AMRL_Fast_batched"]["complete_outer_ppo_update_us"]
    payload = {
        "development_only": True,
        "paper_result_eligible": False,
        "purpose": "Numerical equivalence and CUDA-synchronized profile of batched outer PPO replay.",
        "semantics": {
            "saved_state_independence": "Each replay transition uses its own stored h_before/c_before; replay does not advance recurrent state across transitions.",
            "epochs": 4,
            "optimizer_steps": 4,
            "stored_transitions": 4,
            "sequential_replays": 16,
            "batched_replays": 4,
        },
        "equivalence": equivalence,
        "profiles": profiles,
        "sequential_amrl_to_batched_fast_meta_update_speedup": baseline / fast_batched,
        "all_equivalence_checks_pass": equivalence["AMRL"]["all_checks_pass"] and equivalence["AMRL_Fast"]["all_checks_pass"],
    }
    target = ROOT / "artifacts" / "batched_outer_replay_validation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_equivalence_checks_pass"]:
        raise SystemExit("Batched replay did not meet the existing fidelity thresholds.")


if __name__ == "__main__":
    main()
