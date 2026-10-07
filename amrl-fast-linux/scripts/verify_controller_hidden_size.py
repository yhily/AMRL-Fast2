#!/usr/bin/env python3
"""Verify that controller hidden size reaches the production LSTM hot path.

This is a development-only structural diagnostic.  It invokes the exact
``MetaController.act`` and ``MetaController.update`` production methods with
the real five-feature, batch-one controller interface and a four-transition,
four-epoch outer PPO update.  It does not execute a benchmark training loop,
change the controller objective, or change any kernel implementation.
"""
from __future__ import annotations

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


HIDDEN_SIZES = (64, 128, 256)
FEATURE_SIZE = 5
TRUNK_SIZE = 128
ACTION_SIZE = 3


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parameter_count(controller: MetaController) -> int:
    return sum(parameter.numel() for parameter in controller.parameters())


def _gate_flops(hidden_size: int) -> dict[str, int]:
    """Count dominant dense arithmetic as two FLOPs per multiply-add.

    The backward count includes dX and dW GEMMs for each forward GEMM.  Gate
    nonlinearities and optimizer arithmetic are intentionally excluded and
    reported as non-GEMM work.
    """
    lstm_forward = 2 * (FEATURE_SIZE * (4 * hidden_size) + hidden_size * (4 * hidden_size))
    heads_forward = 2 * (
        hidden_size * TRUNK_SIZE + TRUNK_SIZE * ACTION_SIZE + TRUNK_SIZE
    )
    return {
        "lstm_gate_forward": lstm_forward,
        "lstm_gate_backward": 2 * lstm_forward,
        "controller_dense_forward": lstm_forward + heads_forward,
        "controller_dense_backward": 2 * (lstm_forward + heads_forward),
    }


def _make_transitions(controller: MetaController) -> list:
    transitions = []
    diagnostics = (
        (10.0, 1.0, 0.5, 0.1, 0.2),
        (11.0, 1.1, 0.6, 0.11, 0.19),
        (12.0, 0.9, 0.4, 0.12, 0.21),
        (13.0, 1.2, 0.7, 0.09, 0.18),
    )
    for index, values in enumerate(diagnostics):
        feature = torch.tensor([values], dtype=torch.float32, device=controller.device)
        _, transition = controller.act(feature, activation_weight=1.0)
        transition.reward = (index + 1) / 10.0
        transitions.append(transition)
    return transitions


def _structure(controller: MetaController) -> dict:
    cell = controller.cell
    return {
        "lstm_input_size": FEATURE_SIZE,
        "lstm_hidden_size": controller.hidden_size,
        "lstm_num_layers": 1,
        "policy_head": {
            "trunk_in_features": controller.trunk[0].in_features,
            "trunk_out_features": controller.trunk[0].out_features,
            "action_mean_in_features": controller.action_mean.in_features,
            "action_mean_out_features": controller.action_mean.out_features,
            "value_head_in_features": controller.value_head.in_features,
            "value_head_out_features": controller.value_head.out_features,
        },
        "parameter_count": _parameter_count(controller),
        "lstm_weight_ih_shape": list(cell.weight_ih.shape),
        "lstm_weight_hh_shape": list(cell.weight_hh.shape),
        "lstm_bias_shape": list(cell.bias.shape),
    }


def _diagnose(hidden_size: int, device: torch.device, use_fast: bool) -> dict:
    torch.manual_seed(9100 + hidden_size)
    backend = RebuiltLSTMBackend()
    calibration = None
    if use_fast:
        calibration = backend.calibrate(device, hidden_size=hidden_size)
        if not calibration.admitted:
            raise RuntimeError(f"Kernel not admitted for hidden={hidden_size}: {calibration.reason}")

    # Pair baseline/Fast controller initialization independently of calibration.
    torch.manual_seed(9100 + hidden_size)
    controller = MetaController(
        device=device,
        backend=backend,
        use_fast=use_fast,
        hidden_size=hidden_size,
    )
    structure = _structure(controller)

    # Warm Adam state allocation with the production update before measurement.
    controller.update(_make_transitions(controller))
    _sync(device)

    transitions = _make_transitions(controller)
    first = transitions[0]
    dispatch_before = backend.dispatch_count
    timer = SynchronizedSegmentTimer(device)
    _sync(device)
    started = time.perf_counter()
    gradient_norm = controller.update(transitions, segments=timer)
    _sync(device)
    total_us = (time.perf_counter() - started) * 1e6
    dispatch_after = backend.dispatch_count

    segments = timer.microseconds()
    lstm_segment = "native_lstm_cell_forward" if use_fast else "standard_lstm_forward"
    forward_us = sum(
        segments.get(name, {}).get("total_us", 0.0)
        for name in ("meta_update_input_preparation", "input_preparation", lstm_segment, "policy_value_head", "loss")
    )
    backward_us = segments.get("backward", {}).get("total_us", 0.0)
    optimizer_us = segments.get("optimizer_step", {}).get("total_us", 0.0)
    cell = controller.cell
    observed = backend.shape_trace.to_dict() if backend.shape_trace else None
    gate_shapes = {
        "input_gemm": [[1, FEATURE_SIZE], [FEATURE_SIZE, 4 * hidden_size]],
        "recurrent_gemm": [[1, hidden_size], [hidden_size, 4 * hidden_size]],
        "four_gate_output": [1, 4 * hidden_size],
    }
    gradient_shapes = {
        "weight_ih_grad": list(cell.weight_ih.grad.shape) if cell.weight_ih.grad is not None else None,
        "weight_hh_grad": list(cell.weight_hh.grad.shape) if cell.weight_hh.grad is not None else None,
        "bias_grad": list(cell.bias.grad.shape) if cell.bias.grad is not None else None,
        "trunk_weight_grad": list(controller.trunk[0].weight.grad.shape) if controller.trunk[0].weight.grad is not None else None,
        "action_mean_weight_grad": list(controller.action_mean.weight.grad.shape) if controller.action_mean.weight.grad is not None else None,
        "value_head_weight_grad": list(controller.value_head.weight.grad.shape) if controller.value_head.weight.grad is not None else None,
    }
    return {
        "path": "AMRL-Fast" if use_fast else "AMRL",
        "controller_structure": structure,
        "real_meta_forward_tensors": {
            "input_shape": list(first.features.shape),
            "hidden_shape": list(first.h_before.shape),
            "cell_shape": list(first.c_before.shape),
            "lstm_weights": {
                "weight_ih": list(cell.weight_ih.shape),
                "weight_hh": list(cell.weight_hh.shape),
                "bias": list(cell.bias.shape),
            },
            "gate_matmul_shapes": gate_shapes,
            "native_lstm_received_shape": observed if use_fast else None,
        },
        "real_meta_backward_gradient_shapes": gradient_shapes,
        "main_dense_flops_per_single_controller_evaluation": _gate_flops(hidden_size),
        "kernel_calls_during_one_real_meta_update": {
            "native_lstm_cell": dispatch_after - dispatch_before if use_fast else 0,
            "production_outer_ppo_epochs": 4,
            "stored_transitions": len(transitions),
            "expected_replay_evaluations": 4 * len(transitions),
        },
        "cuda_synchronized_timing_us": {
            "full_meta_update": total_us,
            "full_forward_and_loss": forward_us,
            "lstm_gate_forward": segments.get(lstm_segment, {}).get("total_us", 0.0),
            "backward": backward_us,
            "optimizer_step": optimizer_us,
            "segments": segments,
        },
        "gradient_norm": gradient_norm,
        "kernel_admission": calibration.to_dict() if calibration else None,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA is required for this timing validation.")
    results = []
    for hidden_size in HIDDEN_SIZES:
        baseline = _diagnose(hidden_size, device, use_fast=False)
        fast = _diagnose(hidden_size, device, use_fast=True)
        if baseline["controller_structure"] != fast["controller_structure"]:
            raise RuntimeError(f"Baseline/Fast controller mismatch for hidden={hidden_size}")
        results.append({"hidden_size": hidden_size, "AMRL": baseline, "AMRL_Fast": fast})
    payload = {
        "development_only": True,
        "paper_result_eligible": False,
        "purpose": "Validate that hidden size changes the real production controller and timed kernel path.",
        "timing_rule": "All reported CUDA timing segments synchronize before and after the measured operation.",
        "results": results,
    }
    output = ROOT / "artifacts" / "controller_hidden_size_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
