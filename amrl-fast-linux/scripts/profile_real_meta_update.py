#!/usr/bin/env python3
"""CUDA-synchronized, segment-level diagnosis of one real AMRL meta update.

This executes ``MetaController.act`` and ``MetaController.update`` directly;
it does not reimplement the PPO outer loss.  The four stored transitions have
the controller's real [1,5] inputs and [1,256] LSTM state tensors, and the
update retains the production four PPO epochs.  It is a development-only
instrument, never a manuscript result generator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrl_fast.agent import MetaController
from amrl_fast.rebuilt_lstm import RebuiltLSTMBackend
from amrl_fast.telemetry import SynchronizedSegmentTimer


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_transitions(controller: MetaController) -> list:
    """Make one production-format four-window outer PPO rollout."""
    transitions = []
    # These values only stand in for emitted inner-loop diagnostics; they use
    # the production five-feature interface, batch size one, and controller
    # state/action/log-probability objects without changing its algorithm.
    diagnostics = (
        (10.0, 1.0, 0.5, 0.1, 0.2),
        (11.0, 1.1, 0.6, 0.11, 0.19),
        (12.0, 0.9, 0.4, 0.12, 0.21),
        (13.0, 1.2, 0.7, 0.09, 0.18),
    )
    for index, values in enumerate(diagnostics):
        feature = torch.tensor([values], dtype=torch.float32, device=controller.device)
        _, transition = controller.act(feature, activation_weight=1.0)
        transition.reward = float(index + 1) / 10.0
        transitions.append(transition)
    return transitions


def _profile_path(device: torch.device, *, use_fast: bool, warmup: int, repeats: int) -> dict:
    torch.manual_seed(71)
    backend = RebuiltLSTMBackend()
    calibration = None
    if use_fast:
        calibration = backend.calibrate(device, warmup=warmup, repeats=repeats)
        if not calibration.admitted:
            raise RuntimeError(f"Rebuilt backend not admitted: {calibration.reason}")
    # Calibration deliberately seeds its test tensors; restore the paired
    # controller initialization before creating either controller.
    torch.manual_seed(71)
    controller = MetaController(device=device, backend=backend, use_fast=use_fast)
    # Warm the complete production PPO meta-update (including Adam state
    # allocation) before collecting per-segment timings.  LSTM calibration
    # alone does not warm policy/value layers or optimizer buffers.
    warm_transitions = _make_transitions(controller)
    controller.update(warm_transitions)
    _sync(device)
    transitions = _make_transitions(controller)
    segments = SynchronizedSegmentTimer(device)
    segments.add_zero("fidelity_runtime_checks_deferred_to_meta_update")
    segments.add_zero("fallback_bookkeeping")
    segments.add_zero("profiling_logging_not_in_meta_update")
    _sync(device)
    import time
    started = time.perf_counter()
    gradient_norm = controller.update(transitions, segments=segments)
    _sync(device)
    total_us = (time.perf_counter() - started) * 1e6
    # This is the production deferred health guard.  It reuses the scalar norm
    # produced by the optimizer step and adds no CUDA reduction.
    health_started = time.perf_counter()
    healthy = backend.note_meta_update_health(gradient_norm) if use_fast else True
    segments.seconds["fidelity_runtime_checks_deferred_to_meta_update"] += time.perf_counter() - health_started
    segments.calls["fidelity_runtime_checks_deferred_to_meta_update"] += 1
    if not healthy:
        backend.fallback_count += 1
        segments.calls["fallback_bookkeeping"] += 1
    cell = controller.cell
    signature = {
        "feature_shape": [1, 5],
        "hidden_shape": [1, 256],
        "cell_shape": [1, 256],
        "weight_ih_shape": list(cell.weight_ih.shape),
        "weight_hh_shape": list(cell.weight_hh.shape),
        "batch_size": 1,
        "dtype": str(cell.weight_ih.dtype).replace("torch.", ""),
        "device": str(device),
        "requires_grad": {
            "features": False,
            "weight_ih": bool(cell.weight_ih.requires_grad),
            "weight_hh": bool(cell.weight_hh.requires_grad),
        },
        "tensor_layout": {
            "weight_ih_contiguous": bool(cell.weight_ih.is_contiguous()),
            "weight_hh_contiguous": bool(cell.weight_hh.is_contiguous()),
        },
        "outer_ppo_epochs": 4,
        "stored_transitions": len(transitions),
        "optimizer": type(controller.optimizer).__name__,
    }
    return {
        "path": "AMRL-Fast" if use_fast else "AMRL",
        "development_only": True,
        "full_meta_update_us": total_us,
        "gradient_norm": gradient_norm,
        "deferred_health_check_passed": healthy,
        "segments_us": segments.microseconds(),
        "call_signature": signature,
        "backend": backend.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration-warmup", type=int, default=30)
    parser.add_argument("--calibration-repeats", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for synchronized GPU diagnostic timing.")
    standard = _profile_path(device, use_fast=False, warmup=args.calibration_warmup, repeats=args.calibration_repeats)
    fast = _profile_path(device, use_fast=True, warmup=args.calibration_warmup, repeats=args.calibration_repeats)
    payload = {
        "development_only": True,
        "paper_result_eligible": False,
        "timing_rule": "Every GPU segment is bracketed by CUDA synchronization; CPU-only bookkeeping is separately timed without a synthetic synchronization.",
        "standard": standard,
        "fast": fast,
        "integrated_meta_update_speedup": standard["full_meta_update_us"] / fast["full_meta_update_us"],
        "integrated_meta_update_reduced": fast["full_meta_update_us"] < standard["full_meta_update_us"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["integrated_meta_update_reduced"]:
        raise SystemExit("Integrated Fast meta update is not yet faster than standard.")


if __name__ == "__main__":
    main()
