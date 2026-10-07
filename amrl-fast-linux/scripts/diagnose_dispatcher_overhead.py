#!/usr/bin/env python3
"""Compare raw and dispatched LSTM execution on the real controller tensors."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrl_fast.rebuilt_lstm import RebuiltLSTMBackend, rebuilt_native_lstm_cell


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time(operation, tensors, device: torch.device, warmup: int, repeats: int) -> float:
    values = tuple(value.detach().clone().requires_grad_(index < 6) for index, value in enumerate(tensors))
    def invoke() -> None:
        for value in values:
            value.grad = None
        h, c = operation(*values)
        (h.square().mean() + c.square().mean()).backward()
    for _ in range(warmup):
        invoke()
    _sync(device)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        invoke()
        _sync(device)
        samples.append(time.perf_counter() - started)
    return 1e6 * statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--calibration-warmup", type=int, default=30)
    parser.add_argument("--calibration-repeats", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    torch.manual_seed(101)
    tensors = (
        torch.randn(1, 5, device=device), torch.randn(1, 256, device=device), torch.randn(1, 256, device=device),
        torch.randn(1024, 5, device=device), torch.randn(1024, 256, device=device), torch.randn(1024, device=device),
        torch.zeros(1024, device=device),
    )
    backend = RebuiltLSTMBackend()
    report = backend.calibrate(
        device, warmup=args.calibration_warmup, repeats=args.calibration_repeats
    )
    if not report.admitted:
        raise SystemExit(report.reason)
    raw_us = _time(rebuilt_native_lstm_cell, tensors, device, args.warmup, args.repeats)
    dispatched_us = _time(
        lambda *values: backend.run(*values, activation_weight=1.0),
        tensors,
        device,
        args.warmup,
        args.repeats,
    )
    payload = {
        "development_only": True,
        "shape": {"input": [1, 5], "hidden_cell": [1, 256], "weight_ih": [1024, 5], "weight_hh": [1024, 256]},
        "backend": backend.to_dict(),
        "raw_native_forward_backward_us": raw_us,
        "dispatched_forward_backward_us": dispatched_us,
        "dispatcher_overhead_us": dispatched_us - raw_us,
        "dispatcher_overhead_ratio": dispatched_us / raw_us - 1.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
