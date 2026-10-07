#!/usr/bin/env python3
"""Capture operator-level development profiles for real LSTM controller shapes.

The output is a kernel-development record, not a paper profiling result.  It
exists to make the selected execution route and its operator composition
auditable before the formal paired runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrl_fast.rebuilt_lstm import public_manual_lstm_cell, rebuilt_native_lstm_cell


def _rows(profile, limit: int) -> list[dict]:
    rows = []
    for event in profile.key_averages():
        rows.append(
            {
                "operator": event.key,
                "calls": int(event.count),
                "self_cpu_us": float(event.self_cpu_time_total),
                "cpu_total_us": float(event.cpu_time_total),
                "self_cuda_us": float(getattr(event, "self_cuda_time_total", 0.0)),
                "cuda_total_us": float(getattr(event, "cuda_time_total", 0.0)),
            }
        )
    return sorted(rows, key=lambda row: row["self_cuda_us"], reverse=True)[:limit]


def _profile_route(name: str, operation, tensors, iterations: int, device: torch.device) -> dict:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True) as profile:
        for _ in range(iterations):
            leaves = tuple(value.detach().clone().requires_grad_(index < 6) for index, value in enumerate(tensors))
            h_out, c_out = operation(*leaves)
            (h_out.square().mean() + c_out.square().mean()).backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"route": name, "top_operators": _rows(profile, 20)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    torch.manual_seed(73)
    tensors = (
        torch.randn(1, 5, device=device),
        torch.randn(1, 256, device=device),
        torch.randn(1, 256, device=device),
        torch.randn(1024, 5, device=device),
        torch.randn(1024, 256, device=device),
        torch.randn(1024, device=device),
        torch.zeros(1024, device=device),
    )
    routes = [
        _profile_route(
            "public-manual-baseline",
            lambda *values: public_manual_lstm_cell(*values[:6]),
            tensors,
            args.iterations,
            device,
        ),
        _profile_route("native-lstm-cell", rebuilt_native_lstm_cell, tensors, args.iterations, device),
    ]
    payload = {
        "kernel_development_only": True,
        "shape": {
            "input": [1, 5], "hidden_cell": [1, 256],
            "weight_ih": [1024, 5], "weight_hh": [1024, 256],
        },
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "iterations": args.iterations,
        "cuda_activity_available": any(
            row["cuda_total_us"] > 0 for route in routes for row in route["top_operators"]
        ),
        "routes": routes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
