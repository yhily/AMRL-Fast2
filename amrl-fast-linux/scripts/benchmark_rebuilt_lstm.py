#!/usr/bin/env python3
"""Benchmark real AMRL meta-controller LSTM execution candidates.

The default tensors mirror the public AMRL-Fast controller exactly:
``[batch, 5]`` input, ``[batch, 256]`` hidden/cell state, a 1024-by-5 input
weight, and a 1024-by-256 recurrent weight.  This script never writes paper
results; it writes a versioned kernel-development JSON record only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrl_fast.rebuilt_lstm import (
    public_manual_lstm_cell,
    public_rank49_reference_lstm_cell,
    rebuilt_addmm_lstm_cell,
    rebuilt_native_lstm_cell,
    trace_from_tensors,
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.detach().reshape(-1).float()
    second = second.detach().reshape(-1).float()
    first = first - first.mean()
    second = second - second.mean()
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    return float(torch.clamp(torch.dot(first, second) / denominator.clamp_min(1e-12), -1.0, 1.0).item())


def median_us(samples: list[float]) -> float:
    return 1e6 * statistics.median(samples)


def mean_us(samples: list[float]) -> float:
    return 1e6 * statistics.fmean(samples)


def time_forward(
    operation: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> list[float]:
    with torch.no_grad():
        for _ in range(warmup):
            operation()
        synchronize(device)
        timings = []
        for _ in range(repeats):
            started = time.perf_counter()
            operation()
            synchronize(device)
            timings.append(time.perf_counter() - started)
    return timings


def time_total(
    operation: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    tensors: tuple[torch.Tensor, ...],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> list[float]:
    working = tuple(
        value.detach().clone().requires_grad_(index < 6)
        for index, value in enumerate(tensors)
    )

    def run_once() -> None:
        for value in working:
            value.grad = None
        output_hidden, output_cell = operation(*working)
        (output_hidden.square().mean() + output_cell.square().mean()).backward()

    for _ in range(warmup):
        run_once()
    synchronize(device)
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        run_once()
        synchronize(device)
        timings.append(time.perf_counter() - started)
    return timings


def time_backward(
    name: str,
    operation: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    tensors: tuple[torch.Tensor, ...],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> list[float]:
    """Time backward alone; end-to-end admission still uses ``time_total``."""
    def build_loss() -> torch.Tensor:
        leaves = tuple(
            value.detach().clone().requires_grad_(index < 6)
            for index, value in enumerate(tensors)
        )
        output_hidden, output_cell = invoke(name, operation, leaves)
        return output_hidden.square().mean() + output_cell.square().mean()

    for _ in range(warmup):
        build_loss().backward()
    synchronize(device)
    timings = []
    for _ in range(repeats):
        loss = build_loss()
        synchronize(device)
        started = time.perf_counter()
        loss.backward()
        synchronize(device)
        timings.append(time.perf_counter() - started)
    return timings


def invoke(
    name: str,
    operation: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    tensors: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    if name == "rebuilt_native_lstm":
        return operation(*tensors)
    return operation(*tensors[:6])


def output_and_gradient_signature(
    name: str,
    operation: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    tensors: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    leaves = tuple(
        value.detach().clone().requires_grad_(index < 6)
        for index, value in enumerate(tensors)
    )
    outputs = invoke(name, operation, leaves)
    combined = torch.cat(outputs, dim=1)
    combined.square().mean().backward()
    gradient = torch.cat([value.grad.detach().reshape(-1) for value in leaves[:6]])
    return combined.detach(), gradient


def evaluate(
    name: str,
    operation: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    tensors: tuple[torch.Tensor, ...],
    reference_output: torch.Tensor,
    reference_gradient: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict:
    call = lambda: invoke(name, operation, tensors)
    output, gradient = output_and_gradient_signature(name, operation, tensors)
    output_difference = reference_output - output
    gradient_difference = reference_gradient - gradient
    forward_samples = time_forward(call, device, warmup, repeats)
    total_samples = time_total(operation, tensors if name == "rebuilt_native_lstm" else tensors[:-1], device, warmup, repeats)
    backward_samples = time_backward(name, operation, tensors, device, warmup, repeats)
    return {
        "version": name,
        "forward_us_median": median_us(forward_samples),
        "forward_us_mean": mean_us(forward_samples),
        "backward_us_median": median_us(backward_samples),
        "backward_us_mean": mean_us(backward_samples),
        "forward_backward_us_median": median_us(total_samples),
        "forward_backward_us_mean": mean_us(total_samples),
        "output_correlation": correlation(reference_output, output),
        "relative_l2_error": float(
            (torch.linalg.vector_norm(output_difference) / torch.linalg.vector_norm(reference_output).clamp_min(1e-12)).item()
        ),
        "max_absolute_error": float(output_difference.abs().max().item()),
        "gradient_correlation": correlation(reference_gradient, gradient),
        "gradient_relative_l2_error": float(
            (torch.linalg.vector_norm(gradient_difference) / torch.linalg.vector_norm(reference_gradient).clamp_min(1e-12)).item()
        ),
        "gradient_max_absolute_error": float(gradient_difference.abs().max().item()),
        "gradient_finite": bool(torch.isfinite(gradient).all().item()),
        "output_finite": bool(torch.isfinite(output).all().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("artifacts/at_rebuild/benchmark_v1.json"))
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    torch.manual_seed(41)
    batch = args.batch_size
    inputs = torch.randn(batch, 5, device=device)
    hidden = torch.randn(batch, 256, device=device)
    cell = torch.randn(batch, 256, device=device)
    weight_ih = torch.randn(1024, 5, device=device)
    weight_hh = torch.randn(1024, 256, device=device)
    bias = torch.randn(1024, device=device)
    bias_hh = torch.zeros(1024, device=device)
    all_tensors = (inputs, hidden, cell, weight_ih, weight_hh, bias, bias_hh)
    reference_output, reference_gradient = output_and_gradient_signature(
        "public_manual_baseline", public_manual_lstm_cell, all_tensors
    )
    results = []
    for name, operation in (
        ("public_manual_baseline", public_manual_lstm_cell),
        ("public_rank49_reference", public_rank49_reference_lstm_cell),
        ("rebuilt_addmm", rebuilt_addmm_lstm_cell),
        ("rebuilt_native_lstm", rebuilt_native_lstm_cell),
    ):
        results.append(
            evaluate(
                name,
                operation,
                all_tensors,
                reference_output,
                reference_gradient,
                device,
                args.warmup,
                args.repeats,
            )
        )
    baseline_total = results[0]["forward_backward_us_median"]
    for entry in results:
        entry["forward_backward_speedup_vs_baseline"] = baseline_total / entry["forward_backward_us_median"]
        entry["admitted"] = bool(
            entry["output_correlation"] >= 0.999
            and entry["gradient_correlation"] >= 0.99
            and entry["relative_l2_error"] <= 5e-3
            and entry["max_absolute_error"] <= 1e-1
            and entry["gradient_relative_l2_error"] <= 5e-3
            and entry["gradient_finite"]
            and entry["forward_backward_speedup_vs_baseline"] >= 1.02
        )
    trace = trace_from_tensors(*all_tensors[:5])
    trace.forward_calls = args.warmup + args.repeats
    trace.backward_calls = args.warmup + args.repeats
    payload = {
        "kernel_development_only": True,
        "shape_trace": trace.to_dict(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "cuda_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
