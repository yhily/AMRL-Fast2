#!/usr/bin/env python3
"""Profile only the public AlphaTensor 4x4x4 reference execution path.

This utility is intentionally independent of AMRL.  It measures the exact
operator shape supplied on the command line, uses CUDA synchronization, and
prints machine-readable JSON.  It is diagnostic evidence: it does not admit a
backend or create experiment results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrl_fast import alphatensor_kernel as at


def parse_shape(value: str) -> tuple[int, int, int]:
    pieces = tuple(int(part.strip()) for part in value.split(","))
    if len(pieces) != 3 or any(part <= 0 for part in pieces):
        raise argparse.ArgumentTypeError("shape must be positive M,N,K (for example 1,256,256)")
    return pieces


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(operation, device: torch.device, *, warmup: int, measured: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            result = operation()
            if not torch.isfinite(result).all():
                raise FloatingPointError("non-finite intermediate during diagnostic warmup")
        synchronize(device)
        started = time.perf_counter()
        for _ in range(measured):
            result = operation()
            if not torch.isfinite(result).all():
                raise FloatingPointError("non-finite intermediate during diagnostic timing")
        synchronize(device)
    return (time.perf_counter() - started) / measured


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=parse_shape, default=(1, 256, 256))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measured", type=int, default=25)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    m, n, k = args.shape
    left = torch.randn(m, n, device=device)
    right = torch.randn(n, k, device=device)
    u, v, w = at._factors(left.device.type, left.device.index, left.dtype)

    def layout():
        padded_rows = ((m + 3) // 4) * 4
        padded_inner = ((n + 3) // 4) * 4
        padded_columns = ((k + 3) // 4) * 4
        a = at._pad_to_multiple_of_four(left, padded_rows, padded_inner)
        b = at._pad_to_multiple_of_four(right, padded_inner, padded_columns)
        row_block, inner_block, column_block = padded_rows // 4, padded_inner // 4, padded_columns // 4
        a_blocks = a.reshape(4, row_block, 4, inner_block).permute(0, 2, 1, 3)
        b_blocks = b.reshape(4, inner_block, 4, column_block).permute(0, 2, 1, 3)
        return a_blocks, b_blocks

    a_blocks, b_blocks = layout()
    product_left = torch.einsum("ijr,ijab->rab", u, a_blocks)
    product_right = torch.einsum("ijr,ijab->rab", v, b_blocks)
    products = torch.bmm(product_left, product_right)

    timings = {
        "standard_matmul_seconds": benchmark(lambda: left @ right, device, warmup=args.warmup, measured=args.measured),
        "reference_full_seconds": benchmark(
            lambda: at.matmul(left, right, "AT-4-4-4-public-v100"),
            device,
            warmup=args.warmup,
            measured=args.measured,
        ),
        "padding_and_block_layout_seconds": benchmark(
            lambda: layout()[0], device, warmup=args.warmup, measured=args.measured
        ),
        "left_linear_combinations_seconds": benchmark(
            lambda: torch.einsum("ijr,ijab->rab", u, a_blocks),
            device,
            warmup=args.warmup,
            measured=args.measured,
        ),
        "right_linear_combinations_seconds": benchmark(
            lambda: torch.einsum("ijr,ijab->rab", v, b_blocks),
            device,
            warmup=args.warmup,
            measured=args.measured,
        ),
        "rank49_block_products_seconds": benchmark(
            lambda: torch.bmm(product_left, product_right),
            device,
            warmup=args.warmup,
            measured=args.measured,
        ),
        "output_recombination_seconds": benchmark(
            lambda: torch.einsum("ijr,rab->ijab", w, products),
            device,
            warmup=args.warmup,
            measured=args.measured,
        ),
    }
    reference = left @ right
    candidate = at.matmul(left, right, "AT-4-4-4-public-v100")
    difference = reference - candidate
    payload = {
        "algorithm": "AT-4-4-4-public-v100",
        "factor_source": str(at._factor_source_path()),
        "shape": [m, n, k],
        "device": str(device),
        "cuda_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
        "relative_l2_error": float(
            (torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference).clamp_min(1e-12)).item()
        ),
        "max_absolute_error": float(difference.abs().max().item()),
        "timings": timings,
        "reference_speedup": timings["standard_matmul_seconds"] / timings["reference_full_seconds"],
        "note": (
            "The stage timings overlap and are diagnostic only; the reference path has no CUDA/Triton/C++ fusion."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
