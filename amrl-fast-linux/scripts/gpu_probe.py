#!/usr/bin/env python3
"""Minimal CUDA sanity check used before any AMRL experiment is launched."""

from __future__ import annotations

import json

import torch


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable to PyTorch in the amrl-fast environment.")
    device = torch.device("cuda:0")
    matrix = torch.randn(1024, 1024, device=device)
    result = matrix @ matrix
    torch.cuda.synchronize(device)
    if not torch.isfinite(result).all():
        raise SystemExit("CUDA matrix-product output contains a non-finite value.")
    print(json.dumps({
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "capability": torch.cuda.get_device_capability(device),
        "finite_output": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
