"""Executable public AlphaTensor 4x4x4 factorization.

The public DeepMind release supplies a GPU-tailored rank-49 *factorization*
for a V100/8192-square benchmark.  It does not supply an ``AT-4-4-4-v2``
CUDA, Triton, or C++ extension.  In particular, the ``AT-4-4-4-v2`` label in
the public AMRL-Fast repository is only registry metadata: that repository
explicitly routes it to ``torch.matmul``.

This module therefore keeps two identities separate:

* ``AT-4-4-4-public-v100`` executes the exact public V100-oriented factors;
* ``AT-4-4-4-v2`` is deliberately rejected until its actual artifact is
  supplied.  Mapping that label to a reference implementation would make a
  reproducibility claim that the available code cannot support.

The public path is vectorized PyTorch (no Python loop over ranks or tiles) and
preserves autograd.  It is a correctness/reference path, not a claim of a
fused deployment kernel.  ``OperatorBackend`` still requires a per-device
speedup measurement before any training run can be called AMRL-Fast.
"""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


PUBLIC_V100_ALGORITHMS = frozenset({
    "AT-4-4-4-public-v100",
    "alphatensor-4x4x4-v100",
})
UNAVAILABLE_V2_ALGORITHMS = frozenset({"AT-4-4-4-v2"})


def _factor_source_path() -> Path:
    """Return the official source that defines the V100-specific factors."""
    explicit = os.getenv("AMRL_FAST_ALPHATENSOR_FACTOR_SOURCE")
    if explicit:
        path = Path(explicit)
    else:
        path = (
            Path(__file__).resolve().parents[2]
            / "third_party"
            / "alphatensor"
            / "benchmarking"
            / "factorizations.py"
        )
    if not path.is_file():
        raise FileNotFoundError(
            "The public AlphaTensor V100 factor source is missing. Clone "
            "https://github.com/google-deepmind/alphatensor.git into "
            "third_party/alphatensor or set AMRL_FAST_ALPHATENSOR_FACTOR_SOURCE."
        )
    return path


@lru_cache(maxsize=1)
def _public_v100_factorization() -> np.ndarray:
    """Load ``get_4x4x4_alphatensor_gpu`` from the official public release."""
    path = _factor_source_path()
    spec = importlib.util.spec_from_file_location("_alphatensor_public_factors", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load AlphaTensor factor source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = np.asarray(module.get_4x4x4_alphatensor_gpu(), dtype=np.float32)
    if values.shape != (3, 16, 49):
        raise ValueError(f"Expected [3, 16, 49] public V100 factors, got {values.shape}.")
    return values


@lru_cache(maxsize=8)
def _factors(
    device_type: str,
    device_index: int | None,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # ``w`` is transposed exactly as in DeepMind's algorithm_from_factors.
    values = _public_v100_factorization()
    device = torch.device(device_type if device_index is None else f"{device_type}:{device_index}")
    u = torch.as_tensor(values[0].reshape(4, 4, 49), device=device, dtype=dtype)
    v = torch.as_tensor(values[1].reshape(4, 4, 49), device=device, dtype=dtype)
    w = torch.as_tensor(values[2].reshape(4, 4, 49).transpose(1, 0, 2), device=device, dtype=dtype)
    return u, v, w


def _pad_to_multiple_of_four(value: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    padded = value.new_zeros(rows, columns)
    padded[: value.shape[0], : value.shape[1]] = value
    return padded


def _factorized_matmul(
    left: torch.Tensor,
    right: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    """Execute a vectorized rank-49 4x4 block decomposition.

    The three ``einsum`` calls and the batched block products are separate
    PyTorch GPU operations.  This is intentionally explicit because it makes
    the missing kernel fusion observable in profiling.
    """
    rows, inner, columns = left.shape[0], left.shape[1], right.shape[1]
    padded_rows = ((rows + 3) // 4) * 4
    padded_inner = ((inner + 3) // 4) * 4
    padded_columns = ((columns + 3) // 4) * 4
    a = _pad_to_multiple_of_four(left, padded_rows, padded_inner)
    b = _pad_to_multiple_of_four(right, padded_inner, padded_columns)
    row_block, inner_block, column_block = padded_rows // 4, padded_inner // 4, padded_columns // 4
    a_blocks = a.reshape(4, row_block, 4, inner_block).permute(0, 2, 1, 3)
    b_blocks = b.reshape(4, inner_block, 4, column_block).permute(0, 2, 1, 3)
    product_left = torch.einsum("ijr,ijab->rab", u, a_blocks)
    product_right = torch.einsum("ijr,ijab->rab", v, b_blocks)
    products = torch.bmm(product_left, product_right)
    c_blocks = torch.einsum("ijr,rab->ijab", w, products)
    output = c_blocks.permute(0, 2, 1, 3).reshape(padded_rows, padded_columns)
    return output[:rows, :columns]


@lru_cache(maxsize=8)
def _compiled_algorithm(
    device_type: str,
    device_index: int | None,
    dtype: torch.dtype,
):
    """Compile the same reference graph only when explicitly requested.

    This is not a substitute for the unpublished v2 kernel.  It requires a
    hardware/software combination supported by TorchInductor/Triton.
    """
    u, v, w = _factors(device_type, device_index, dtype)

    def operation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return _factorized_matmul(left, right, u, v, w)

    return torch.compile(operation, mode="reduce-overhead", fullgraph=True)


def matmul(
    left: torch.Tensor,
    right: torch.Tensor,
    algorithm: str = "AT-4-4-4-public-v100",
) -> torch.Tensor:
    """Multiply matrices with the public, V100-specific AlphaTensor factors."""
    if algorithm in UNAVAILABLE_V2_ALGORITHMS:
        raise RuntimeError(
            "AT-4-4-4-v2 has no implementation artifact in the public AMRL-Fast "
            "repository. Provide its CUDA/Triton/C++ extension and dispatch it "
            "explicitly; the public V100 factorization is not silently substituted."
        )
    if algorithm not in PUBLIC_V100_ALGORITHMS:
        raise ValueError(f"Unsupported AlphaTensor algorithm identifier: {algorithm}")
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("AlphaTensor backend expects compatible two-dimensional matrices.")
    if left.device != right.device or left.dtype != right.dtype:
        raise ValueError("AlphaTensor operands must share device and dtype.")
    if not left.dtype.is_floating_point:
        raise TypeError("AlphaTensor backend requires floating-point tensors.")
    u, v, w = _factors(left.device.type, left.device.index, left.dtype)
    if os.getenv("AMRL_FAST_TORCH_COMPILE", "0") == "1":
        return _compiled_algorithm(left.device.type, left.device.index, left.dtype)(left, right)
    return _factorized_matmul(left, right, u, v, w)
