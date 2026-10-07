"""Server-side numerical validation for the public AlphaTensor backend."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from amrl_fast.alphatensor_kernel import matmul


def test_alphatensor_matches_standard_output_and_gradient() -> None:
    torch.manual_seed(17)
    left = torch.randn(3, 256, dtype=torch.float32, requires_grad=True)
    right = torch.randn(256, 256, dtype=torch.float32, requires_grad=True)
    candidate = matmul(left, right)
    candidate.square().mean().backward()
    candidate_left_gradient = left.grad.detach().clone()
    reference_left = left.detach().clone().requires_grad_(True)
    reference_right = right.detach().clone().requires_grad_(True)
    reference = reference_left @ reference_right
    reference.square().mean().backward()
    assert torch.allclose(candidate, reference, rtol=5e-4, atol=5e-4)
    assert torch.allclose(candidate_left_gradient, reference_left.grad, rtol=8e-4, atol=8e-4)


def test_unpublished_v2_identifier_is_not_silently_substituted() -> None:
    left = torch.randn(1, 256)
    right = torch.randn(256, 256)
    with pytest.raises(RuntimeError, match="no implementation artifact"):
        matmul(left, right, algorithm="AT-4-4-4-v2")
