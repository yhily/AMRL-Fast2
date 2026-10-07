"""Numerical invariants for the fused four-gate execution backend."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from amrl_fast.rebuilt_lstm import (
    public_manual_lstm_cell,
    rebuilt_native_lstm_cell,
)


def _run(operation, tensors, *, native: bool):
    leaves = [value.detach().clone().requires_grad_(index < 6) for index, value in enumerate(tensors)]
    output = operation(*leaves) if native else operation(*leaves[:6])
    loss = output[0].square().mean() + output[1].square().mean()
    loss.backward()
    return output, [value.grad.detach().clone() for value in leaves[:6]]


def test_native_lstm_cell_matches_manual_four_gate_equation_and_gradients():
    torch.manual_seed(29)
    tensors = (
        torch.randn(1, 5),
        torch.randn(1, 256),
        torch.randn(1, 256),
        torch.randn(1024, 5),
        torch.randn(1024, 256),
        torch.randn(1024),
        torch.zeros(1024),
    )
    manual_output, manual_gradients = _run(public_manual_lstm_cell, tensors, native=False)
    fused_output, fused_gradients = _run(rebuilt_native_lstm_cell, tensors, native=True)
    assert torch.allclose(fused_output[0], manual_output[0], rtol=5e-5, atol=5e-6)
    assert torch.allclose(fused_output[1], manual_output[1], rtol=5e-5, atol=5e-6)
    for fused, manual in zip(fused_gradients, manual_gradients):
        assert torch.allclose(fused, manual, rtol=5e-5, atol=5e-6)
