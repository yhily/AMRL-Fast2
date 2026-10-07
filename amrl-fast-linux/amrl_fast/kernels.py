"""Measured and fidelity-gated operator substitution for AMRL-Fast.

No run is labelled ``AMRL-Fast`` unless the configured backend is a real
callable, has a matching autograd path, and is faster under the same shapes
used by the meta-controller.  A normal ``torch.matmul`` fallback is useful for
debugging only and is deliberately not admitted as an accelerated backend.
"""

from __future__ import annotations

import importlib
import os
import time
from dataclasses import asdict, dataclass
from typing import Protocol

import torch


class MatmulCallable(Protocol):
    def __call__(self, left: torch.Tensor, right: torch.Tensor, algorithm: str) -> torch.Tensor: ...


@dataclass
class FidelityReport:
    available: bool
    admitted: bool
    backend_module: str | None
    output_correlation: float | None
    gradient_correlation: float | None
    relative_l2_error: float | None
    max_absolute_error: float | None
    standard_seconds: float | None
    optimized_seconds: float | None
    measured_speedup: float | None
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


class OperatorBackend:
    """Dispatch one fixed-shape meta-controller MatMul through an admitted backend."""

    def __init__(
        self,
        *,
        algorithm: str = "AT-4-4-4-public-v100",
        module_name: str | None = None,
        output_correlation_threshold: float = 0.999,
        gradient_correlation_threshold: float = 0.99,
        relative_l2_threshold: float = 5e-3,
        max_absolute_threshold: float = 1e-1,
        minimum_speedup: float = 1.02,
    ) -> None:
        self.algorithm = algorithm
        self.module_name = module_name or os.getenv("AMRL_FAST_KERNEL_MODULE")
        self.output_correlation_threshold = output_correlation_threshold
        self.gradient_correlation_threshold = gradient_correlation_threshold
        self.relative_l2_threshold = relative_l2_threshold
        self.max_absolute_threshold = max_absolute_threshold
        self.minimum_speedup = minimum_speedup
        self._kernel: MatmulCallable | None = None
        self.report = FidelityReport(
            available=False,
            admitted=False,
            backend_module=self.module_name,
            output_correlation=None,
            gradient_correlation=None,
            relative_l2_error=None,
            max_absolute_error=None,
            standard_seconds=None,
            optimized_seconds=None,
            measured_speedup=None,
            reason="No optimized backend configured.",
        )
        self.dispatch_count = 0
        self.fallback_count = 0
        self.nan_inf_count = 0

    @staticmethod
    def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
        left = left.detach().reshape(-1).float()
        right = right.detach().reshape(-1).float()
        left = left - left.mean()
        right = right - right.mean()
        denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
        if denominator.item() == 0:
            return 1.0 if torch.allclose(left, right) else 0.0
        return float(torch.clamp(torch.dot(left, right) / denominator, -1.0, 1.0).item())

    @staticmethod
    def _synchronize(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _load_kernel(self) -> MatmulCallable | None:
        if not self.module_name:
            return None
        module = importlib.import_module(self.module_name)
        kernel = getattr(module, "matmul", None)
        if kernel is None or not callable(kernel):
            raise TypeError(
                f"{self.module_name} must expose callable matmul(left, right, algorithm)."
            )
        return kernel

    def _time(self, op, device: torch.device, *, warmup: int = 5, measured: int = 25) -> float:
        for _ in range(warmup):
            value = op()
            if not torch.isfinite(value).all():
                raise FloatingPointError("Non-finite output during kernel warmup.")
        self._synchronize(device)
        started = time.perf_counter()
        for _ in range(measured):
            value = op()
            if not torch.isfinite(value).all():
                raise FloatingPointError("Non-finite output during kernel timing.")
        self._synchronize(device)
        return (time.perf_counter() - started) / measured

    def calibrate(self, device: torch.device, hidden_size: int = 256) -> FidelityReport:
        """Apply the manuscript's output, gradient, and synchronized timing screen."""
        try:
            self._kernel = self._load_kernel()
            if self._kernel is None:
                return self.report

            # This is the actual LSTM hidden-to-hidden operator shape: a
            # sequence/batch row multiplied by a 256x256 recurrent weight.
            left = torch.randn(1, hidden_size, device=device, requires_grad=True)
            right = torch.randn(hidden_size, hidden_size, device=device, requires_grad=True)
            ref_left = left.detach().clone().requires_grad_(True)
            ref_right = right.detach().clone().requires_grad_(True)
            reference = ref_left @ ref_right
            candidate = self._kernel(left, right, self.algorithm)
            if candidate.shape != reference.shape:
                raise ValueError(f"Kernel returned {tuple(candidate.shape)}, expected {tuple(reference.shape)}")
            if not torch.isfinite(candidate).all():
                raise FloatingPointError("Candidate output contains NaN/Inf.")
            reference.square().mean().backward()
            candidate.square().mean().backward()
            if left.grad is None or ref_left.grad is None:
                raise RuntimeError("Candidate kernel does not preserve gradients.")
            output_corr = self._correlation(reference, candidate)
            grad_corr = self._correlation(ref_left.grad, left.grad)
            diff = reference.detach() - candidate.detach()
            rel_l2 = float(
                (torch.linalg.vector_norm(diff) /
                 torch.clamp(torch.linalg.vector_norm(reference.detach()), min=1e-12)).item()
            )
            max_abs = float(diff.abs().max().item())
            with torch.no_grad():
                standard_seconds = self._time(lambda: left.detach() @ right.detach(), device)
                optimized_seconds = self._time(
                    lambda: self._kernel(left.detach(), right.detach(), self.algorithm), device
                )
            speedup = standard_seconds / optimized_seconds
            admitted = (
                output_corr >= self.output_correlation_threshold
                and grad_corr >= self.gradient_correlation_threshold
                and rel_l2 <= self.relative_l2_threshold
                and max_abs <= self.max_absolute_threshold
                and speedup >= self.minimum_speedup
            )
            reason = "Admitted by fidelity and timing screen." if admitted else (
                "Rejected: output/gradient fidelity or measured speedup did not meet the deployment rule."
            )
            self.report = FidelityReport(
                available=True,
                admitted=admitted,
                backend_module=self.module_name,
                output_correlation=output_corr,
                gradient_correlation=grad_corr,
                relative_l2_error=rel_l2,
                max_absolute_error=max_abs,
                standard_seconds=standard_seconds,
                optimized_seconds=optimized_seconds,
                measured_speedup=speedup,
                reason=reason,
            )
        except Exception as exc:
            self._kernel = None
            self.report = FidelityReport(
                available=False,
                admitted=False,
                backend_module=self.module_name,
                output_correlation=None,
                gradient_correlation=None,
                relative_l2_error=None,
                max_absolute_error=None,
                standard_seconds=None,
                optimized_seconds=None,
                measured_speedup=None,
                reason=f"Backend unavailable or invalid: {type(exc).__name__}: {exc}",
            )
        return self.report

    @property
    def optimized(self) -> bool:
        return self.report.admitted and self._kernel is not None

    def matmul(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Run the admitted path, falling back only after a runtime fault."""
        self.dispatch_count += 1
        if not self.optimized:
            return left @ right
        try:
            result = self._kernel(left, right, self.algorithm)
            if not torch.isfinite(result).all():
                self.nan_inf_count += 1
                raise FloatingPointError("Non-finite optimized result")
            return result
        except Exception:
            self.fallback_count += 1
            return left @ right
