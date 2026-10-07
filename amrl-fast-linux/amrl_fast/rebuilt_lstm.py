"""Fixed-shape LSTM execution candidates for AMRL-Fast-Rebuilt-Kernel.

All functions implement the same four-gate LSTM equations.  They differ only
in execution layout: the baseline follows the public AMRL-Fast manual path,
the AlphaTensor reference preserves the public rank-49 decomposition, and the
rebuilt candidates use a combined 1024-output gate operation.  This module is
deliberately independent of the AMRL/PPO update rule.
"""

from __future__ import annotations

import statistics
import time
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as functional

from . import alphatensor_kernel as alpha_reference
from .telemetry import SynchronizedSegmentTimer


def _gate_update(gates: torch.Tensor, cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=1)
    next_cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(cell_gate)
    next_hidden = torch.sigmoid(output_gate) * torch.tanh(next_cell)
    return next_hidden, next_cell


def public_manual_lstm_cell(
    inputs: torch.Tensor,
    hidden: torch.Tensor,
    cell: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The public AMRL-Fast equation: two GEMMs, additions, then four gates."""
    gates = inputs @ weight_ih.t()
    gates = gates + hidden @ weight_hh.t()
    gates = gates + bias
    return _gate_update(gates, cell)


def public_rank49_reference_lstm_cell(
    inputs: torch.Tensor,
    hidden: torch.Tensor,
    cell: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference-only rank-49 route for the recurrent four gate blocks.

    This intentionally retains four calls because the public rank-49 factors
    describe a square 256-by-256 operation, while the real recurrent weight is
    1024-by-256.  It is not a deployment candidate.
    """
    recurrent = torch.cat(
        [
            alpha_reference.matmul(
                hidden,
                weight_hh[index * 256 : (index + 1) * 256].t(),
                algorithm="AT-4-4-4-public-v100",
            )
            for index in range(4)
        ],
        dim=1,
    )
    gates = functional.linear(inputs, weight_ih) + recurrent + bias
    return _gate_update(gates, cell)


def rebuilt_addmm_lstm_cell(
    inputs: torch.Tensor,
    hidden: torch.Tensor,
    cell: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse recurrent GEMM and bias through one vendor addmm invocation."""
    gates = torch.addmm(bias, hidden, weight_hh.t())
    gates = gates + functional.linear(inputs, weight_ih)
    return _gate_update(gates, cell)


def rebuilt_native_lstm_cell(
    inputs: torch.Tensor,
    hidden: torch.Tensor,
    cell: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
    bias_hh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use PyTorch's native combined four-gate LSTM cell with autograd.

    ``torch._VF.lstm_cell`` is a native ATen operation. It keeps the same gate
    order and equations, consumes the same weight tensors, and delegates the
    complete forward/backward path to PyTorch rather than launching four
    Python-level gate MatMuls.
    """
    return torch._VF.lstm_cell(inputs, (hidden, cell), weight_ih, weight_hh, bias, bias_hh)


@dataclass
class LSTMShapeTrace:
    input_shape: tuple[int, ...]
    hidden_shape: tuple[int, ...]
    cell_shape: tuple[int, ...]
    weight_ih_shape: tuple[int, ...]
    weight_hh_shape: tuple[int, ...]
    dtype: str
    device: str
    batch_size: int
    forward_calls: int = 0
    backward_calls: int = 0

    def record_forward(self) -> None:
        self.forward_calls += 1

    def record_backward(self) -> None:
        self.backward_calls += 1

    def to_dict(self) -> dict:
        return asdict(self)


def trace_from_tensors(
    inputs: torch.Tensor,
    hidden: torch.Tensor,
    cell: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
) -> LSTMShapeTrace:
    return LSTMShapeTrace(
        input_shape=tuple(inputs.shape),
        hidden_shape=tuple(hidden.shape),
        cell_shape=tuple(cell.shape),
        weight_ih_shape=tuple(weight_ih.shape),
        weight_hh_shape=tuple(weight_hh.shape),
        dtype=str(inputs.dtype).replace("torch.", ""),
        device=str(inputs.device),
        batch_size=int(inputs.shape[0]),
    )



def _correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.detach().reshape(-1).float()
    second = second.detach().reshape(-1).float()
    first = first - first.mean()
    second = second - second.mean()
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    return float(torch.clamp(torch.dot(first, second) / denominator.clamp_min(1e-12), -1.0, 1.0).item())


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class RebuiltKernelReport:
    available: bool
    admitted: bool
    backend_name: str
    implementation: str | None
    candidate_results: dict[str, dict] | None
    output_correlation: float | None
    gradient_correlation: float | None
    relative_l2_error: float | None
    max_absolute_error: float | None
    gradient_relative_l2_error: float | None
    output_finite: bool | None
    gradient_finite: bool | None
    baseline_forward_us: float | None
    optimized_forward_us: float | None
    baseline_backward_us: float | None
    optimized_backward_us: float | None
    baseline_forward_backward_us: float | None
    optimized_forward_backward_us: float | None
    measured_speedup: float | None
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


class RebuiltLSTMBackend:
    """Fidelity-gated native fused LSTM backend for the real controller shape."""

    name = "AMRL-Fast-Rebuilt-Kernel"

    def __init__(self, *, minimum_speedup: float = 1.02) -> None:
        self.minimum_speedup = minimum_speedup
        self.report = RebuiltKernelReport(
            available=True,
            admitted=False,
            backend_name=self.name,
            implementation=None,
            candidate_results=None,
            output_correlation=None,
            gradient_correlation=None,
            relative_l2_error=None,
            max_absolute_error=None,
            gradient_relative_l2_error=None,
            output_finite=None,
            gradient_finite=None,
            baseline_forward_us=None,
            optimized_forward_us=None,
            baseline_backward_us=None,
            optimized_backward_us=None,
            baseline_forward_backward_us=None,
            optimized_forward_backward_us=None,
            measured_speedup=None,
            reason="Not calibrated.",
        )
        self.dispatch_count = 0
        self.fallback_count = 0
        self.nan_inf_count = 0
        self.shape_trace: LSTMShapeTrace | None = None
        self._implementation: str | None = None

    @staticmethod
    def _signature(
        operation,
        tensors: tuple[torch.Tensor, ...],
        *,
        native: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        leaves = tuple(
            value.detach().clone().requires_grad_(index < 6)
            for index, value in enumerate(tensors)
        )
        output = operation(*leaves) if native else operation(*leaves[:6])
        combined = torch.cat(output, dim=1)
        combined.square().mean().backward()
        gradients = torch.cat([value.grad.detach().reshape(-1) for value in leaves[:6]])
        return combined.detach(), gradients

    @staticmethod
    def _time(operation, device: torch.device, *, warmup: int, repeats: int) -> float:
        for _ in range(warmup):
            operation()
        _synchronize(device)
        samples: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            operation()
            _synchronize(device)
            samples.append(time.perf_counter() - started)
        return 1e6 * statistics.median(samples)

    @staticmethod
    def _time_forward_backward(operation, tensors: tuple[torch.Tensor, ...], device: torch.device, *, warmup: int, repeats: int) -> float:
        leaves = tuple(
            value.detach().clone().requires_grad_(index < 6)
            for index, value in enumerate(tensors)
        )

        def run_once() -> None:
            for value in leaves:
                value.grad = None
            h_out, c_out = operation(*leaves)
            (h_out.square().mean() + c_out.square().mean()).backward()

        return RebuiltLSTMBackend._time(run_once, device, warmup=warmup, repeats=repeats)

    @staticmethod
    def _time_forward(operation, tensors: tuple[torch.Tensor, ...], device: torch.device, *, warmup: int, repeats: int) -> float:
        frozen = tuple(value.detach() for value in tensors)
        return RebuiltLSTMBackend._time(
            lambda: operation(*frozen), device, warmup=warmup, repeats=repeats
        )

    @staticmethod
    def _time_backward(operation, tensors: tuple[torch.Tensor, ...], device: torch.device, *, warmup: int, repeats: int) -> float:
        """Measure backward separately while preserving the full-path timing above.

        Forward graph construction intentionally occurs before each timed
        backward call.  The separate figure is diagnostic only; admission is
        based on the end-to-end forward-plus-backward measurement.
        """
        def build_loss() -> torch.Tensor:
            leaves = tuple(
                value.detach().clone().requires_grad_(index < 6)
                for index, value in enumerate(tensors)
            )
            h_out, c_out = operation(*leaves)
            return h_out.square().mean() + c_out.square().mean()

        for _ in range(warmup):
            build_loss().backward()
        _synchronize(device)
        samples: list[float] = []
        for _ in range(repeats):
            loss = build_loss()
            _synchronize(device)
            started = time.perf_counter()
            loss.backward()
            _synchronize(device)
            samples.append(time.perf_counter() - started)
        return 1e6 * statistics.median(samples)

    def calibrate(
        self,
        device: torch.device,
        *,
        hidden_size: int = 256,
        warmup: int = 30,
        repeats: int = 200,
    ) -> RebuiltKernelReport:
        """Apply the manuscript fidelity/speed screen to actual LSTM tensors."""
        try:
            torch.manual_seed(607)
            inputs = torch.randn(1, 5, device=device)
            hidden = torch.randn(1, hidden_size, device=device)
            cell = torch.randn(1, hidden_size, device=device)
            weight_ih = torch.randn(4 * hidden_size, 5, device=device)
            weight_hh = torch.randn(4 * hidden_size, hidden_size, device=device)
            bias = torch.randn(4 * hidden_size, device=device)
            bias_hh = torch.zeros(4 * hidden_size, device=device)
            tensors = (inputs, hidden, cell, weight_ih, weight_hh, bias, bias_hh)
            reference_output, reference_gradient = self._signature(public_manual_lstm_cell, tensors, native=False)
            baseline_total = self._time_forward_backward(
                lambda *values: public_manual_lstm_cell(*values[:6]), tensors, device, warmup=warmup, repeats=repeats
            )
            baseline_forward = self._time_forward(
                lambda *values: public_manual_lstm_cell(*values[:6]), tensors, device, warmup=warmup, repeats=repeats
            )
            baseline_backward = self._time_backward(
                lambda *values: public_manual_lstm_cell(*values[:6]), tensors, device, warmup=warmup, repeats=repeats
            )
            candidates = (
                ("addmm-four-gate", lambda *values: rebuilt_addmm_lstm_cell(*values[:6]), False),
                ("native-lstm-cell", rebuilt_native_lstm_cell, True),
            )
            candidate_results: dict[str, dict] = {}
            admitted_candidates: list[dict] = []
            for implementation, operation, native in candidates:
                candidate_output, candidate_gradient = self._signature(operation, tensors, native=native)
                output_difference = reference_output - candidate_output
                gradient_difference = reference_gradient - candidate_gradient
                optimized_total = self._time_forward_backward(operation, tensors, device, warmup=warmup, repeats=repeats)
                optimized_forward = self._time_forward(operation, tensors, device, warmup=warmup, repeats=repeats)
                optimized_backward = self._time_backward(operation, tensors, device, warmup=warmup, repeats=repeats)
                output_corr = _correlation(reference_output, candidate_output)
                grad_corr = _correlation(reference_gradient, candidate_gradient)
                rel_l2 = float((torch.linalg.vector_norm(output_difference) / torch.linalg.vector_norm(reference_output).clamp_min(1e-12)).item())
                max_abs = float(output_difference.abs().max().item())
                grad_rel_l2 = float((torch.linalg.vector_norm(gradient_difference) / torch.linalg.vector_norm(reference_gradient).clamp_min(1e-12)).item())
                output_finite = bool(torch.isfinite(candidate_output).all().item())
                gradient_finite = bool(torch.isfinite(candidate_gradient).all().item())
                speedup = baseline_total / optimized_total
                admitted = bool(
                    output_corr >= 0.999
                    and grad_corr >= 0.99
                    and rel_l2 <= 5e-3
                    and max_abs <= 1e-1
                    and grad_rel_l2 <= 5e-3
                    and output_finite
                    and gradient_finite
                    and speedup >= self.minimum_speedup
                )
                entry = {
                    "implementation": implementation,
                    "admitted": admitted,
                    "output_correlation": output_corr,
                    "gradient_correlation": grad_corr,
                    "relative_l2_error": rel_l2,
                    "max_absolute_error": max_abs,
                    "gradient_relative_l2_error": grad_rel_l2,
                    "output_finite": output_finite,
                    "gradient_finite": gradient_finite,
                    "forward_us": optimized_forward,
                    "backward_us": optimized_backward,
                    "forward_backward_us": optimized_total,
                    "forward_backward_speedup": speedup,
                }
                candidate_results[implementation] = entry
                if admitted:
                    admitted_candidates.append(entry)
            selected = max(admitted_candidates, key=lambda item: item["forward_backward_speedup"], default=None)
            self._implementation = selected["implementation"] if selected else None
            if selected is None:
                output_corr = grad_corr = rel_l2 = max_abs = grad_rel_l2 = None
                output_finite = gradient_finite = None
                optimized_forward = optimized_backward = optimized_total = speedup = None
            else:
                output_corr = selected["output_correlation"]
                grad_corr = selected["gradient_correlation"]
                rel_l2 = selected["relative_l2_error"]
                max_abs = selected["max_absolute_error"]
                grad_rel_l2 = selected["gradient_relative_l2_error"]
                output_finite = selected["output_finite"]
                gradient_finite = selected["gradient_finite"]
                optimized_forward = selected["forward_us"]
                optimized_backward = selected["backward_us"]
                optimized_total = selected["forward_backward_us"]
                speedup = selected["forward_backward_speedup"]
            self.report = RebuiltKernelReport(
                available=True,
                admitted=selected is not None,
                backend_name=f"{self.name}/{self._implementation}" if self._implementation else self.name,
                implementation=self._implementation,
                candidate_results=candidate_results,
                output_correlation=output_corr,
                gradient_correlation=grad_corr,
                relative_l2_error=rel_l2,
                max_absolute_error=max_abs,
                gradient_relative_l2_error=grad_rel_l2,
                output_finite=output_finite,
                gradient_finite=gradient_finite,
                baseline_forward_us=baseline_forward,
                optimized_forward_us=optimized_forward,
                baseline_backward_us=baseline_backward,
                optimized_backward_us=optimized_backward,
                baseline_forward_backward_us=baseline_total,
                optimized_forward_backward_us=optimized_total,
                measured_speedup=speedup,
                reason=(
                    "Admitted by real LSTM forward/backward fidelity and timing screen."
                    if selected is not None
                    else "Rejected by real LSTM forward/backward fidelity or timing screen."
                ),
            )
        except Exception as exc:
            self.report = RebuiltKernelReport(
                available=False,
                admitted=False,
                backend_name=self.name,
                implementation=None,
                candidate_results=None,
                output_correlation=None,
                gradient_correlation=None,
                relative_l2_error=None,
                max_absolute_error=None,
                gradient_relative_l2_error=None,
                output_finite=None,
                gradient_finite=None,
                baseline_forward_us=None,
                optimized_forward_us=None,
                baseline_backward_us=None,
                optimized_backward_us=None,
                baseline_forward_backward_us=None,
                optimized_forward_backward_us=None,
                measured_speedup=None,
                reason=f"Backend unavailable or invalid: {type(exc).__name__}: {exc}",
            )
        return self.report

    @property
    def optimized(self) -> bool:
        return self.report.admitted

    def run(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias: torch.Tensor,
        bias_hh: torch.Tensor,
        *,
        activation_weight: float,
        validate_outputs: bool = False,
        segments: SynchronizedSegmentTimer | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.shape_trace is None:
            self.shape_trace = trace_from_tensors(inputs, hidden, cell, weight_ih, weight_hh)
        self.shape_trace.record_forward()
        if activation_weight <= 0.0 or not self.optimized:
            operation = lambda: public_manual_lstm_cell(inputs, hidden, cell, weight_ih, weight_hh, bias)
            return segments.measure("standard_lstm_forward", operation) if segments else operation()

        def dispatch() -> None:
            self.dispatch_count += 1
            # All tensors are already owned by the controller and share its
            # device/dtype/layout.  Do not materialize, reshape, copy, or pack
            # them in this hot path; calibration validated this exact layout.
            if segments:
                segments.add_zero("tensor_preparation_reshape_copy")

        if segments:
            segments.record_cpu("dispatcher", dispatch)
        else:
            dispatch()
        try:
            if self._implementation == "native-lstm-cell":
                operation = lambda: rebuilt_native_lstm_cell(
                    inputs, hidden, cell, weight_ih, weight_hh, bias, bias_hh
                )
            elif self._implementation == "addmm-four-gate":
                operation = lambda: rebuilt_addmm_lstm_cell(inputs, hidden, cell, weight_ih, weight_hh, bias)
            else:
                raise RuntimeError("no admitted rebuilt implementation")
            optimized = segments.measure("native_lstm_cell_forward", operation) if segments else operation()
            # Checking every recurrent output creates two extra GPU reductions
            # and host synchronizations for every PPO replay evaluation.  The
            # normal path instead checks the completed meta update once before
            # the next adaptation cycle.  Direct probes and action execution
            # can request this immediate guard explicitly.
            if validate_outputs:
                def finite_check() -> bool:
                    return bool(torch.isfinite(optimized[0]).all().item() and torch.isfinite(optimized[1]).all().item())
                finite = segments.measure("runtime_finite_value_check", finite_check) if segments else finite_check()
                if not finite:
                    self.nan_inf_count += 1
                    raise FloatingPointError("non-finite native LSTM output")
            if activation_weight < 1.0:
                # Only transition blending evaluates both routes.  At full
                # activation, evaluating the standard route first would erase
                # the execution-layer speedup that this backend was admitted
                # to provide.
                standard_operation = lambda: public_manual_lstm_cell(inputs, hidden, cell, weight_ih, weight_hh, bias)
                standard = segments.measure("transition_standard_lstm_forward", standard_operation) if segments else standard_operation()
                return (
                    torch.lerp(standard[0], optimized[0], activation_weight),
                    torch.lerp(standard[1], optimized[1], activation_weight),
                )
            return optimized
        except Exception:
            def fallback() -> tuple[torch.Tensor, torch.Tensor]:
                self.fallback_count += 1
                return public_manual_lstm_cell(inputs, hidden, cell, weight_ih, weight_hh, bias)
            return segments.measure("fallback_bookkeeping_and_standard_forward", fallback) if segments else fallback()

    def note_meta_update_health(self, gradient_norm: float) -> bool:
        """Record the deferred finite-value guard once per completed update.

        ``clip_grad_norm_`` already synchronizes when its scalar norm is read
        by the controller.  Reusing that observed value avoids adding a GPU
        reduction to every LSTM-cell replay call while retaining a next-cycle
        fallback when numerical health is lost.
        """
        if not math.isfinite(gradient_norm):
            self.nan_inf_count += 1
            return False
        return True

    def record_backward(self) -> None:
        if self.shape_trace is not None:
            self.shape_trace.record_backward()

    def to_dict(self) -> dict:
        payload = self.report.to_dict()
        payload["observed_training_shape"] = self.shape_trace.to_dict() if self.shape_trace else None
        return payload
