"""Timing and stability accounting shared by AMRL and AMRL-Fast."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import torch


def synchronized_time(device: torch.device):
    """Context manager-like timer helper with CUDA synchronization."""
    class _Timer:
        def __enter__(self):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            self.start = time.perf_counter()
            return self

        def __exit__(self, *_):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            self.seconds = time.perf_counter() - self.start
    return _Timer()


@dataclass
class SynchronizedSegmentTimer:
    """Accumulate CUDA-synchronized timings for one diagnostic code path.

    This is intentionally opt-in.  Segment-level synchronization is useful for
    a preflight diagnosis, but must never be part of the normal AMRL-Fast hot
    path because it would itself perturb kernel scheduling.
    """

    device: torch.device
    seconds: dict[str, float] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)

    def measure(self, name: str, operation):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        result = operation()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.seconds[name] = self.seconds.get(name, 0.0) + (time.perf_counter() - started)
        self.calls[name] = self.calls.get(name, 0) + 1
        return result

    def record_cpu(self, name: str, operation):
        """Time dispatch/bookkeeping without adding an artificial CUDA sync."""
        started = time.perf_counter()
        result = operation()
        self.seconds[name] = self.seconds.get(name, 0.0) + (time.perf_counter() - started)
        self.calls[name] = self.calls.get(name, 0) + 1
        return result

    def add_zero(self, name: str) -> None:
        self.seconds.setdefault(name, 0.0)
        self.calls.setdefault(name, 0)

    def microseconds(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "total_us": seconds * 1e6,
                "calls": self.calls.get(name, 0),
                "mean_us": seconds * 1e6 / self.calls[name] if self.calls.get(name, 0) else 0.0,
            }
            for name, seconds in sorted(self.seconds.items())
        }


@dataclass
class ComponentTiming:
    base_agent_training_time: float = 0.0
    meta_learner_forward_time: float = 0.0
    meta_learner_backward_time: float = 0.0
    hyperparameter_update_time: float = 0.0
    evaluation_time: float = 0.0
    logging_time: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StabilityCounters:
    fallback_count: int = 0
    eligible_dispatch_count: int = 0
    nan_inf_count: int = 0

    @property
    def fallback_ratio(self) -> float:
        return self.fallback_count / self.eligible_dispatch_count if self.eligible_dispatch_count else 0.0

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["fallback_ratio"] = self.fallback_ratio
        return payload


@dataclass
class TrialTelemetry:
    timing: ComponentTiming = field(default_factory=ComponentTiming)
    stability: StabilityCounters = field(default_factory=StabilityCounters)
