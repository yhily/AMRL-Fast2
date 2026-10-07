#!/usr/bin/env python3
"""Profile the real SB3 PPO base-agent hot path without changing PPO semantics.

One profile is one genuine 5,000-step PPO rollout plus its standard PPO
training update.  The instrumentation is development-only: it wraps existing
methods, records CPU dispatch and CUDA-event durations, and restores every
method before evaluation.  It never replaces PPO loss, policy modules,
optimizer, reward, or network execution.
"""
from __future__ import annotations

import json
import time
import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amrl_fast.agent import PPOBaseAgent, set_global_seed
from amrl_fast.envs import make_env


ENVIRONMENTS = ("CartPole-v1", "MountainCar-v0", "BSuite Memory")
DEFAULT_ROLLOUT_STEPS = 5_000
SEED = 0


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _shape(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (tuple, list)):
        return [_shape(item) for item in value]
    return type(value).__name__


class Instrumentation:
    """Nested method timers with CPU dispatch and stream-ordered CUDA events."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.cpu_seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.cuda_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        self.shapes: dict[str, dict[str, Any]] = {}
        self.restore: list[tuple[object, str, Any]] = []

    def patch(self, owner: object, attribute: str, name: str, *, cuda: bool = False, generator: bool = False) -> None:
        original = getattr(owner, attribute)
        self.restore.append((owner, attribute, original))

        if generator:
            def wrapped(*args, **kwargs):
                iterator = original(*args, **kwargs)
                def timed_iterator():
                    while True:
                        started = time.perf_counter()
                        try:
                            item = next(iterator)
                        except StopIteration:
                            self.cpu_seconds[name] += time.perf_counter() - started
                            return
                        self.cpu_seconds[name] += time.perf_counter() - started
                        self.calls[name] += 1
                        self.shapes.setdefault(name, {"output": _shape(item)})
                        yield item
                return timed_iterator()
        else:
            def wrapped(*args, **kwargs):
                start_event = end_event = None
                if cuda and self.device.type == "cuda":
                    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start_event.record()
                started = time.perf_counter()
                result = original(*args, **kwargs)
                self.cpu_seconds[name] += time.perf_counter() - started
                self.calls[name] += 1
                self.shapes.setdefault(name, {"input": _shape(args[0]) if args else None, "output": _shape(result)})
                if end_event is not None:
                    end_event.record()
                    self.cuda_events[name].append((start_event, end_event))
                return result
        setattr(owner, attribute, wrapped)

    def patch_tensor_backward(self, name: str) -> None:
        original = torch.Tensor.backward
        self.restore.append((torch.Tensor, "backward", original))

        def wrapped(tensor, *args, **kwargs):
            start_event = end_event = None
            if self.device.type == "cuda":
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start_event.record()
            started = time.perf_counter()
            result = original(tensor, *args, **kwargs)
            self.cpu_seconds[name] += time.perf_counter() - started
            self.calls[name] += 1
            if end_event is not None:
                end_event.record()
                self.cuda_events[name].append((start_event, end_event))
            return result
        torch.Tensor.backward = wrapped

    def finalize(self, total_training_seconds: float) -> list[dict]:
        _sync(self.device)
        rows = []
        for name in sorted(set(self.cpu_seconds) | set(self.cuda_events)):
            cuda_seconds = sum(start.elapsed_time(end) for start, end in self.cuda_events[name]) / 1e3
            # CUDA duration is the execution time for compute wrappers;
            # CPU-only wrappers such as environment interaction retain CPU time.
            absolute = cuda_seconds if cuda_seconds > 0 else self.cpu_seconds[name]
            rows.append({
                "component": name,
                "absolute_time_seconds": absolute,
                "cpu_dispatch_seconds": self.cpu_seconds[name],
                "cuda_execution_seconds": cuda_seconds,
                "percentage_of_profiled_training_wall_clock": absolute / total_training_seconds if total_training_seconds else 0.0,
                "call_count": self.calls[name],
                "input_output_shape": self.shapes.get(name),
                "device": "CUDA" if cuda_seconds > 0 else "CPU",
                "measurement_scope": "Direct wrapped call; nested components are not additive.",
            })
        return rows

    def unpatch(self) -> None:
        for owner, attribute, original in reversed(self.restore):
            setattr(owner, attribute, original)
        self.restore.clear()


def _profiler_operator_summary(profiler) -> dict:
    categories = {
        "policy_value_linear_matmul": {"aten::linear", "aten::addmm", "aten::mm", "aten::matmul", "aten::bmm"},
        "activation": {"aten::relu", "aten::tanh", "aten::leaky_relu", "aten::sigmoid"},
        "ppo_loss_construction": {"aten::exp", "aten::clamp", "aten::minimum", "aten::min", "aten::mean", "aten::mse_loss", "aten::sub", "aten::mul", "aten::add"},
        "tensor_copy_reshape_device_transfer": {"aten::to", "aten::_to_copy", "aten::copy_", "aten::view", "aten::reshape", "aten::flatten", "aten::contiguous"},
    }
    payload = {name: {"cpu_self_seconds": 0.0, "cuda_self_seconds": 0.0, "call_count": 0, "operators": []} for name in categories}
    top = []
    for event in profiler.key_averages():
        name = event.key
        cpu_seconds = float(getattr(event, "self_cpu_time_total", 0.0)) / 1e6
        cuda_us = float(getattr(event, "self_cuda_time_total", getattr(event, "self_device_time_total", 0.0)))
        cuda_seconds = cuda_us / 1e6
        entry = {"operator": name, "cpu_self_seconds": cpu_seconds, "cuda_self_seconds": cuda_seconds, "call_count": int(event.count)}
        top.append(entry)
        for category, operators in categories.items():
            if name in operators:
                payload[category]["cpu_self_seconds"] += cpu_seconds
                payload[category]["cuda_self_seconds"] += cuda_seconds
                payload[category]["call_count"] += int(event.count)
                payload[category]["operators"].append(entry)
    top.sort(key=lambda item: max(item["cpu_self_seconds"], item["cuda_self_seconds"]), reverse=True)
    return {"categories": payload, "top_operators": top[:30]}


def _sample_one_real_ppo_minibatch(model) -> dict:
    """Profile one genuine PPO minibatch after the full timed rollout.

    Full-step operator tracing records hundreds of thousands of tiny CUDA
    events and materially changes wall-clock time.  The main profile therefore
    times the complete untouched 5k-step PPO update with CUDA events, while
    this isolated post-measurement sample identifies the operators and shapes
    of one production batch without contaminating that timing.
    """
    original_get = model.rollout_buffer.get
    original_epochs = model.n_epochs

    def one_batch(batch_size):
        iterator = original_get(batch_size)
        try:
            first = next(iterator)
        except StopIteration:
            return iter(())
        return iter((first,))

    model.rollout_buffer.get = one_batch
    model.n_epochs = 1
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, profile_memory=False, with_stack=False) as profiler:
            model.train()
        _sync(torch.device(model.device))
        return {
            "scope": "One real PPO minibatch, post-measurement diagnostic only; excluded from training wall-clock.",
            **_profiler_operator_summary(profiler),
        }
    finally:
        model.rollout_buffer.get = original_get
        model.n_epochs = original_epochs


def _run(environment: str, *, fast: bool = False, rollout_steps: int = DEFAULT_ROLLOUT_STEPS) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(SEED)
    env, resolved = make_env(environment, seed=SEED, allow_version_fallback=False)
    base = PPOBaseAgent(
        env, device=device, seed=SEED, rollout_steps=rollout_steps,
        enable_rollout_backend=fast,
    )
    backend_audit = base.admit_rollout_backend() if fast else None
    model, policy = base.model, base.model.policy
    instrument = Instrumentation(device)
    # Rollout collection and its CPU-only environment/buffer work.
    instrument.patch(model.env, "step", "environment_interaction")
    instrument.patch(model.env, "reset", "environment_reset")
    instrument.patch(model.rollout_buffer, "add", "rollout_buffer_processing")
    instrument.patch(model.rollout_buffer, "get", "rollout_buffer_minibatch_processing", generator=True)
    instrument.patch(model.rollout_buffer, "compute_returns_and_advantage", "advantage_return_computation")
    instrument.patch(model, "collect_rollouts", "rollout_collection_total", cuda=True)
    instrument.patch(model, "train", "ppo_training_total", cuda=True)
    instrument.patch(model, "_dump_logs", "logging")
    # Actual policy execution boundaries; their nested time is intentionally
    # reported in addition to operator-level statistics below.
    instrument.patch(policy, "forward", "rollout_policy_value_forward", cuda=True)
    instrument.patch(policy, "evaluate_actions", "ppo_policy_value_forward", cuda=True)
    instrument.patch(policy.mlp_extractor, "forward_actor", "policy_mlp_forward", cuda=True)
    instrument.patch(policy.mlp_extractor, "forward_critic", "value_mlp_forward", cuda=True)
    instrument.patch(policy.action_net, "forward", "policy_linear_head", cuda=True)
    instrument.patch(policy.value_net, "forward", "value_linear_head", cuda=True)
    instrument.patch(policy.optimizer, "step", "optimizer_step", cuda=True)
    instrument.patch(policy.optimizer, "zero_grad", "optimizer_zero_grad")
    instrument.patch_tensor_backward("backward")
    _sync(device)
    try:
        # This is the complete, unmodified 5k rollout and full PPO update.
        # CUDA events are recorded without synchronizing every call, so the
        # training wall clock remains representative.
        started = time.perf_counter()
        stats = base.train_step(rollout_steps)
        _sync(device)
        training_seconds = time.perf_counter() - started
        components = instrument.finalize(training_seconds)
    finally:
        instrument.unpatch()
    # Evaluation is outside the base training hot path and is recorded only as
    # a distinct end-to-end component.
    _sync(device)
    started = time.perf_counter()
    rewards = []
    for _ in range(20):
        observation, _ = base.env.reset()
        done, total = False, 0.0
        while not done:
            action, _ = base.model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = base.env.step(action)
            total += float(reward)
            done = bool(terminated or truncated)
        rewards.append(total)
    _sync(device)
    evaluation_seconds = time.perf_counter() - started
    operators = _sample_one_real_ppo_minibatch(model)
    env.close()
    return {
        "development_only": True,
        "execution_stack": "cuda_graph_policy_body" if fast else "standard",
        "base_agent_backend_admission": backend_audit,
        "base_agent_backend_runtime": None if base.rollout_backend is None else {
            "backend": base.rollout_backend.name,
            "dispatch_count": base.rollout_backend.dispatch_count,
            "fallback_count": base.rollout_backend.fallback_count,
            "nan_inf_count": base.rollout_backend.nan_inf_count,
        },
        "paper_result_eligible": False,
        "environment": environment,
        "resolved_environment_id": resolved,
        "seed": SEED,
        "profiled_inner_ppo_rollout_steps": rollout_steps,
        "training_wall_clock_seconds": training_seconds,
        "evaluation_wall_clock_seconds": evaluation_seconds,
        "evaluation_return_mean": float(sum(rewards) / len(rewards)),
        "evaluation_return_std": float(torch.tensor(rewards).float().std(unbiased=False).item()),
        "training_stats": stats,
        "components": components,
        "operator_profile": operators,
        "policy_structure": str(policy),
        "profiling_notes": {
            "component_times_are_nested": "High-level wrappers and their children overlap; percentages identify candidate hotspots and are not summed.",
            "cuda_timing": "GPU compute wrappers use CUDA events and synchronize only after the complete real PPO update.",
            "operator_sampling": "Operator-level data comes from one post-measurement real PPO minibatch to prevent profiler overhead from dominating full-rollout timing.",
            "algorithm_integrity": "The profile runs the unmodified SB3 PPO collect_rollouts/train methods and merely wraps timing probes.",
        },
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this base-agent hotspot profile.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--environment", action="append", choices=ENVIRONMENTS)
    parser.add_argument("--rollout-steps", type=int, default=DEFAULT_ROLLOUT_STEPS)
    parser.add_argument("--label", default="")
    arguments = parser.parse_args()
    if arguments.rollout_steps <= 0:
        raise SystemExit("--rollout-steps must be positive.")
    environments = tuple(arguments.environment) if arguments.environment else ENVIRONMENTS
    results = []
    for environment in environments:
        result = _run(environment, fast=arguments.fast, rollout_steps=arguments.rollout_steps)
        stem = "fast_profile" if arguments.fast else "profile"
        filename = f"{stem}_{arguments.label}.json" if arguments.label else f"{stem}.json"
        target = ROOT / "artifacts" / "base_agent_hotspots" / environment / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        results.append(result)
        print(json.dumps({
            "environment": environment,
            "training_wall_clock_seconds": result["training_wall_clock_seconds"],
            "component_count": len(result["components"]),
        }), flush=True)
    target = ROOT / "artifacts" / "base_agent_hotspots" / "complete.json"
    target.write_text(json.dumps({"results": results}, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
