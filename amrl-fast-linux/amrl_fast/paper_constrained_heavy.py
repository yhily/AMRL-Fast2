"""Paper-constrained heavy AMRL reconstruction.

This module is deliberately separate from the ordinary AMRL/AMRL-Fast runner.
The public repositories do not contain a runnable outer-PPO training loop, so
this is a documented reconstruction protocol, not a claim to recover the
authors' unavailable implementation.  It contains no artificial delay or
dummy arithmetic: all additional work is on-policy inner-PPO branch probing or
a differentiable outer PPO update.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from torch.distributions import Normal

from .agent import OptimizedLSTMCell, PPOBaseAgent, set_global_seed
from .envs import make_env
from .rebuilt_lstm import RebuiltLSTMBackend
from .statistics import discrete_mutual_information, information_diagnostics
from .telemetry import synchronized_time


PROTOCOL_NAME = "paper-constrained-heavy-amrl-r1"
PROTOCOL = {
    "main_training_budget_env_steps": 100_000,
    "adaptation_interval_env_steps": 10_000,
    "adaptation_events": 10,
    "controller": {"input_size": 5, "hidden_size": 256, "output_size": 3, "action_std": 0.10},
    "outer_ppo": {
        "branch_actions_per_checkpoint": 8,
        "probe_training_steps_per_action": 1_000,
        "clip": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "value_coefficient": 0.5,
        "entropy_coefficient": 1e-3,
        "learning_rate": 1e-3,
        "gradient_clip": 1.0,
        "epochs": 10,
        "batch_size": 8,
    },
    "evaluation": {"episodes_per_checkpoint": 20, "episodes_per_probe_endpoint": 20},
    "action_mapping": {
        "source": "reconstructed from public AMRL-Fast apply_hyperparameters semantics",
        "mean_action": "tanh(controller_head) * [0.30, 0.02, 0.05]",
        "learning_rate": "lr * exp(a0), clipped to [3e-5, 3e-3]",
        "gamma": "clip(gamma + a1, 0.90, 0.999)",
        "ppo_clip_range": "clip(epsilon * exp(a2), 0.05, 0.30)",
    },
    "reward": "branch deterministic 20-episode return after the 1k-step probe minus the main checkpoint deterministic 20-episode return before probing",
    "timing": "all branch train/evaluation, outer update, checkpoint evaluation and logging are included; no sleep, dummy compute or timing padding",
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _now(device: torch.device) -> float:
    _sync(device)
    return time.perf_counter()


def _parameter_norm(module: nn.Module) -> float:
    with torch.no_grad():
        flattened = [parameter.detach().reshape(-1) for parameter in module.parameters()]
        return float(torch.linalg.vector_norm(torch.cat(flattened)).item()) if flattened else 0.0


def _rng_snapshot() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(snapshot: dict[str, Any]) -> None:
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch"])
    if snapshot["cuda"] is not None:
        torch.cuda.set_rng_state_all(snapshot["cuda"])


def _snapshot_digest(payload: dict[str, Any]) -> str:
    """Stable audit digest for a paired initialisation payload."""
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _paired_initialisation_payload(
    main: PPOBaseAgent,
    controller: nn.Module,
    *,
    seed: int,
    env: Monitor,
) -> dict[str, Any]:
    """Save all mutable learner/controller and RNG state before training."""
    return {
        "seed": int(seed),
        "main_policy_state": copy.deepcopy(main.model.policy.state_dict()),
        "main_optimizer_state": copy.deepcopy(main.model.policy.optimizer.state_dict()),
        "main_hyperparameters": copy.deepcopy(main.hyperparameters),
        "controller_state": copy.deepcopy(controller.state_dict()),
        "controller_optimizer_state": copy.deepcopy(controller.optimizer.state_dict()),
        "rng": _rng_snapshot(),
        "environment": asdict(_capture_cartpole_environment(env)),
        "environment_seed": int(seed),
        "environment_state_note": "uninitialized before the first SB3 rollout; seed/state are recorded before training",
    }


def _restore_paired_initialisation(
    payload: dict[str, Any],
    main: PPOBaseAgent,
    controller: nn.Module,
) -> None:
    main.model.policy.load_state_dict(payload["main_policy_state"])
    main.model.policy.optimizer.load_state_dict(copy.deepcopy(payload["main_optimizer_state"]))
    _apply_main_action(main, copy.deepcopy(payload["main_hyperparameters"]))
    controller.load_state_dict(payload["controller_state"])
    controller.optimizer.load_state_dict(copy.deepcopy(payload["controller_optimizer_state"]))
    _restore_rng(payload["rng"])


@dataclass
class CartPoleEnvironmentSnapshot:
    state: list[float] | None
    steps_beyond_terminated: int | None
    elapsed_steps: int | None
    np_random_state: dict | None
    monitor_needs_reset: bool


def _time_limit(env) -> Any | None:
    current = env
    while current is not None:
        if hasattr(current, "_elapsed_steps"):
            return current
        current = getattr(current, "env", None)
    return None


def _capture_cartpole_environment(env: Monitor) -> CartPoleEnvironmentSnapshot:
    raw = env.unwrapped
    if raw.__class__.__name__ != "CartPoleEnv":
        raise TypeError("The paper-constrained r1 runner is intentionally CartPole-only.")
    state = None if raw.state is None else np.asarray(raw.state, dtype=np.float64).copy().tolist()
    generator = getattr(raw, "np_random", None)
    bit_state = copy.deepcopy(generator.bit_generator.state) if generator is not None else None
    limit = _time_limit(env)
    return CartPoleEnvironmentSnapshot(
        state=state,
        steps_beyond_terminated=getattr(raw, "steps_beyond_terminated", None),
        elapsed_steps=getattr(limit, "_elapsed_steps", None),
        np_random_state=bit_state,
        monitor_needs_reset=bool(getattr(env, "needs_reset", False)),
    )


def _restore_cartpole_environment(env: Monitor, snapshot: CartPoleEnvironmentSnapshot) -> None:
    # Establish Monitor's bookkeeping first, then restore the exact live state
    # consumed by SB3's already-populated ``_last_obs``.
    env.reset()
    raw = env.unwrapped
    raw.state = None if snapshot.state is None else np.asarray(snapshot.state, dtype=np.float64).copy()
    raw.steps_beyond_terminated = snapshot.steps_beyond_terminated
    if snapshot.np_random_state is not None:
        raw.np_random.bit_generator.state = copy.deepcopy(snapshot.np_random_state)
    limit = _time_limit(env)
    if limit is not None and snapshot.elapsed_steps is not None:
        limit._elapsed_steps = snapshot.elapsed_steps
    env.needs_reset = snapshot.monitor_needs_reset


@dataclass
class InnerCheckpoint:
    policy_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    hyperparameters: dict[str, float]
    last_observation: np.ndarray
    last_episode_starts: np.ndarray
    num_timesteps: int
    environment: CartPoleEnvironmentSnapshot
    rng: dict[str, Any]
    normalization_state: None = None  # CartPole has no VecNormalize in this protocol.


def _snapshot_inner(agent: PPOBaseAgent) -> InnerCheckpoint:
    model = agent.model
    if model._last_obs is None or model._last_episode_starts is None:
        raise RuntimeError("Cannot branch before SB3 has collected a live main rollout state.")
    return InnerCheckpoint(
        policy_state=copy.deepcopy(model.policy.state_dict()),
        optimizer_state=copy.deepcopy(model.policy.optimizer.state_dict()),
        hyperparameters=copy.deepcopy(agent.hyperparameters),
        last_observation=np.asarray(model._last_obs).copy(),
        last_episode_starts=np.asarray(model._last_episode_starts).copy(),
        num_timesteps=int(model.num_timesteps),
        environment=_capture_cartpole_environment(agent.env),
        rng=_rng_snapshot(),
    )


def _set_clip_range(model: PPO, value: float) -> None:
    model.clip_range = lambda _progress_remaining, fixed=float(value): fixed


def _load_branch_checkpoint(branch: PPOBaseAgent, checkpoint: InnerCheckpoint) -> None:
    branch.model.policy.load_state_dict(checkpoint.policy_state)
    branch.model.policy.optimizer.load_state_dict(copy.deepcopy(checkpoint.optimizer_state))
    # The public wrapper also carries an entropy-coefficient field, whereas
    # this fixed reconstruction controls PPO's third action as clip range.
    # Apply the stored three-action mapping directly rather than silently
    # inventing an entropy value for the branch.
    branch.hyperparameters = copy.deepcopy(checkpoint.hyperparameters)
    _apply_main_action(branch, checkpoint.hyperparameters)
    branch.model.num_timesteps = checkpoint.num_timesteps
    branch.model._last_obs = checkpoint.last_observation.copy()
    branch.model._last_episode_starts = checkpoint.last_episode_starts.copy()
    _restore_cartpole_environment(branch.env, checkpoint.environment)


def _apply_action_to_values(values: dict[str, float], action: torch.Tensor) -> dict[str, float]:
    raw = action.detach().cpu().reshape(-1).numpy()
    return {
        "learning_rate": float(np.clip(values["learning_rate"] * np.exp(raw[0]), 3e-5, 3e-3)),
        "gamma": float(np.clip(values["gamma"] + raw[1], 0.90, 0.999)),
        "ppo_clip_range": float(np.clip(values["ppo_clip_range"] * np.exp(raw[2]), 0.05, 0.30)),
    }


def _apply_main_action(agent: PPOBaseAgent, values: dict[str, float]) -> None:
    agent.hyperparameters.update(values)
    agent.model.lr_schedule = lambda _progress_remaining, fixed=values["learning_rate"]: fixed
    for group in agent.model.policy.optimizer.param_groups:
        group["lr"] = values["learning_rate"]
    agent.model.gamma = values["gamma"]
    _set_clip_range(agent.model, values["ppo_clip_range"])


def _features(stats: dict[str, float], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[stats["mean_reward"], stats["reward_std"], stats["gradient_norm"], stats["policy_entropy"], stats["value_loss"]]],
        dtype=torch.float32,
        device=device,
    )


class HeavyOuterController(nn.Module):
    """Five-input, 256-unit LSTM Gaussian policy and value function."""

    action_scales = torch.tensor([0.30, 0.02, 0.05])

    def __init__(self, device: torch.device, *, use_fast: bool = False) -> None:
        super().__init__()
        self.device = device
        self.feature_norm = nn.LayerNorm(5)
        self.backend = RebuiltLSTMBackend()
        self.cell = OptimizedLSTMCell(256, self.backend, use_fast=use_fast)
        self.trunk = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.action_mean = nn.Linear(128, 3)
        self.value_head = nn.Linear(128, 1)
        self.register_buffer("action_std", torch.full((3,), 0.10))
        self.register_buffer("scales", self.action_scales.clone())
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.to(device)

    def initial_state(self, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch, 256, device=self.device),
            torch.zeros(batch, 256, device=self.device),
        )

    def evaluate(
        self,
        state: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[Normal, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        normalized = self.feature_norm(state)
        # The two experiment arms execute identical controller equations.  The
        # Fast arm changes only this admitted cell dispatch boundary.
        h_new, c_new = self.cell(
            normalized,
            hidden,
            1.0 if self.cell.use_fast else 0.0,
            validate_outputs=False,
        )
        latent = self.trunk(h_new)
        mean = torch.tanh(self.action_mean(latent)) * self.scales
        return Normal(mean, self.action_std.expand_as(mean)), self.value_head(latent).squeeze(-1), (h_new, c_new)


@dataclass
class BranchTransition:
    state: torch.Tensor
    hidden_h: torch.Tensor
    hidden_c: torch.Tensor
    action: torch.Tensor
    old_log_probability: torch.Tensor
    old_value: torch.Tensor
    reward: float
    next_state: torch.Tensor
    next_hidden_h: torch.Tensor
    next_hidden_c: torch.Tensor
    baseline_return: float
    endpoint_return: float
    applied_hyperparameters: dict[str, float]


@dataclass
class HeavyTiming:
    main_inner_base_training: float = 0.0
    branch_probe_rollout: float = 0.0
    branch_probe_training: float = 0.0
    branch_probe_endpoint_evaluation: float = 0.0
    meta_controller_forward: float = 0.0
    outer_ppo_backward_update: float = 0.0
    outer_ppo_forward_loss: float = 0.0
    outer_ppo_backward: float = 0.0
    outer_ppo_optimizer_step: float = 0.0
    hyperparameter_application: float = 0.0
    checkpoint_evaluation: float = 0.0
    logging_monitoring: float = 0.0
    offline_calibration: float = 0.0
    total_wall_clock: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass
class HeavyCounters:
    main_env_steps: int = 0
    auxiliary_probe_training_steps: int = 0
    main_checkpoint_evaluation_episodes: int = 0
    branch_endpoint_evaluation_episodes: int = 0
    adaptation_count: int = 0
    branch_transition_count: int = 0
    outer_ppo_update_count: int = 0
    outer_ppo_epoch_count: int = 0
    outer_optimizer_step_count: int = 0
    meta_forward_call_count: int = 0
    main_hyperparameter_application_count: int = 0
    main_checkpoint_evaluation_env_steps: int = 0
    branch_endpoint_evaluation_env_steps: int = 0


def _evaluate_model(model: PPO, env, episodes: int, seed_base: int) -> tuple[float, float, list[float]]:
    returns: list[float] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed_base + episode)
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = bool(terminated or truncated)
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns)), returns


def _make_branch_agent(
    device: torch.device,
    seed: int,
    *,
    enable_rollout_backend: bool = False,
) -> PPOBaseAgent:
    env, resolved = make_env("CartPole-v1", seed=seed, allow_version_fallback=False)
    if resolved != "CartPole-v1":
        env.close()
        raise RuntimeError(f"Expected CartPole-v1; resolved {resolved}.")
    return PPOBaseAgent(
        env,
        device=device,
        seed=seed,
        rollout_steps=1_000,
        enable_rollout_backend=enable_rollout_backend,
    )


def _outer_update(
    controller: HeavyOuterController,
    transitions: list[BranchTransition],
    device: torch.device,
    counters: HeavyCounters,
    timing: HeavyTiming | None = None,
) -> float:
    if len(transitions) != 8:
        raise RuntimeError(f"Protocol requires eight on-policy branches, received {len(transitions)}.")
    states = torch.cat([item.state for item in transitions], dim=0)
    hidden_h = torch.cat([item.hidden_h for item in transitions], dim=0)
    hidden_c = torch.cat([item.hidden_c for item in transitions], dim=0)
    actions = torch.cat([item.action for item in transitions], dim=0)
    old_log_probabilities = torch.cat([item.old_log_probability.reshape(1) for item in transitions], dim=0)
    old_values = torch.cat([item.old_value.reshape(1) for item in transitions], dim=0)
    next_states = torch.cat([item.next_state for item in transitions], dim=0)
    next_h = torch.cat([item.next_hidden_h for item in transitions], dim=0)
    next_c = torch.cat([item.next_hidden_c for item in transitions], dim=0)
    rewards = torch.tensor([item.reward for item in transitions], dtype=torch.float32, device=device)
    with torch.no_grad():
        _, next_values, _ = controller.evaluate(next_states, (next_h, next_c))
    # Each branch is an independent one-step on-policy rollout.  GAE is
    # implemented with the frozen lambda=0.95; the recurrent term is empty
    # for a one-step horizon, while the correct next-state value bootstrap is
    # retained.
    deltas = rewards + 0.99 * next_values - old_values
    advantages = deltas.clone()  # one-step GAE(lambda=0.95)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    returns = advantages.detach() * 0.0 + (rewards + 0.99 * next_values)
    gradient_norm = 0.0
    for _ in range(10):
        forward_start = _now(device) if timing is not None else 0.0
        distribution, values, _ = controller.evaluate(states, (hidden_h, hidden_c))
        log_probabilities = distribution.log_prob(actions).sum(-1)
        ratios = torch.exp(log_probabilities - old_log_probabilities)
        unclipped = ratios * advantages
        clipped = torch.clamp(ratios, 0.8, 1.2) * advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        value_loss = torch.nn.functional.mse_loss(values, returns)
        entropy = distribution.entropy().sum(-1).mean()
        loss = policy_loss + 0.5 * value_loss - 1e-3 * entropy
        if timing is not None:
            timing.outer_ppo_forward_loss += _now(device) - forward_start
            backward_start = _now(device)
        controller.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        controller.backend.record_backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0).item())
        if timing is not None:
            timing.outer_ppo_backward += _now(device) - backward_start
            optimizer_start = _now(device)
        controller.optimizer.step()
        if timing is not None:
            timing.outer_ppo_optimizer_step += _now(device) - optimizer_start
        counters.outer_ppo_epoch_count += 1
        counters.outer_optimizer_step_count += 1
    counters.outer_ppo_update_count += 1
    return gradient_norm


def _t90(checkpoints: list[dict[str, Any]]) -> float | None:
    if not checkpoints:
        return None
    final_return = float(checkpoints[-1]["mean_return"])
    if final_return <= 0:
        return None
    threshold = 0.90 * final_return
    for record in checkpoints:
        if float(record["mean_return"]) >= threshold:
            return float(record["main_env_step"])
    return None


def run_cartpole_seed0(
    output_directory: str | Path,
    *,
    seed: int = 0,
    device_name: str = "auto",
    use_fast_execution: bool = False,
    paired_initialisation_path: str | Path | None = None,
    paired_initialisation_mode: str = "none",
) -> dict[str, Any]:
    """Run the frozen baseline or its execution-only AMRL-Fast counterpart.

    ``use_fast_execution`` does not alter training, probing, outer PPO, or
    evaluation.  It admits the already audited CUDA-graph rollout backend and
    native LSTM cell before timing starts; their one-off cost is reported
    separately as offline calibration.
    """
    if seed != 0:
        raise ValueError("The r1 validation protocol is frozen to CartPole seed=0.")
    if paired_initialisation_mode not in {"none", "write", "load"}:
        raise ValueError("paired_initialisation_mode must be one of: none, write, load.")
    if paired_initialisation_mode != "none" and paired_initialisation_path is None:
        raise ValueError("A paired_initialisation_path is required when paired initialization is requested.")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)
    if device.type == "cpu" and device_name == "auto":
        device = torch.device("cpu")
    set_global_seed(seed)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    timing, counters = HeavyTiming(), HeavyCounters()
    started = _now(device)
    env, resolved = make_env("CartPole-v1", seed=seed, allow_version_fallback=False)
    eval_env, eval_resolved = make_env("CartPole-v1", seed=1_000_000 + seed, allow_version_fallback=False)
    if resolved != "CartPole-v1" or eval_resolved != "CartPole-v1":
        env.close(); eval_env.close()
        raise RuntimeError(f"Expected CartPole-v1, got {resolved}/{eval_resolved}.")
    # The H=1000 condition is implemented by the project environment factory.
    if getattr(_time_limit(env), "_max_episode_steps", None) != 1_000:
        env.close(); eval_env.close()
        raise RuntimeError("CartPole H=1000 was not applied.")
    backend = RebuiltLSTMBackend()
    main = PPOBaseAgent(
        env,
        device=device,
        seed=seed,
        rollout_steps=10_000,
        enable_rollout_backend=use_fast_execution,
    )
    main.hyperparameters = {"learning_rate": 3e-4, "gamma": 0.99, "ppo_clip_range": 0.10}
    _apply_main_action(main, main.hyperparameters)
    # A dedicated calibration worker proves admission for the fixed branch
    # shape without changing any branch trajectory.  Actual Fast branches
    # are still constructed independently with the identical per-branch seed
    # used by the baseline; reusing one worker would alter constructor RNG
    # consumption and would therefore not be a paired algorithmic execution.
    calibration_rng = _rng_snapshot() if use_fast_execution else None
    calibration_started = _now(device) if use_fast_execution else None
    branch_calibration_worker = (
        _make_branch_agent(device, seed=50_000_000, enable_rollout_backend=True)
        if use_fast_execution else None
    )
    calibration: dict[str, Any] = {"meta_kernel": None, "main_rollout": None, "branch_rollout": None}
    if use_fast_execution:
        try:
            calibration["meta_kernel"] = backend.calibrate(device).to_dict()
            calibration["main_rollout"] = main.admit_rollout_backend()
            calibration["branch_rollout"] = (
                branch_calibration_worker.admit_rollout_backend() if branch_calibration_worker else None
            )
        finally:
            _restore_rng(calibration_rng)
        timing.offline_calibration = _now(device) - calibration_started
        if branch_calibration_worker is not None:
            branch_calibration_worker.env.close()
    controller = HeavyOuterController(device, use_fast=use_fast_execution and backend.optimized)
    paired_initialisation: dict[str, Any] | None = None
    if paired_initialisation_mode == "write":
        path = Path(paired_initialisation_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        paired_initialisation = _paired_initialisation_payload(main, controller, seed=seed, env=main.env)
        torch.save(paired_initialisation, path)
    elif paired_initialisation_mode == "load":
        path = Path(paired_initialisation_path)
        if not path.exists():
            raise FileNotFoundError(f"Paired initialisation snapshot does not exist: {path}")
        paired_initialisation = torch.load(path, map_location=device)
        _restore_paired_initialisation(paired_initialisation, main, controller)
    if paired_initialisation is not None:
        paired_initialisation = {
            "mode": paired_initialisation_mode,
            "path": str(Path(paired_initialisation_path).resolve()),
            "sha256": _snapshot_digest(
                torch.load(paired_initialisation_path, map_location="cpu")
                if paired_initialisation_mode == "load"
                else paired_initialisation
            ),
            "environment_seed": int(seed),
            "environment_state_note": "uninitialized before first rollout; shared seed and RNG snapshot restored",
        }
    controller_norm_before = _parameter_norm(controller)
    hidden = controller.initial_state()
    checkpoints: list[dict[str, Any]] = []
    raw_transitions: list[dict[str, Any]] = []
    branch_execution_totals = {
        "actual_environment_steps": 0,
        "rollout_count": 0,
        "ppo_train_call_count": 0,
        "optimizer_step_count": 0,
        "rollout_collection_time": 0.0,
        "ppo_update_time": 0.0,
    }
    main_actions: list[list[float]] = []
    main_reward_deltas: list[float] = []
    previous_main_return: float | None = None
    try:
        for checkpoint_index in range(10):
            main_start = _now(device)
            stats = main.train_step(10_000)
            timing.main_inner_base_training += _now(device) - main_start
            counters.main_env_steps += 10_000

            checkpoint_start = _now(device)
            baseline_mean, baseline_std, baseline_episodes = _evaluate_model(
                main.model, eval_env, 20, seed_base=10_000_000 + checkpoint_index * 10_000
            )
            timing.checkpoint_evaluation += _now(device) - checkpoint_start
            counters.main_checkpoint_evaluation_episodes += 20
            counters.main_checkpoint_evaluation_env_steps += int(sum(baseline_episodes))
            if previous_main_return is not None and main_actions:
                main_reward_deltas.append(float(baseline_mean - previous_main_return))
            previous_main_return = baseline_mean

            snapshot = _snapshot_inner(main)
            state = _features(stats, device)
            h_before = (hidden[0].detach(), hidden[1].detach())
            forward_start = _now(device)
            with torch.no_grad():
                distribution, value, h_after = controller.evaluate(state, h_before)
                sampled_actions = distribution.sample((8,)).squeeze(1)
                sampled_log_probabilities = distribution.log_prob(sampled_actions).sum(-1)
            timing.meta_controller_forward += _now(device) - forward_start
            counters.meta_forward_call_count += 1

            transitions: list[BranchTransition] = []
            for branch_index in range(8):
                branch_start = _now(device)
                # Each branch starts from the same serialised learner, runtime
                # state and RNG snapshot.  The branch state is never written
                # back to the main learner.
                caller_rng = _rng_snapshot()
                _restore_rng(snapshot.rng)
                branch = _make_branch_agent(
                    device,
                    seed=50_000_000 + checkpoint_index * 100 + branch_index,
                    enable_rollout_backend=use_fast_execution,
                )
                if getattr(_time_limit(branch.env), "_max_episode_steps", None) != 1_000:
                    branch.env.close()
                    raise RuntimeError("CartPole H=1000 was not applied to a branch probe.")
                _load_branch_checkpoint(branch, snapshot)
                if use_fast_execution:
                    # The one-time calibration worker admitted this exact
                    # B=1 CartPole policy shape.  Capture only binds that
                    # already-admitted graph to this branch's independently
                    # constructed parameter allocation; it consumes no RNG.
                    candidate = branch._rollout_backend_candidate
                    if candidate is None:
                        raise RuntimeError("Fast branch is missing its admitted rollout candidate.")
                    candidate._capture()
                    candidate.install()
                    branch.rollout_backend = candidate
                values = _apply_action_to_values(snapshot.hyperparameters, sampled_actions[branch_index])
                _apply_main_action(branch, values)
                branch_train_start = _now(device)
                branch_stats = branch.train_step(1_000)
                timing.branch_probe_training += _now(device) - branch_train_start
                if int(branch.model.num_timesteps) - int(snapshot.num_timesteps) != 1_000:
                    branch.env.close()
                    raise RuntimeError("Branch PPO did not collect exactly the frozen 1,000 probe-training steps.")
                counters.auxiliary_probe_training_steps += 1_000
                branch_eval_env, branch_eval_resolved = make_env(
                    "CartPole-v1", seed=20_000_000 + checkpoint_index * 1_000 + branch_index, allow_version_fallback=False
                )
                if branch_eval_resolved != "CartPole-v1":
                    branch.env.close(); branch_eval_env.close(); _restore_rng(caller_rng)
                    raise RuntimeError(f"Branch evaluator resolved {branch_eval_resolved}.")
                endpoint_start = _now(device)
                endpoint_mean, endpoint_std, endpoint_episodes = _evaluate_model(
                    branch.model, branch_eval_env, 20, seed_base=30_000_000 + checkpoint_index * 10_000
                )
                timing.branch_probe_endpoint_evaluation += _now(device) - endpoint_start
                counters.branch_endpoint_evaluation_episodes += 20
                counters.branch_endpoint_evaluation_env_steps += int(sum(endpoint_episodes))
                branch_eval_env.close()
                next_state = _features(branch_stats, device)
                transitions.append(BranchTransition(
                    state=state.detach(), hidden_h=h_before[0].detach(), hidden_c=h_before[1].detach(),
                    action=sampled_actions[branch_index:branch_index + 1].detach(),
                    old_log_probability=sampled_log_probabilities[branch_index:branch_index + 1].detach(),
                    old_value=value.detach(), reward=float(endpoint_mean - baseline_mean), next_state=next_state.detach(),
                    next_hidden_h=h_after[0].detach(), next_hidden_c=h_after[1].detach(),
                    baseline_return=float(baseline_mean), endpoint_return=float(endpoint_mean), applied_hyperparameters=values,
                ))
                raw_transitions.append({
                    "checkpoint": checkpoint_index + 1, "branch": branch_index + 1,
                    "state": state.detach().cpu().reshape(-1).tolist(),
                    "action": sampled_actions[branch_index].detach().cpu().tolist(),
                    "old_log_probability": float(sampled_log_probabilities[branch_index].item()),
                    "old_value": float(value.item()), "probe_reward": float(endpoint_mean - baseline_mean),
                    "baseline_return": float(baseline_mean), "endpoint_return": float(endpoint_mean),
                    "next_state": next_state.detach().cpu().reshape(-1).tolist(),
                    "hyperparameters": values, "endpoint_episode_returns": endpoint_episodes,
                })
                audit = branch.execution_audit()
                # ``model.num_timesteps`` is an absolute counter restored
                # from the main checkpoint; it must never be summed across
                # branches.  The actual branch training contribution is the
                # frozen 1,000 steps recorded in ``counters`` below.
                for key in ("rollout_count", "ppo_train_call_count", "optimizer_step_count"):
                    branch_execution_totals[key] += int(audit[key])
                for key in ("rollout_collection_time", "ppo_update_time"):
                    branch_execution_totals[key] += float(audit[key])
                branch.env.close()
                _restore_rng(caller_rng)
                timing.branch_probe_rollout += _now(device) - branch_start
            counters.branch_transition_count += len(transitions)

            update_start = _now(device)
            gradient_norm = _outer_update(controller, transitions, device, counters, timing)
            timing.outer_ppo_backward_update += _now(device) - update_start
            if use_fast_execution:
                backend.note_meta_update_health(gradient_norm)

            apply_forward_start = _now(device)
            with torch.no_grad():
                updated_distribution, _, hidden_after_update = controller.evaluate(state, h_before)
                main_action = updated_distribution.mean.detach()
            timing.meta_controller_forward += _now(device) - apply_forward_start
            counters.meta_forward_call_count += 1
            apply_start = _now(device)
            main_values = _apply_action_to_values(main.hyperparameters, main_action)
            _apply_main_action(main, main_values)
            timing.hyperparameter_application += _now(device) - apply_start
            counters.main_hyperparameter_application_count += 1
            hidden = (hidden_after_update[0].detach(), hidden_after_update[1].detach())
            main_actions.append(main_action.cpu().reshape(-1).tolist())
            counters.adaptation_count += 1

            log_start = _now(device)
            checkpoints.append({
                "checkpoint": checkpoint_index + 1,
                "main_env_step": (checkpoint_index + 1) * 10_000,
                "mean_return": float(baseline_mean), "std_return": float(baseline_std),
                "episode_returns": baseline_episodes, "main_hyperparameters_after_update": main_values,
            })
            timing.logging_monitoring += _now(device) - log_start
        _sync(device)
        timing.total_wall_clock = time.perf_counter() - started
    finally:
        main.env.close()
        eval_env.close()

    # One-cycle-delayed pairing: the action after checkpoint t acts on the
    # main learner in the next 10k window and is paired with that checkpoint's
    # return change.  The final action intentionally has no following window.
    mi_actions = main_actions[:-1]
    diagnostics = information_diagnostics(
        mi_actions, main_reward_deltas, wall_clock_seconds=timing.total_wall_clock
    ) if mi_actions else {}
    pure_meta = timing.meta_controller_forward + timing.outer_ppo_backward_update + timing.hyperparameter_application
    outer_total = timing.branch_probe_rollout + pure_meta
    main_execution = main.execution_audit()
    branch_execution = branch_execution_totals if use_fast_execution else None
    if branch_execution is not None:
        branch_execution["actual_environment_steps"] = int(counters.auxiliary_probe_training_steps)
    branch_setup_copy = max(
        0.0,
        timing.branch_probe_rollout - timing.branch_probe_training - timing.branch_probe_endpoint_evaluation,
    )
    training_evaluation_wall_clock = timing.total_wall_clock - timing.offline_calibration
    execution_mode = "AMRL-Fast" if use_fast_execution else "AMRL"
    result = {
        "protocol_name": PROTOCOL_NAME,
        "protocol": PROTOCOL,
        "environment": {"requested": "CartPole-v1", "resolved": resolved, "max_episode_steps": 1_000},
        "seed": seed,
        "device": str(device),
        "development_only": True,
        "paper_result_eligible": False,
        "interpretation": (
            "Reconstructed heavy AMRL-Fast execution-only counterpart; not a recovery of unavailable original source."
            if use_fast_execution
            else "Reconstructed heavy AMRL baseline; not a recovery of unavailable original source."
        ),
        "method": execution_mode,
        "counters": asdict(counters),
        "step_accounting": {
            "main_training_env_steps": int(counters.main_env_steps),
            "branch_training_env_steps": int(counters.auxiliary_probe_training_steps),
            "branch_endpoint_evaluation_env_steps": int(counters.branch_endpoint_evaluation_env_steps),
            "main_checkpoint_evaluation_env_steps": int(counters.main_checkpoint_evaluation_env_steps),
            "total_auxiliary_env_steps": int(
                counters.auxiliary_probe_training_steps + counters.branch_endpoint_evaluation_env_steps
            ),
            "total_all_environment_steps": int(
                counters.main_env_steps
                + counters.auxiliary_probe_training_steps
                + counters.branch_endpoint_evaluation_env_steps
                + counters.main_checkpoint_evaluation_env_steps
            ),
            "branch_train_n_steps": 1_000,
            "main_train_n_steps": 10_000,
            "branch_evaluation_included_in_model_num_timesteps": False,
            "note": (
                "Branch model.num_timesteps is restored from each main checkpoint and is an absolute SB3 counter; "
                "it is not aggregated as actual branch environment steps."
            ),
        },
        "paired_initialisation": paired_initialisation,
        "controller_parameter_norm": {"before": controller_norm_before, "after": _parameter_norm(controller), "changed": _parameter_norm(controller) != controller_norm_before},
        "main_execution": main_execution,
        "branch_execution": branch_execution,
        "execution_backends": {
            "meta": calibration["meta_kernel"] if use_fast_execution else "standard_manual_lstm",
            "main_rollout": calibration["main_rollout"] if use_fast_execution else "standard_sb3_policy",
            "branch_rollout": calibration["branch_rollout"] if use_fast_execution else "standard_sb3_policy",
            "branch_worker_cache": False,
            "branch_graph_binding": (
                "per-independent-branch capture after one offline shape/fidelity admission"
                if use_fast_execution else "not_applicable"
            ),
        },
        "checkpoints": checkpoints,
        "main_final_return": checkpoints[-1]["mean_return"],
        "T90_main_env_step": _t90(checkpoints),
        "information": {
            "U": mi_actions, "delta_R": main_reward_deltas,
            "discrete_mi_bits": discrete_mutual_information(mi_actions, main_reward_deltas),
            "diagnostics": diagnostics,
        },
        "timing_seconds": timing.as_dict(),
        "timing_boundaries": {
            "offline_calibration": float(timing.offline_calibration),
            "training_evaluation_wall_clock": float(training_evaluation_wall_clock),
            "lifecycle_wall_clock": float(timing.total_wall_clock),
            "branch_setup_copy_overhead": float(branch_setup_copy),
            "main_rollout_collection": float(main_execution["rollout_collection_time"]),
            "main_ppo_update": float(main_execution["ppo_update_time"]),
            "branch_rollout_collection": (
                float(branch_execution["rollout_collection_time"]) if branch_execution is not None else None
            ),
            "branch_ppo_update": (
                float(branch_execution["ppo_update_time"]) if branch_execution is not None else None
            ),
        },
        "shares": {
            "pure_meta_compute_share": pure_meta / timing.total_wall_clock if timing.total_wall_clock else float("nan"),
            "outer_loop_total_share": outer_total / timing.total_wall_clock if timing.total_wall_clock else float("nan"),
            "branch_probe_share": timing.branch_probe_rollout / timing.total_wall_clock if timing.total_wall_clock else float("nan"),
            "main_inner_share": timing.main_inner_base_training / timing.total_wall_clock if timing.total_wall_clock else float("nan"),
        },
        "raw_outer_transitions": raw_transitions,
        "integrity": {
            "main_steps_complete": counters.main_env_steps == 100_000,
            "auxiliary_probe_steps_complete": counters.auxiliary_probe_training_steps == 80_000,
            "branch_transition_count_complete": counters.branch_transition_count == 80,
            "outer_update_count_complete": counters.outer_ppo_update_count == 10,
            "outer_epoch_count_complete": counters.outer_ppo_epoch_count == 100,
            "outer_optimizer_count_complete": counters.outer_optimizer_step_count == 100,
            "controller_parameters_changed": _parameter_norm(controller) != controller_norm_before,
            "dummy_compute_used": False,
            "sleep_used": False,
        },
    }
    with (output / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump({"protocol_name": PROTOCOL_NAME, "protocol": PROTOCOL}, handle, indent=2)
    filename = "cartpole_seed0_heavy_fast.json" if use_fast_execution else "cartpole_seed0_heavy_baseline.json"
    with (output / filename).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
