"""Common AMRL / AMRL-Fast training path.

The public repositories expose the AMRL components but do not connect the
meta-policy to a valid optimisation step.  This module restores that stated
PPO-style outer update: the LSTM meta-policy emits bounded hyperparameter
actions, a subsequent adaptation window supplies its reward, and the stored
state/action/log-probability tuple is optimised with clipped PPO.  AMRL-Fast
changes only the recurrent gate MatMul dispatch inside that same controller.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from .base_agent_backend import CudaGraphPolicyBodyBackend, FusedRolloutMlpBackend
from .rebuilt_lstm import RebuiltLSTMBackend
from .telemetry import SynchronizedSegmentTimer, TrialTelemetry, synchronized_time


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PPOBaseAgent:
    """PPO inner learner with the original AMRL hyperparameter interface."""

    def __init__(
        self,
        env,
        *,
        device: torch.device,
        seed: int,
        rollout_steps: int,
        enable_rollout_backend: bool = False,
        enable_foreach_adam: bool = False,
        enable_fused_adam: bool = False,
    ) -> None:
        self.env = Monitor(env)
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=3e-4,
            gamma=0.99,
            ent_coef=0.01,
            # Match one inner PPO rollout to one paper adaptation window so
            # that a requested 100k-step budget is not silently rounded up by
            # SB3's fixed rollout collector.
            n_steps=rollout_steps,
            batch_size=64,
            clip_range=0.2,
            max_grad_norm=1.0,
            device=str(device),
            seed=seed,
            verbose=0,
        )
        # Audit-only counters.  Wrapping the already constructed SB3 methods
        # does not change rollout, loss, gradient or optimiser semantics.
        self.rollout_count = 0
        self.ppo_train_call_count = 0
        self.optimizer_step_count = 0
        self.rollout_collection_time = 0.0
        self.ppo_update_time = 0.0
        self.policy_parameter_norm_before_training = self._parameter_norm(self.model.policy.mlp_extractor.policy_net)
        self.value_parameter_norm_before_training = self._parameter_norm(self.model.policy.mlp_extractor.value_net)
        original_collect_rollouts = self.model.collect_rollouts
        original_train = self.model.train
        original_optimizer_step = self.model.policy.optimizer.step

        def counted_collect_rollouts(*args, **kwargs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = original_collect_rollouts(*args, **kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            self.rollout_collection_time += time.perf_counter() - started
            self.rollout_count += 1
            return result

        def counted_train(*args, **kwargs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = original_train(*args, **kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            self.ppo_update_time += time.perf_counter() - started
            self.ppo_train_call_count += 1
            return result

        def counted_optimizer_step(*args, **kwargs):
            self.optimizer_step_count += 1
            return original_optimizer_step(*args, **kwargs)

        self.model.collect_rollouts = counted_collect_rollouts
        self.model.train = counted_train
        self.model.policy.optimizer.step = counted_optimizer_step
        self.hyperparameters = {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "entropy_coefficient": 0.01,
        }
        self.rollout_backend: FusedRolloutMlpBackend | CudaGraphPolicyBodyBackend | None = None
        self.rollout_backend_audit: dict | None = None
        # Base-agent substitution has the same conservative admission rule as
        # the meta backend.  No unadmitted path changes policy.forward.
        if enable_rollout_backend:
            self._rollout_backend_candidate: FusedRolloutMlpBackend | CudaGraphPolicyBodyBackend | None = CudaGraphPolicyBodyBackend(
                self.model, device=device
            )
        else:
            self._rollout_backend_candidate = None
        self.foreach_adam_enabled = False
        if enable_fused_adam:
            self.model.policy.optimizer.defaults["foreach"] = None
            self.model.policy.optimizer.defaults["fused"] = True
            for group in self.model.policy.optimizer.param_groups:
                group["foreach"] = None
                group["fused"] = True
            self.foreach_adam_enabled = True
        elif enable_foreach_adam:
            # This changes only Adam's CUDA execution scheduling.  Its update
            # rule, parameter groups, LR schedule and PPO gradients remain
            # identical; admission verifies the actual parameter update.
            self.model.policy.optimizer.defaults["foreach"] = True
            for group in self.model.policy.optimizer.param_groups:
                group["foreach"] = True
            self.foreach_adam_enabled = True

    @staticmethod
    def _parameter_norm(module: nn.Module) -> float:
        with torch.no_grad():
            values = [parameter.detach().reshape(-1) for parameter in module.parameters()]
            return float(torch.linalg.vector_norm(torch.cat(values)).item()) if values else 0.0

    def execution_audit(self) -> dict[str, float | int]:
        """Return persisted SB3 execution counters after a completed seed."""
        return {
            "actual_environment_steps": int(self.model.num_timesteps),
            "rollout_count": int(self.rollout_count),
            "ppo_train_call_count": int(self.ppo_train_call_count),
            "optimizer_step_count": int(self.optimizer_step_count),
            "rollout_collection_time": float(self.rollout_collection_time),
            "ppo_update_time": float(self.ppo_update_time),
            "policy_parameter_norm_before_training": float(self.policy_parameter_norm_before_training),
            "policy_parameter_norm_after_training": self._parameter_norm(self.model.policy.mlp_extractor.policy_net),
            "value_parameter_norm_before_training": float(self.value_parameter_norm_before_training),
            "value_parameter_norm_after_training": self._parameter_norm(self.model.policy.mlp_extractor.value_net),
        }

    def admit_rollout_backend(self) -> dict | None:
        """Calibrate once before training; never put admission work in the hot path."""
        if self._rollout_backend_candidate is None:
            return None
        # Calibration input generation must not alter the seeded stochastic
        # policy trajectory used by the subsequent AMRL/Fast comparison.
        cpu_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            audit = self._rollout_backend_candidate.calibrate()
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        self.rollout_backend_audit = audit.as_dict()
        if audit.admitted:
            self._rollout_backend_candidate.install()
            self.rollout_backend = self._rollout_backend_candidate
        return self.rollout_backend_audit

    def train_step(self, n_steps: int) -> dict[str, float]:
        self.model.learn(total_timesteps=n_steps, reset_num_timesteps=False, progress_bar=False)
        episodes = list(self.model.ep_info_buffer)[-10:]
        rewards = [float(item["r"]) for item in episodes]
        logger = self.model.logger.name_to_value
        squared_gradients = [
            parameter.grad.detach().square().sum()
            for parameter in self.model.policy.parameters()
            if parameter.grad is not None
        ]
        gradient_norm = float(torch.sqrt(torch.stack(squared_gradients).sum()).item()) if squared_gradients else 0.0
        return {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "reward_std": float(np.std(rewards)) if rewards else 1.0,
            "gradient_norm": gradient_norm,
            "policy_entropy": float(-logger.get("train/entropy_loss", 0.0)),
            "policy_loss": float(logger.get("train/policy_gradient_loss", 0.0)),
            "value_loss": float(logger.get("train/value_loss", 0.0)),
        }

    def set_hyperparameters(self, values: dict[str, float]) -> None:
        self.hyperparameters.update(values)
        # SB3 refreshes its optimiser LR from this schedule inside every PPO
        # training call, so updating param groups alone would silently undo an
        # AMRL learning-rate action at the next inner-loop update.
        self.model.lr_schedule = lambda _progress_remaining: values["learning_rate"]
        for group in self.model.policy.optimizer.param_groups:
            group["lr"] = values["learning_rate"]
        self.model.gamma = values["gamma"]
        self.model.ent_coef = values["entropy_coefficient"]

    def predict(self, observation, deterministic: bool = True):
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action


class OptimizedLSTMCell(nn.Module):
    """Public AMRL LSTM with an execution-layer fused backend boundary."""

    def __init__(self, hidden_size: int, backend: RebuiltLSTMBackend, use_fast: bool) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.backend = backend
        self.use_fast = use_fast
        # These are the public AMRL-Fast combined four-gate tensors, not four
        # independently parameterized substitute cells.
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, 5))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))
        self.register_buffer("bias_hh", torch.zeros(4 * hidden_size))
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)

    def forward(
        self,
        features: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
        activation_weight: float,
        *,
        validate_outputs: bool = False,
        segments: SynchronizedSegmentTimer | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = hidden
        return self.backend.run(
            features,
            h_prev,
            c_prev,
            self.weight_ih,
            self.weight_hh,
            self.bias,
            self.bias_hh,
            activation_weight=activation_weight if self.use_fast else 0.0,
            validate_outputs=validate_outputs,
            segments=segments,
        )


@dataclass
class MetaTransition:
    features: torch.Tensor
    h_before: torch.Tensor
    c_before: torch.Tensor
    action: torch.Tensor
    # The bounded controller mean, distinct from the stochastic PPO action.
    # This is the manuscript U_t logging quantity and never alters training.
    controller_output: torch.Tensor
    old_log_probability: torch.Tensor
    old_value: torch.Tensor
    activation_weight: float
    reward: float | None = None


class MetaController(nn.Module):
    """LSTM Gaussian meta-policy and its PPO-style outer-loop update."""

    action_scales = torch.tensor([0.30, 0.02, 0.05])

    def __init__(
        self,
        *,
        device: torch.device,
        backend: RebuiltLSTMBackend,
        use_fast: bool,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.feature_norm = nn.LayerNorm(5)
        self.cell = OptimizedLSTMCell(hidden_size, backend, use_fast)
        self.trunk = nn.Sequential(nn.Linear(hidden_size, 128), nn.ReLU())
        self.action_mean = nn.Linear(128, 3)
        self.value_head = nn.Linear(128, 1)
        # The manuscript specifies meta-controller action standard deviation 0.10.
        self.register_buffer("action_std", torch.full((3,), 0.10))
        self.register_buffer("scales", self.action_scales.clone())
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self.to(device)

    def _initial_hidden(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch_size, self.hidden_size, device=self.device),
            torch.zeros(batch_size, self.hidden_size, device=self.device),
        )

    def _evaluate(
        self,
        features: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
        activation_weight: float,
        *,
        validate_outputs: bool = False,
        segments: SynchronizedSegmentTimer | None = None,
    ) -> tuple[Normal, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        normalise = lambda: self.feature_norm(features)
        normalized = segments.measure("input_preparation", normalise) if segments else normalise()
        h_new, c_new = self.cell(
            normalized,
            hidden,
            activation_weight,
            validate_outputs=validate_outputs,
            segments=segments,
        )
        def policy_value():
            latent = self.trunk(h_new)
            mean = torch.tanh(self.action_mean(latent)) * self.scales
            distribution = Normal(mean, self.action_std.expand_as(mean))
            return distribution, self.value_head(latent).squeeze(-1)
        distribution, value = segments.measure("policy_value_head", policy_value) if segments else policy_value()
        return distribution, value, (h_new, c_new)

    @torch.no_grad()
    def act(self, features: torch.Tensor, activation_weight: float) -> tuple[torch.Tensor, MetaTransition]:
        if self.hidden is None:
            self.hidden = self._initial_hidden(features.shape[0])
        h_before, c_before = (self.hidden[0].detach(), self.hidden[1].detach())
        distribution, value, hidden_after = self._evaluate(
            features, (h_before, c_before), activation_weight, validate_outputs=self.cell.use_fast
        )
        action = distribution.sample()
        old_log_probability = distribution.log_prob(action).sum(-1)
        self.hidden = (hidden_after[0].detach(), hidden_after[1].detach())
        return action, MetaTransition(
            features=features.detach(),
            h_before=h_before,
            c_before=c_before,
            action=action.detach(),
            controller_output=distribution.mean.detach(),
            old_log_probability=old_log_probability.detach(),
            old_value=value.detach(),
            activation_weight=activation_weight,
        )

    def update(
        self,
        transitions: list[MetaTransition],
        *,
        gamma: float = 0.99,
        segments: SynchronizedSegmentTimer | None = None,
        batched_replay: bool = False,
        debug: dict | None = None,
    ) -> float:
        """Optimise the stored sequential controller decisions with clipped PPO."""
        if not transitions or any(item.reward is None for item in transitions):
            return 0.0
        def prepare_update():
            rewards = torch.tensor([float(item.reward) for item in transitions], device=self.device)
            old_values = torch.cat([item.old_value.reshape(1) for item in transitions]).to(self.device)
            returns = torch.zeros_like(rewards)
            running = torch.zeros((), device=self.device)
            for index in range(len(transitions) - 1, -1, -1):
                running = rewards[index] + gamma * running
                returns[index] = running
            advantages = returns - old_values
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
            return returns, advantages
        returns, advantages = segments.measure("meta_update_input_preparation", prepare_update) if segments else prepare_update()
        policy_loss = torch.zeros((), device=self.device)
        if debug is not None:
            debug["replay_execution"] = "batched" if batched_replay else "sequential"
            debug["epoch_losses"] = []
            debug["replay_group_counts"] = []
        for _ in range(4):
            total_loss = torch.zeros((), device=self.device)
            values: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            ratios: list[torch.Tensor] = []
            if not batched_replay:
                for transition in transitions:
                    distribution, value, _ = self._evaluate(
                        transition.features,
                        (transition.h_before, transition.c_before),
                        transition.activation_weight,
                        segments=segments,
                    )
                    log_probability = distribution.log_prob(transition.action).sum(-1)
                    ratios.append(torch.exp(log_probability - transition.old_log_probability))
                    values.append(value)
                    entropies.append(distribution.entropy().sum(-1))
                if debug is not None:
                    debug["replay_group_counts"].append(len(transitions))
            else:
                # A replay transition carries its own detached h/c state, so
                # replay has no recurrent dependency between transitions.
                # Group only equal activation weights: during full activation
                # this is one B=4 call per epoch; during staged blending it
                # preserves the original per-transition blend exactly.
                groups: dict[float, list[tuple[int, MetaTransition]]] = {}
                for index, transition in enumerate(transitions):
                    groups.setdefault(float(transition.activation_weight), []).append((index, transition))
                replayed: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = [None] * len(transitions)
                for activation_weight, group in groups.items():
                    batch_features = torch.cat([item.features for _, item in group], dim=0)
                    batch_hidden = torch.cat([item.h_before for _, item in group], dim=0)
                    batch_cell = torch.cat([item.c_before for _, item in group], dim=0)
                    batch_actions = torch.cat([item.action for _, item in group], dim=0)
                    batch_old_log_probability = torch.cat(
                        [item.old_log_probability.reshape(1) for _, item in group], dim=0
                    )
                    distribution, batch_values, _ = self._evaluate(
                        batch_features,
                        (batch_hidden, batch_cell),
                        activation_weight,
                        segments=segments,
                    )
                    batch_log_probability = distribution.log_prob(batch_actions).sum(-1)
                    batch_ratios = torch.exp(batch_log_probability - batch_old_log_probability)
                    batch_entropies = distribution.entropy().sum(-1)
                    for local_index, (original_index, _) in enumerate(group):
                        replayed[original_index] = (
                            batch_ratios[local_index : local_index + 1],
                            batch_values[local_index : local_index + 1],
                            batch_entropies[local_index : local_index + 1],
                        )
                for item in replayed:
                    if item is None:
                        raise RuntimeError("Incomplete batched replay assembly.")
                    ratio, value, entropy = item
                    ratios.append(ratio)
                    values.append(value)
                    entropies.append(entropy)
                if debug is not None:
                    debug["replay_group_counts"].append(len(groups))
            def assemble_loss():
                nonlocal policy_loss
                ratio = torch.cat(ratios)
                surrogate_one = ratio * advantages
                surrogate_two = torch.clamp(ratio, 0.8, 1.2) * advantages
                policy_loss = -torch.minimum(surrogate_one, surrogate_two).mean()
                value_loss = torch.nn.functional.mse_loss(torch.cat(values), returns)
                entropy = torch.cat(entropies).mean()
                return policy_loss + 0.5 * value_loss - 0.01 * entropy
            total_loss = segments.measure("loss", assemble_loss) if segments else assemble_loss()
            if debug is not None:
                debug["epoch_losses"].append(float(total_loss.detach().item()))
            zero_grad = lambda: self.optimizer.zero_grad(set_to_none=True)
            if segments:
                segments.measure("optimizer_zero_grad", zero_grad)
                segments.measure("backward", total_loss.backward)
            else:
                zero_grad()
                total_loss.backward()
            self.cell.backend.record_backward()
            def optimizer_step() -> float:
                norm = float(torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0).item())
                self.optimizer.step()
                return norm
            gradient_norm = segments.measure("optimizer_step", optimizer_step) if segments else optimizer_step()
        self.hidden = None
        return gradient_norm


@dataclass
class TrainingHistory:
    controller_outputs: list[list[float]]
    reward_deltas: list[float]
    meta_rewards: list[float]
    hyperparameters: list[dict[str, float]]
    mean_reward_curve: list[float]
    checkpoint_evaluations: list[dict]


class AMRLAgent:
    """Same learning objective and controller logic for AMRL and AMRL-Fast."""

    def __init__(
        self,
        env,
        *,
        evaluation_env=None,
        device: torch.device,
        seed: int,
        use_fast: bool,
        backend: RebuiltLSTMBackend,
        adaptation_interval: int = 10_000,
        controller_start_after_steps: int = 0,
        controller_hidden_size: int = 256,
        controller_batched_replay: bool = False,
        enable_rollout_backend: bool = False,
        enable_foreach_adam: bool = False,
        enable_fused_adam: bool = False,
        warmup_steps: int = 10_000,
        transition_steps: int = 10_000,
    ) -> None:
        set_global_seed(seed)
        self.device = device
        self.base = PPOBaseAgent(
            env,
            device=device,
            seed=seed,
            rollout_steps=adaptation_interval,
            enable_rollout_backend=enable_rollout_backend,
            enable_foreach_adam=enable_foreach_adam,
            enable_fused_adam=enable_fused_adam,
        )
        self.evaluation_env = evaluation_env if evaluation_env is not None else self.base.env
        self.backend = backend
        self.use_fast = use_fast
        self.controller = MetaController(
            device=device,
            backend=backend,
            use_fast=use_fast,
            hidden_size=controller_hidden_size,
        )
        self.adaptation_interval = adaptation_interval
        self.controller_start_after_steps = controller_start_after_steps
        self.controller_batched_replay = controller_batched_replay
        self.warmup_steps = warmup_steps
        self.transition_steps = transition_steps
        self.telemetry = TrialTelemetry()
        self.total_steps = 0
        self.previous_mean_reward: float | None = None
        self.pending_meta: MetaTransition | None = None
        self.meta_rollout: list[MetaTransition] = []
        self.warmup_gradients: list[float] = []
        self.warmup_entropies: list[float] = []
        self.controller_gradient_samples: list[float] = []
        self.controller_entropy_samples: list[float] = []
        self.gradient_limit: float | None = None
        self.entropy_mean: float | None = None
        self.entropy_std: float | None = None
        self.force_standard_cycles = 0
        self.health_fallback_count = 0
        # Audit-only ledger for the requested post-transition stability rate.
        # It is populated from real dispatch/fallback deltas after the staged
        # transition has completed; it does not participate in admission.
        self.post_transition_eligible_dispatch_count = 0
        self.post_transition_fallback_count = 0
        self.post_transition_nan_inf_count = 0

    def _activation_weight(self) -> float:
        if not self.use_fast or self.force_standard_cycles > 0:
            return 0.0
        if self.total_steps <= self.warmup_steps:
            return 0.0
        if self.total_steps <= self.warmup_steps + self.transition_steps:
            return (self.total_steps - self.warmup_steps) / self.transition_steps
        return 1.0

    def _meta_features(self, stats: dict[str, float]) -> torch.Tensor:
        return torch.tensor(
            [[
                stats["mean_reward"], stats["reward_std"], stats["gradient_norm"],
                stats["policy_entropy"], stats["value_loss"],
            ]],
            dtype=torch.float32,
            device=self.device,
        )

    def _apply_adjustments(self, adjustments: torch.Tensor) -> dict[str, float]:
        raw = adjustments.detach().squeeze(0).cpu().numpy()
        values = self.base.hyperparameters.copy()
        values["learning_rate"] = float(np.clip(values["learning_rate"] * np.exp(raw[0]), 3e-5, 3e-3))
        values["gamma"] = float(np.clip(values["gamma"] + raw[1], 0.90, 0.99))
        values["entropy_coefficient"] = float(
            np.clip(values["entropy_coefficient"] * np.exp(raw[2]), 1e-5, 1.0)
        )
        self.base.set_hyperparameters(values)
        return values

    def _calibrate_control_limits(self, stats: dict[str, float], meta_gradient_norm: float) -> None:
        if self.total_steps <= self.warmup_steps:
            self.warmup_gradients.append(meta_gradient_norm)
            self.warmup_entropies.append(stats["policy_entropy"])
        # A PPO meta-gradient is not available until a complete meta-rollout
        # exists.  Do not invent a zero-valued reference threshold; estimate it
        # only after observed finite gradients have been collected.
        if meta_gradient_norm > 0 and np.isfinite(meta_gradient_norm):
            self.controller_gradient_samples.append(meta_gradient_norm)
            self.controller_entropy_samples.append(stats["policy_entropy"])
        if len(self.controller_gradient_samples) >= 2 and self.gradient_limit is None:
            self.gradient_limit = float(
                np.mean(self.controller_gradient_samples) + 3 * np.std(self.controller_gradient_samples)
            )
            self.entropy_mean = float(np.mean(self.controller_entropy_samples))
            self.entropy_std = float(np.std(self.controller_entropy_samples))

    def _requires_fallback(self, stats: dict[str, float], meta_gradient_norm: float) -> bool:
        if not np.isfinite(meta_gradient_norm):
            return True
        if self.gradient_limit is None:
            return False
        entropy_outside = (
            self.entropy_std is not None
            and self.entropy_std > 0
            and abs(stats["policy_entropy"] - float(self.entropy_mean)) > 3 * self.entropy_std
        )
        return meta_gradient_norm > self.gradient_limit or entropy_outside

    def train(
        self,
        total_timesteps: int,
        checkpoint_callback: Callable[[int], tuple] | None = None,
    ) -> TrainingHistory:
        history = TrainingHistory([], [], [], [], [], [])
        while self.total_steps < total_timesteps:
            dispatch_before = self.backend.dispatch_count
            fallback_before = self.backend.fallback_count + self.health_fallback_count
            nonfinite_before = self.backend.nan_inf_count
            chunk = min(self.adaptation_interval, total_timesteps - self.total_steps)
            with synchronized_time(self.device) as timer:
                stats = self.base.train_step(chunk)
            self.telemetry.timing.base_agent_training_time += timer.seconds
            self.total_steps += chunk
            history.mean_reward_curve.append(float(stats["mean_reward"]))
            # Reproduce the public train_meta.py condition exactly when a
            # nonzero threshold is configured: t > threshold, not t >= it.
            if self.total_steps <= self.controller_start_after_steps:
                self.previous_mean_reward = float(stats["mean_reward"])
                continue
            meta_reward = 0.0 if self.previous_mean_reward is None else float(
                stats["mean_reward"] - self.previous_mean_reward
            )
            self.previous_mean_reward = float(stats["mean_reward"])

            # Pair U_t with the reward change in the following adaptation
            # window, exactly as the manuscript's one-cycle delayed MI rule.
            if self.pending_meta is not None:
                self.pending_meta.reward = meta_reward
                self.meta_rollout.append(self.pending_meta)
                # U_t is the controller's bounded three-dimensional mean
                # output, not the stochastic action used by outer PPO.  The
                # action remains unchanged for optimisation/application.
                history.controller_outputs.append(
                    self.pending_meta.controller_output.squeeze(0).cpu().tolist()
                )
                history.reward_deltas.append(meta_reward)
                history.meta_rewards.append(meta_reward)
                self.pending_meta = None

            meta_gradient_norm = 0.0
            if len(self.meta_rollout) >= 4:
                with synchronized_time(self.device) as timer:
                    meta_gradient_norm = self.controller.update(
                        self.meta_rollout,
                        batched_replay=self.controller_batched_replay,
                    )
                self.telemetry.timing.meta_learner_backward_time += timer.seconds
                self.meta_rollout.clear()
            self._calibrate_control_limits(stats, meta_gradient_norm)
            meta_update_finite = (not self.use_fast) or self.backend.note_meta_update_health(meta_gradient_norm)
            if (not meta_update_finite) or self._requires_fallback(stats, meta_gradient_norm):
                if self.force_standard_cycles == 0:
                    self.health_fallback_count += 1
                self.force_standard_cycles = 1

            features = self._meta_features(stats)
            activation_weight = self._activation_weight()
            with synchronized_time(self.device) as timer:
                adjustments, transition = self.controller.act(features, activation_weight)
            self.telemetry.timing.meta_learner_forward_time += timer.seconds
            with synchronized_time(self.device) as timer:
                values = self._apply_adjustments(adjustments)
            self.telemetry.timing.hyperparameter_update_time += timer.seconds
            history.hyperparameters.append(values)
            self.pending_meta = transition
            if self.force_standard_cycles:
                self.force_standard_cycles -= 1
            if checkpoint_callback is not None:
                checkpoint = checkpoint_callback(self.total_steps)
                if checkpoint is not None:
                    mean_return, std_return, episode_returns = checkpoint
                    history.checkpoint_evaluations.append(
                        {
                            "step": float(self.total_steps),
                            "mean_return": float(mean_return),
                            "std_return": float(std_return),
                            "episode_returns": [float(value) for value in episode_returns],
                        }
                    )
            if self.total_steps > self.warmup_steps + self.transition_steps:
                self.post_transition_eligible_dispatch_count += max(
                    0, self.backend.dispatch_count - dispatch_before
                )
                self.post_transition_fallback_count += max(
                    0, self.backend.fallback_count + self.health_fallback_count - fallback_before
                )
                self.post_transition_nan_inf_count += max(
                    0, self.backend.nan_inf_count - nonfinite_before
                )

        # The final controller action has no subsequent adaptation window and
        # is intentionally excluded from I(U; Delta R) and the outer PPO loss.
        self.telemetry.stability.eligible_dispatch_count = self.backend.dispatch_count
        self.telemetry.stability.fallback_count = (
            self.backend.fallback_count + self.health_fallback_count
        )
        self.telemetry.stability.nan_inf_count = self.backend.nan_inf_count
        return history

    def stability_audit(self) -> dict[str, int | float]:
        """Additional non-semantic stability fields for formal result JSON."""
        return {
            "post_transition_eligible_dispatch_count": self.post_transition_eligible_dispatch_count,
            "post_transition_fallback_count": self.post_transition_fallback_count,
            "post_transition_fallback_rate": (
                self.post_transition_fallback_count / self.post_transition_eligible_dispatch_count
                if self.post_transition_eligible_dispatch_count else 0.0
            ),
            "post_transition_nan_inf_count": self.post_transition_nan_inf_count,
        }

    def evaluate(self, n_episodes: int = 20, *, return_episode_returns: bool = False):
        rewards: list[float] = []
        for _ in range(n_episodes):
            obs, _ = self.evaluation_env.reset()
            done = False
            reward_sum = 0.0
            while not done:
                action = self.base.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.evaluation_env.step(action)
                reward_sum += float(reward)
                done = terminated or truncated
            rewards.append(reward_sum)
        mean, std = float(np.mean(rewards)), float(np.std(rewards))
        return (mean, std, rewards) if return_episode_returns else (mean, std)
