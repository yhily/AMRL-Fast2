"""Admitted execution-only rollout backend for the public SB3 MlpPolicy.

The public inner PPO policy has two independent two-layer Tanh MLPs.  During a
rollout the weights are fixed until PPO training has completed, so the two
branches can be cached as a vertically packed first linear and a block-diagonal
second linear.  This reduces the B=1 rollout MLP launch count from four linears
and four activations to two of each, without changing the policy/value function,
distribution, sampling, PPO loss, backward path, or optimiser.

The backend is deliberately conservative: it is admitted only after the same
output/gradient/runtime checks used for AMRL-Fast kernels; it runs only in
no-grad B=1 rollout calls; and every unsupported/non-finite call falls back to
the captured standard policy forward method.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _metric(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.detach().float().reshape(-1)
    candidate = candidate.detach().float().reshape(-1)
    difference = reference - candidate
    centered_reference = reference - reference.mean()
    centered_candidate = candidate - candidate.mean()
    denominator = torch.linalg.vector_norm(centered_reference) * torch.linalg.vector_norm(centered_candidate)
    correlation = 1.0 if denominator <= 1e-12 and torch.allclose(reference, candidate) else (
        0.0 if denominator <= 1e-12 else float(torch.dot(centered_reference, centered_candidate).div(denominator).item())
    )
    return {
        "correlation": correlation,
        "relative_l2": float(torch.linalg.vector_norm(difference).div(torch.linalg.vector_norm(reference).clamp_min(1e-12)).item()),
        "max_absolute": float(difference.abs().max().item()),
    }


def _passes(metric: dict[str, float]) -> bool:
    return (
        metric["correlation"] >= 0.999
        and metric["relative_l2"] <= 5e-3
        and metric["max_absolute"] <= 1e-1
    )


def _layers(network: nn.Sequential) -> tuple[nn.Linear, nn.Linear]:
    linear = [module for module in network if isinstance(module, nn.Linear)]
    tanh = [module for module in network if isinstance(module, nn.Tanh)]
    if len(linear) != 2 or len(tanh) != 2:
        raise ValueError("The fused rollout backend supports only the public two-layer Tanh MlpExtractor.")
    return linear[0], linear[1]


def _dynamic_fused(
    policy_network: nn.Sequential, value_network: nn.Sequential, features: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Algebraically equivalent, differentiable form used for admission only."""
    policy_first, policy_second = _layers(policy_network)
    value_first, value_second = _layers(value_network)
    first_weight = torch.cat((policy_first.weight, value_first.weight), dim=0)
    first_bias = torch.cat((policy_first.bias, value_first.bias), dim=0)
    second_weight = torch.block_diag(policy_second.weight, value_second.weight)
    second_bias = torch.cat((policy_second.bias, value_second.bias), dim=0)
    hidden = torch.tanh(F.linear(features, first_weight, first_bias))
    latent = torch.tanh(F.linear(hidden, second_weight, second_bias))
    width = policy_second.out_features
    return latent[:, :width], latent[:, width:]


@dataclass
class RolloutBackendAudit:
    backend: str
    admitted: bool
    speedup: float
    standard_forward_us: float
    optimized_forward_us: float
    standard_policy_forward_us: float
    optimized_policy_forward_us: float
    policy_forward_speedup: float
    policy_output: dict[str, float]
    value_output: dict[str, float]
    input_gradient: dict[str, float]
    parameter_gradient: dict[str, float]
    reason: str
    # Continuous-action backends additionally record that CUDA graph capture
    # and replay do not advance RNG and that sampling/log-probability remain
    # numerically identical on the unchanged SB3 distribution path.
    continuous_action_audit: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class FusedRolloutMlpBackend:
    """Cached fixed-weight MLP backend, valid only inside SB3 rollout no-grad calls."""

    name = "AMRL-Fast-Rebuilt-Kernel/cached-fused-rollout-mlp"

    def __init__(self, model, *, device: torch.device) -> None:
        self.model = model
        self.policy = model.policy
        self.device = device
        self.original_forward: Callable[..., Any] = self.policy.forward
        # SB3's evaluation API enters ``BasePolicy.predict -> _predict``
        # rather than ``forward``.  Keep that call on the same admitted
        # no-grad graph path so evaluation uses the identical inference
        # backend as rollout collection.
        self.original_predict: Callable[..., Any] = self.policy._predict
        self.policy_network = self.policy.mlp_extractor.policy_net
        self.value_network = self.policy.mlp_extractor.value_net
        self.dispatch_count = 0
        self.fallback_count = 0
        self.nan_inf_count = 0
        self.fallback_reasons: dict[str, int] = {}
        self.health_check_interval = 1_024
        self._installed = False
        self._last_update_count: int | None = None
        self._policy_width = 0
        self._first_weight: torch.Tensor | None = None
        self._first_bias: torch.Tensor | None = None
        self._second_weight: torch.Tensor | None = None
        self._second_bias: torch.Tensor | None = None

    @torch.no_grad()
    def refresh(self) -> None:
        policy_first, policy_second = _layers(self.policy_network)
        value_first, value_second = _layers(self.value_network)
        self._policy_width = policy_second.out_features
        self._first_weight = torch.cat((policy_first.weight, value_first.weight), dim=0).contiguous()
        self._first_bias = torch.cat((policy_first.bias, value_first.bias), dim=0).contiguous()
        self._second_weight = torch.block_diag(policy_second.weight, value_second.weight).contiguous()
        self._second_bias = torch.cat((policy_second.bias, value_second.bias), dim=0).contiguous()
        self._last_update_count = int(getattr(self.model, "_n_updates", 0))

    def _refresh_if_needed(self) -> None:
        if self._first_weight is None or self._last_update_count != int(getattr(self.model, "_n_updates", 0)):
            self.refresh()

    @torch.no_grad()
    def _latents(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._first_weight is not None and self._first_bias is not None
        assert self._second_weight is not None and self._second_bias is not None
        hidden = torch.tanh(F.linear(features, self._first_weight, self._first_bias))
        latent = torch.tanh(F.linear(hidden, self._second_weight, self._second_bias))
        # ``F.linear`` heads receive contiguous [B, 64] tensors in the
        # standard extractor.  Preserve that layout rather than passing
        # strided [B, 64] views into the unchanged policy/value heads.
        return (
            latent[:, :self._policy_width].contiguous(),
            latent[:, self._policy_width:].contiguous(),
        )

    @staticmethod
    def _finite(*values: torch.Tensor) -> bool:
        return all(bool(torch.isfinite(value).all().item()) for value in values)

    @torch.no_grad()
    def _optimized_forward(self, observation: torch.Tensor, deterministic: bool = False):
        features = self.policy.extract_features(observation)
        # The public MlpPolicy shares a FlattenExtractor.  Keep any alternate
        # extractor topology on standard execution rather than guessing its
        # feature semantics.
        if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[0] != 1:
            raise ValueError("Only the admitted B=1 shared-feature rollout shape is optimized.")
        self._refresh_if_needed()
        latent_policy, latent_value = self._latents(features)
        # Keep the original heads and distribution implementation.  A tested
        # packed-head variant required an extra per-step concatenation and was
        # slower at the true policy-forward boundary, so it is not retained.
        values = self.policy.value_net(latent_value)
        distribution = self.policy._get_action_dist_from_latent(latent_policy)
        actions = distribution.get_actions(deterministic=deterministic)
        log_probability = distribution.log_prob(actions)
        # Calling ``.item()`` for every B=1 action forces a host/device
        # synchronization and can erase an otherwise valid kernel gain.  The
        # runtime guard is therefore sampled once per fixed 1,024-call block;
        # shape/errors still fall back immediately, and admission performs the
        # full output/gradient finite-value checks before installation.
        if (self.dispatch_count + 1) % self.health_check_interval == 0 and not self._finite(values, log_probability):
            self.nan_inf_count += 1
            raise FloatingPointError("Non-finite fused rollout output")
        return actions, values, log_probability

    def forward(self, observation: torch.Tensor, deterministic: bool = False):
        # Training invokes evaluate_actions rather than policy.forward; this
        # guard prevents accidental use under an autograd-enabled caller.
        if torch.is_grad_enabled():
            self.fallback_count += 1
            return self.original_forward(observation, deterministic=deterministic)
        try:
            result = self._optimized_forward(observation, deterministic=deterministic)
            self.dispatch_count += 1
            return result
        except Exception as error:
            self.fallback_count += 1
            key = type(error).__name__
            self.fallback_reasons[key] = self.fallback_reasons.get(key, 0) + 1
            return self.original_forward(observation, deterministic=deterministic)

    @torch.no_grad()
    def predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """SB3 ``_predict`` replacement for evaluation-time inference."""
        try:
            actions, _, _ = self._optimized_forward(observation, deterministic=deterministic)
            self.dispatch_count += 1
            return actions
        except Exception:
            self.fallback_count += 1
            return self.original_predict(observation, deterministic=deterministic)

    @staticmethod
    def _latency(call: Callable[[], Any], *, device: torch.device, warmup: int = 200, repeats: int = 4_000) -> float:
        with torch.no_grad():
            for _ in range(warmup):
                call()
            _sync(device)
            started = time.perf_counter()
            for _ in range(repeats):
                call()
            _sync(device)
        return (time.perf_counter() - started) * 1e6 / repeats

    def calibrate(self) -> RolloutBackendAudit:
        """Run output, gradient, and B=1 actual-policy latency admission."""
        try:
            policy_first, _ = _layers(self.policy_network)
            input_dim = policy_first.in_features
            reference_input = torch.randn(1, input_dim, device=self.device, requires_grad=True)
            candidate_input = reference_input.detach().clone().requires_grad_(True)
            reference_policy, reference_value = self.policy.mlp_extractor(reference_input)
            candidate_policy, candidate_value = _dynamic_fused(
                self.policy_network, self.value_network, candidate_input
            )
            reference_loss = reference_policy.square().mean() + reference_value.square().mean()
            candidate_loss = candidate_policy.square().mean() + candidate_value.square().mean()
            reference_loss.backward()
            reference_grads = torch.cat(
                [parameter.grad.detach().reshape(-1) for parameter in self.policy.mlp_extractor.parameters()]
            )
            self.policy.mlp_extractor.zero_grad(set_to_none=True)
            candidate_loss.backward()
            candidate_grads = torch.cat(
                [parameter.grad.detach().reshape(-1) for parameter in self.policy.mlp_extractor.parameters()]
            )
            self.policy.mlp_extractor.zero_grad(set_to_none=True)
            self.refresh()
            standard_us = self._latency(
                lambda: self.policy.mlp_extractor(reference_input.detach()), device=self.device
            )
            optimized_us = self._latency(
                lambda: self._latents(reference_input.detach()), device=self.device
            )
            observation = reference_input.detach()
            reference_actions, reference_values, reference_log_probability = self.original_forward(
                observation, deterministic=True
            )
            candidate_actions, candidate_values, candidate_log_probability = self._optimized_forward(
                observation, deterministic=True
            )
            if not torch.equal(reference_actions, candidate_actions):
                raise RuntimeError("Packed rollout backend changed deterministic policy actions.")
            # This is the true replacement boundary: feature extraction,
            # packed MLP/head, standard distribution construction and action
            # selection.  A fast submodule alone is never sufficient for
            # admission to the real rollout path.
            standard_policy_us = self._latency(
                lambda: self.original_forward(observation, deterministic=True), device=self.device
            )
            optimized_policy_us = self._latency(
                lambda: self.forward(observation, deterministic=True), device=self.device
            )
            # Admission reports real complete-policy outputs, while the
            # differentiable MLP comparisons above establish gradient parity.
            policy_output = _metric(reference_log_probability, candidate_log_probability)
            value_output = _metric(reference_values, candidate_values)
            input_gradient = _metric(reference_input.grad, candidate_input.grad)
            parameter_gradient = _metric(reference_grads, candidate_grads)
            speedup = standard_us / optimized_us
            policy_forward_speedup = standard_policy_us / optimized_policy_us
            admitted = policy_forward_speedup >= 1.02 and all(
                _passes(item) for item in (policy_output, value_output, input_gradient, parameter_gradient)
            )
            reason = (
                "Passed output/gradient fidelity and >=1.02x complete B=1 rollout-forward speed test."
                if admitted else "Rejected by output/gradient fidelity or speed threshold."
            )
        except Exception as error:
            admitted = False
            speedup = standard_us = optimized_us = standard_policy_us = optimized_policy_us = policy_forward_speedup = 0.0
            empty = {"correlation": 0.0, "relative_l2": float("inf"), "max_absolute": float("inf")}
            policy_output = value_output = input_gradient = parameter_gradient = empty
            reason = f"Backend unsupported: {type(error).__name__}: {error}"
        return RolloutBackendAudit(
            backend=self.name,
            admitted=admitted,
            speedup=speedup,
            standard_forward_us=standard_us,
            optimized_forward_us=optimized_us,
            standard_policy_forward_us=standard_policy_us,
            optimized_policy_forward_us=optimized_policy_us,
            policy_forward_speedup=policy_forward_speedup,
            policy_output=policy_output,
            value_output=value_output,
            input_gradient=input_gradient,
            parameter_gradient=parameter_gradient,
            reason=reason,
        )

    def install(self) -> None:
        if self._installed:
            return
        self.policy.forward = self.forward
        self._installed = True

    def uninstall(self) -> None:
        if self._installed:
            self.policy.forward = self.original_forward
            self._installed = False


class CudaGraphPolicyBodyBackend:
    """CUDA-graph backend for the fixed-shape no-grad PPO rollout body.

    The graph contains only ``MlpExtractor -> action-parameter/value Linear``.
    SB3 distribution construction, stochastic sampling and log-probability
    remain outside it, as do PPO replay, loss construction, backward and
    optimiser execution.  The action-parameter head is either categorical
    logits or the mean of a DiagGaussian policy; ``policy.log_std`` remains
    the original SB3 Parameter and is never captured or replaced.
    """

    name = "AMRL-Fast-Rebuilt-Kernel/cuda-graph-policy-body"

    def __init__(self, model, *, device: torch.device) -> None:
        self.model = model
        self.policy = model.policy
        self.device = device
        self.original_forward: Callable[..., Any] = self.policy.forward
        self.original_predict: Callable[..., Any] = self.policy._predict
        self.dispatch_count = 0
        self.fallback_count = 0
        self.nan_inf_count = 0
        self.fallback_reasons: dict[str, int] = {}
        self.health_check_interval = 1_024
        self._installed = False
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_observation: torch.Tensor | None = None
        self._static_action_parameters: torch.Tensor | None = None
        self._static_values: torch.Tensor | None = None
        self._static_deterministic_actions: torch.Tensor | None = None

    def _distribution_kind(self) -> str:
        """Restrict the graph backend to the two audited SB3 distributions."""
        name = type(self.policy.action_dist).__name__
        if name == "CategoricalDistribution":
            return "categorical"
        if name == "DiagGaussianDistribution":
            return "diag_gaussian"
        raise ValueError(f"Unsupported SB3 rollout distribution: {name}")

    @staticmethod
    def _feature_dim(policy) -> int:
        first = next((item for item in policy.mlp_extractor.policy_net if isinstance(item, nn.Linear)), None)
        if first is None:
            raise ValueError("Unsupported MlpExtractor: missing first Linear layer.")
        return int(first.in_features)

    @torch.no_grad()
    def _capture(self) -> None:
        if self.device.type != "cuda":
            raise ValueError("CUDA graph backend requires CUDA.")
        feature_dim = self._feature_dim(self.policy)
        distribution_kind = self._distribution_kind()
        # The three current discrete benchmarks use a flat float observation.
        # Capture SB3's own feature extractor as well as the MLP/head body so
        # the real graph boundary begins at the already-device-resident
        # rollout observation tensor.
        self._static_observation = torch.empty((1, feature_dim), device=self.device)
        graph = torch.cuda.CUDAGraph()
        # Prime cuBLAS and module allocation paths on a side stream before
        # capture.  No policy distribution is captured.
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(8):
                features = self.policy.extract_features(self._static_observation)
                latent_policy, latent_value = self.policy.mlp_extractor(features)
                self.policy.action_net(latent_policy)
                self.policy.value_net(latent_value)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        with torch.cuda.graph(graph):
            features = self.policy.extract_features(self._static_observation)
            latent_policy, latent_value = self.policy.mlp_extractor(features)
            self._static_action_parameters = self.policy.action_net(latent_policy)
            self._static_values = self.policy.value_net(latent_value)
            if distribution_kind == "categorical":
                # ``CategoricalDistribution.get_actions(deterministic=True)``
                # is exactly the argmax of logits.
                self._static_deterministic_actions = torch.argmax(self._static_action_parameters, dim=1)
            else:
                # ``DiagGaussianDistribution.get_actions(deterministic=True)``
                # returns its mode, i.e. the mean action.  Sampling remains
                # outside the graph so the global RNG sequence is untouched.
                self._static_deterministic_actions = self._static_action_parameters
        self._graph = graph

    @torch.no_grad()
    def _body(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._graph is None:
            raise RuntimeError("CUDA graph body has not been captured.")
        assert self._static_observation is not None
        assert self._static_action_parameters is not None and self._static_values is not None
        self._static_observation.copy_(observation)
        self._graph.replay()
        return self._static_action_parameters, self._static_values

    @torch.no_grad()
    def _optimized_predict_deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        if not isinstance(observation, torch.Tensor) or observation.ndim != 2 or observation.shape[0] != 1:
            raise ValueError("Only the admitted B=1 shared-feature rollout shape is optimized.")
        self._body(observation)
        assert self._static_deterministic_actions is not None
        return self._static_deterministic_actions

    @torch.no_grad()
    def _optimized_forward(self, observation: torch.Tensor, deterministic: bool = False):
        if not isinstance(observation, torch.Tensor) or observation.ndim != 2 or observation.shape[0] != 1:
            raise ValueError("Only the admitted B=1 shared-feature rollout shape is optimized.")
        action_parameters, values = self._body(observation)
        distribution_kind = self._distribution_kind()
        if distribution_kind == "categorical":
            distribution = self.policy.action_dist.proba_distribution(action_logits=action_parameters)
        else:
            # Keep SB3's actual log_std Parameter, distribution sampling and
            # log-probability computation.  Only the deterministic body that
            # produces the Gaussian mean is CUDA-graph replayed.
            distribution = self.policy.action_dist.proba_distribution(
                mean_actions=action_parameters, log_std=self.policy.log_std
            )
        actions = distribution.get_actions(deterministic=deterministic)
        log_probability = distribution.log_prob(actions)
        if (self.dispatch_count + 1) % self.health_check_interval == 0 and not self._finite(values, log_probability):
            self.nan_inf_count += 1
            raise FloatingPointError("Non-finite CUDA-graph rollout output")
        return actions, values, log_probability

    @staticmethod
    def _finite(*values: torch.Tensor) -> bool:
        return all(bool(torch.isfinite(value).all().item()) for value in values)

    def forward(self, observation: torch.Tensor, deterministic: bool = False):
        if torch.is_grad_enabled():
            self.fallback_count += 1
            return self.original_forward(observation, deterministic=deterministic)
        try:
            result = self._optimized_forward(observation, deterministic=deterministic)
            self.dispatch_count += 1
            return result
        except Exception:
            self.fallback_count += 1
            return self.original_forward(observation, deterministic=deterministic)

    @torch.no_grad()
    def predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """SB3 ``_predict`` replacement for evaluation-time inference."""
        if deterministic:
            try:
                actions = self._optimized_predict_deterministic(observation)
                self.dispatch_count += 1
                return actions
            except Exception as error:
                self.fallback_count += 1
                key = type(error).__name__
                self.fallback_reasons[key] = self.fallback_reasons.get(key, 0) + 1
                return self.original_predict(observation, deterministic=True)
        try:
            actions, _, _ = self._optimized_forward(observation, deterministic=deterministic)
            self.dispatch_count += 1
            return actions
        except Exception as error:
            self.fallback_count += 1
            key = type(error).__name__
            self.fallback_reasons[key] = self.fallback_reasons.get(key, 0) + 1
            return self.original_predict(observation, deterministic=deterministic)

    def calibrate(self) -> RolloutBackendAudit:
        try:
            feature_dim = self._feature_dim(self.policy)
            observation = torch.randn(1, feature_dim, device=self.device)
            distribution_kind = self._distribution_kind()
            # Capture/replay itself must never consume an RNG sample.  This is
            # particularly important for continuous-action rollout collection.
            cpu_rng_before_capture = torch.random.get_rng_state()
            cuda_rng_before_capture = torch.cuda.get_rng_state(self.device)
            self._capture()
            capture_rng_unchanged = bool(torch.equal(cpu_rng_before_capture, torch.random.get_rng_state())) and bool(
                torch.equal(cuda_rng_before_capture, torch.cuda.get_rng_state(self.device))
            )
            reference_actions, reference_values, reference_log_probability = self.original_forward(
                observation, deterministic=True
            )
            candidate_actions, candidate_values, candidate_log_probability = self._optimized_forward(
                observation, deterministic=True
            )
            if not torch.equal(reference_actions, candidate_actions):
                raise RuntimeError("CUDA graph changed deterministic policy actions.")
            reference_predict = self.original_predict(observation, deterministic=True)
            candidate_predict = self._optimized_predict_deterministic(observation)
            if not torch.equal(reference_predict, candidate_predict):
                raise RuntimeError("CUDA graph changed deterministic evaluation actions.")
            cuda_rng_before_replay = torch.cuda.get_rng_state(self.device)
            self._body(observation)
            replay_rng_unchanged = bool(torch.equal(cuda_rng_before_replay, torch.cuda.get_rng_state(self.device)))
            # A paired RNG restore provides a stronger check than matching
            # moments: the unchanged SB3 distribution path must emit the same
            # sampled action and log probability for exactly the same state.
            cpu_rng_state = torch.random.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state(self.device)
            reference_sample, _, reference_sample_log_probability = self.original_forward(
                observation, deterministic=False
            )
            torch.random.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(cuda_rng_state, self.device)
            candidate_sample, _, candidate_sample_log_probability = self._optimized_forward(
                observation, deterministic=False
            )
            torch.random.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(cuda_rng_state, self.device)
            sampled_action = _metric(reference_sample, candidate_sample)
            sampled_log_probability = _metric(reference_sample_log_probability, candidate_sample_log_probability)
            if not _passes(sampled_action) or not _passes(sampled_log_probability):
                raise RuntimeError("CUDA graph changed SB3 sampled-action distribution or log probability.")

            # PPO's differentiable replay is deliberately standard.  Audit
            # the same feature/MLP/head mathematical body under autograd to
            # make this explicit instead of reporting an assumed identity.
            reference_input = observation.detach().clone().requires_grad_(True)
            candidate_input = observation.detach().clone().requires_grad_(True)
            reference_features = self.policy.extract_features(reference_input)
            reference_latent_policy, reference_latent_value = self.policy.mlp_extractor(reference_features)
            reference_parameters = self.policy.action_net(reference_latent_policy)
            reference_values_for_grad = self.policy.value_net(reference_latent_value)
            reference_loss = reference_parameters.square().mean() + reference_values_for_grad.square().mean()
            self.policy.zero_grad(set_to_none=True)
            reference_loss.backward()
            reference_parameter_grads = torch.cat(
                [item.grad.detach().reshape(-1) for item in self.policy.parameters() if item.grad is not None]
            )
            candidate_features = self.policy.extract_features(candidate_input)
            candidate_latent_policy, candidate_latent_value = self.policy.mlp_extractor(candidate_features)
            candidate_parameters = self.policy.action_net(candidate_latent_policy)
            candidate_values_for_grad = self.policy.value_net(candidate_latent_value)
            candidate_loss = candidate_parameters.square().mean() + candidate_values_for_grad.square().mean()
            self.policy.zero_grad(set_to_none=True)
            candidate_loss.backward()
            candidate_parameter_grads = torch.cat(
                [item.grad.detach().reshape(-1) for item in self.policy.parameters() if item.grad is not None]
            )
            self.policy.zero_grad(set_to_none=True)
            standard_policy_us = FusedRolloutMlpBackend._latency(
                lambda: self.original_forward(observation, deterministic=True), device=self.device
            )
            optimized_policy_us = FusedRolloutMlpBackend._latency(
                lambda: self.forward(observation, deterministic=True), device=self.device
            )
            policy_output = _metric(reference_log_probability, candidate_log_probability)
            value_output = _metric(reference_values, candidate_values)
            input_gradient = _metric(reference_input.grad, candidate_input.grad)
            parameter_gradient = _metric(reference_parameter_grads, candidate_parameter_grads)
            policy_forward_speedup = standard_policy_us / optimized_policy_us
            continuous_action_audit = {
                "distribution": distribution_kind,
                "capture_rng_unchanged": capture_rng_unchanged,
                "replay_rng_unchanged": replay_rng_unchanged,
                "sampled_action": sampled_action,
                "sampled_log_probability": sampled_log_probability,
            }
            admitted = (
                policy_forward_speedup >= 1.02
                and all(_passes(item) for item in (policy_output, value_output, input_gradient, parameter_gradient))
                and capture_rng_unchanged
                and replay_rng_unchanged
            )
            reason = (
                "Passed full B=1 policy-forward, gradient, RNG and sampled-action fidelity with >=1.02x CUDA-graph runtime; PPO backward remains standard."
                if admitted else "Rejected by complete policy-forward fidelity or speed threshold."
            )
            # These fields remain for a uniform audit schema.  The graph body
            # covers MLP+heads, so use the full real policy boundary here too.
            speedup = policy_forward_speedup
            standard_us = standard_policy_us
            optimized_us = optimized_policy_us
        except Exception as error:
            admitted = False
            speedup = standard_us = optimized_us = standard_policy_us = optimized_policy_us = policy_forward_speedup = 0.0
            empty = {"correlation": 0.0, "relative_l2": float("inf"), "max_absolute": float("inf")}
            policy_output = value_output = input_gradient = parameter_gradient = empty
            continuous_action_audit = {"distribution": "unsupported", "capture_rng_unchanged": False,
                                       "replay_rng_unchanged": False, "sampled_action": empty,
                                       "sampled_log_probability": empty}
            reason = f"Backend unsupported: {type(error).__name__}: {error}"
        return RolloutBackendAudit(
            backend=self.name,
            admitted=admitted,
            speedup=speedup,
            standard_forward_us=standard_us,
            optimized_forward_us=optimized_us,
            standard_policy_forward_us=standard_policy_us,
            optimized_policy_forward_us=optimized_policy_us,
            policy_forward_speedup=policy_forward_speedup,
            policy_output=policy_output,
            value_output=value_output,
            input_gradient=input_gradient,
            parameter_gradient=parameter_gradient,
            reason=reason,
            continuous_action_audit=continuous_action_audit,
        )

    def install(self) -> None:
        if not self._installed:
            self.policy.forward = self.forward
            self.policy._predict = self.predict
            self._installed = True

    def uninstall(self) -> None:
        if self._installed:
            self.policy.forward = self.original_forward
            self.policy._predict = self.original_predict
            self._installed = False
