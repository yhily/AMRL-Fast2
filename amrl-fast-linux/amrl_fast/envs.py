"""Shared Gymnasium/MuJoCo environment factory used by every benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium.wrappers import TimeLimit


@dataclass(frozen=True)
class EnvironmentSpec:
    """Paper-facing environment name and runtime-compatible Gym IDs."""

    name: str
    candidates: tuple[str, ...]
    default_max_episode_steps: Optional[int] = None
    continuous_actions: bool = False
    bsuite_name: Optional[str] = None


class BSuiteGymnasiumAdapter(gym.Env):
    """Minimal Gymnasium adapter for the public ``bsuite/memory_len`` task."""

    metadata = {"render_modes": []}

    def __init__(self, experiment_name: str) -> None:
        super().__init__()
        try:
            import bsuite
        except ImportError as exc:
            raise RuntimeError("BSuite Memory requires the optional 'bsuite' package.") from exc
        # The public AMRL repository calls bsuite.load('memory_len').  Keep
        # that behavior first, with the documented instance-ID fallback used
        # by newer bsuite releases.
        try:
            self._env = bsuite.load(experiment_name)
        except Exception:
            self._env = bsuite.load_from_id(f"{experiment_name}/0")
        obs_spec = self._env.observation_spec()
        shape = tuple(obs_spec.shape)
        dtype = np.dtype(getattr(obs_spec, "dtype", np.float32))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=shape, dtype=dtype
        )
        action_spec = self._env.action_spec()
        self.action_space = gym.spaces.Discrete(int(action_spec.num_values))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        timestep = self._env.reset()
        return np.asarray(timestep.observation, dtype=self.observation_space.dtype), {}

    def step(self, action):
        timestep = self._env.step(int(action))
        reward = 0.0 if timestep.reward is None else float(timestep.reward)
        terminated = bool(timestep.last())
        return (
            np.asarray(timestep.observation, dtype=self.observation_space.dtype),
            reward,
            terminated,
            False,
            {},
        )


ENVIRONMENTS: dict[str, EnvironmentSpec] = {
    "CartPole-v1": EnvironmentSpec(
        name="CartPole-v1", candidates=("CartPole-v1",),
        default_max_episode_steps=1000, continuous_actions=False,
    ),
    "MountainCar-v0": EnvironmentSpec(
        name="MountainCar-v0", candidates=("MountainCar-v0",),
        continuous_actions=False,
    ),
    # The paper-facing v2 names are tried first. Gymnasium installations normally
    # expose v4 instead; the resolved ID is always recorded in result JSON.
    "HalfCheetah-v2": EnvironmentSpec(
        name="HalfCheetah-v2", candidates=("HalfCheetah-v2", "HalfCheetah-v4"),
        continuous_actions=True,
    ),
    # New formal continuous-control protocol: exact runtime ID only.  Keeping
    # this separate from the legacy v2 alias prevents a v4 run from ever being
    # reported under a paper-facing v2 label.
    "HalfCheetah-v4": EnvironmentSpec(
        name="HalfCheetah-v4", candidates=("HalfCheetah-v4",),
        continuous_actions=True,
    ),
    "Hopper-v2": EnvironmentSpec(
        name="Hopper-v2", candidates=("Hopper-v2", "Hopper-v4"),
        continuous_actions=True,
    ),
    # Formal extension configurations use these explicit IDs.  The legacy v2
    # aliases remain available only for compatibility probes and are never
    # silently relabeled in the formal result directory.
    "Hopper-v4": EnvironmentSpec(
        name="Hopper-v4", candidates=("Hopper-v4",),
        continuous_actions=True,
    ),
    "Walker2d-v2": EnvironmentSpec(
        name="Walker2d-v2", candidates=("Walker2d-v2", "Walker2d-v4"),
        continuous_actions=True,
    ),
    "Walker2d-v4": EnvironmentSpec(
        name="Walker2d-v4", candidates=("Walker2d-v4",),
        continuous_actions=True,
    ),
    # Box2D continuous-control screening environments.  These use exact
    # Gymnasium v3 IDs and the same unified PPO/AMRL pipeline as MuJoCo.
    "LunarLanderContinuous-v3": EnvironmentSpec(
        name="LunarLanderContinuous-v3",
        candidates=("LunarLanderContinuous-v3",),
        continuous_actions=True,
    ),
    "BipedalWalker-v3": EnvironmentSpec(
        name="BipedalWalker-v3",
        candidates=("BipedalWalker-v3",),
        continuous_actions=True,
    ),
    "BSuite Memory": EnvironmentSpec(
        name="BSuite Memory",
        candidates=("bsuite/memory_len",),
        continuous_actions=False,
        bsuite_name="memory_len",
    ),
}


def get_environment_spec(env_name: str) -> EnvironmentSpec:
    if env_name not in ENVIRONMENTS:
        raise ValueError(
            f"Unsupported environment {env_name!r}. Supported: {sorted(ENVIRONMENTS)}"
        )
    return ENVIRONMENTS[env_name]


def resolve_environment(
    env_name: str,
    *,
    allow_version_fallback: bool = False,
) -> tuple[EnvironmentSpec, str]:
    """Resolve an exact paper ID, or an explicit compatible MuJoCo version."""
    spec = get_environment_spec(env_name)
    if spec.bsuite_name is not None:
        # The adapter is instantiated by make_env; do not attempt gym.make.
        return spec, spec.candidates[0]
    candidates = spec.candidates if allow_version_fallback else spec.candidates[:1]
    errors: list[str] = []
    for candidate in candidates:
        try:
            env = _make_candidate(candidate)
            env.close()
            return spec, candidate
        except Exception as exc:  # Gym version errors are backend-specific.
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    policy = "enabled" if allow_version_fallback else "disabled"
    raise RuntimeError(
        f"Could not create {env_name}. Version fallback is {policy}. "
        f"Attempts: {' | '.join(errors)}"
    )


def _make_candidate(
    candidate: str,
    render_mode: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
) -> gym.Env:
    """Use Gymnasium first, then an optional legacy Gym-v2 compatibility path."""
    kwargs = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    if max_episode_steps is not None:
        # Gymnasium applies this while constructing its sole TimeLimit wrapper.
        # Wrapping an already-created CartPole-v1 afterwards leaves its inner
        # default 500-step TimeLimit active and does *not* create H=1000.
        kwargs["max_episode_steps"] = max_episode_steps
    try:
        return gym.make(candidate, **kwargs)
    except Exception as gymnasium_error:
        try:
            import gym as legacy_gym
            from shimmy import GymV21CompatibilityV0

            legacy = legacy_gym.make(candidate)
            return GymV21CompatibilityV0(env=legacy)
        except Exception:
            raise gymnasium_error


def make_env(
    env_name: str,
    *,
    seed: Optional[int] = None,
    max_episode_steps: Optional[int] = None,
    allow_version_fallback: bool = False,
    render_mode: Optional[str] = None,
) -> tuple[gym.Env, str]:
    """Create one consistently seeded environment and return its resolved ID."""
    spec, resolved_id = resolve_environment(
        env_name, allow_version_fallback=allow_version_fallback
    )
    horizon = max_episode_steps
    if horizon is None:
        horizon = spec.default_max_episode_steps
    env = (
        BSuiteGymnasiumAdapter(spec.bsuite_name)
        if spec.bsuite_name is not None
        else _make_candidate(
            resolved_id, render_mode=render_mode, max_episode_steps=horizon
        )
    )
    # The Gymnasium branch above already has the requested TimeLimit.  Keep
    # this fallback only for adapters/legacy environments that cannot accept
    # max_episode_steps during construction.
    if horizon is not None and getattr(env, "_max_episode_steps", None) != horizon:
        env = TimeLimit(env, max_episode_steps=horizon)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env, resolved_id
