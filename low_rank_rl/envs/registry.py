"""Gymnasium environments with some defaults discretisation to allow config based experimentation."""

import gymnasium as gym

from low_rank_rl.envs.gridworld import GridWorldEnv
from low_rank_rl.envs.wrappers import (DiscreteActionWrapper, DiscretizeObsWrapper,NormalizeObsWrapper)

_ENV_DEFAULTS: dict[str, dict] = {
    "Acrobot-v1":               {"normalize_obs": True,  "n_discrete_actions": None, "discretize_obs": True,  "n_state_bins": [7, 7, 7, 7, 7, 7]},
    "CartPole-v1":              {"normalize_obs": False, "n_discrete_actions": None, "discretize_obs": True,  "n_state_bins": [10, 10, 10, 10]},
    "LunarLander-v2":           {"normalize_obs": False, "n_discrete_actions": None, "discretize_obs": False, "n_state_bins": None},
    "MountainCar-v0":           {"normalize_obs": True,  "n_discrete_actions": None, "discretize_obs": True,  "n_state_bins": [40, 40]},
    "MountainCarContinuous-v0": {"normalize_obs": True,  "n_discrete_actions": 21,   "discretize_obs": True,  "n_state_bins": [40, 40]},
    "Pendulum-v1":              {"normalize_obs": True,  "n_discrete_actions": 15,   "discretize_obs": True,  "n_state_bins": [11, 11, 15]},
    "HalfCheetah-v4":           {"normalize_obs": True,  "n_discrete_actions": 11,   "discretize_obs": False, "n_state_bins": None},
    "Hopper-v4":                {"normalize_obs": True,  "n_discrete_actions": 11,   "discretize_obs": False, "n_state_bins": None},
    "Ant-v4":                   {"normalize_obs": True,  "n_discrete_actions": 11,   "discretize_obs": True, "n_state_bins": [100,100,105,1]},
    "GridWorld-v0":             {"normalize_obs": False, "n_discrete_actions": None, "discretize_obs": False, "n_state_bins": None},
}

_DEFAULTS_FALLBACK = {"normalize_obs": False, "n_discrete_actions": None, "discretize_obs": False, "n_state_bins": None,}


def make_env(
    env_id: str,
    render_mode: str | None = None,
    normalize_obs: bool | None = None,
    n_discrete_actions: int | None = None,
    discretize_obs: bool | None = None,
    n_state_bins: int | list[int] | None = None,
    **gym_kwargs,
) -> gym.Env:

    defaults      = _ENV_DEFAULTS.get(env_id, _DEFAULTS_FALLBACK)
    _normalize    = normalize_obs      if normalize_obs      is not None else defaults["normalize_obs"]
    _n_disc       = n_discrete_actions if n_discrete_actions is not None else defaults["n_discrete_actions"]
    _disc_obs     = discretize_obs     if discretize_obs     is not None else defaults["discretize_obs"]
    _state_bins   = n_state_bins       if n_state_bins       is not None else defaults["n_state_bins"]

    if env_id == "GridWorld-v0": # this is a custom simple grid world implementation that needs testing
        env = GridWorldEnv(render_mode=render_mode, **gym_kwargs)
    else:
        env = gym.make(env_id, render_mode=render_mode, **gym_kwargs)

    env.unwrapped.metadata = dict(env.unwrapped.metadata)

    if _n_disc:
        env = DiscreteActionWrapper(env, n_actions=_n_disc)
        env.metadata["action_type"] = "discrete_wrapped"
    else:
        env.metadata["action_type"] = env.metadata.get("action_type", "discrete_original")

    if _normalize:
        env = NormalizeObsWrapper(env)

    if _disc_obs:
        assert _state_bins is not None, (
            f"discretize_obs=True for {env_id!r} requires n_state_bins"
        )
        env = DiscretizeObsWrapper(env, n_bins=_state_bins)
        env.metadata["obs_type"] = "discretized"
    else:
        env.metadata["obs_type"] = env.metadata.get("obs_type", "continuous")

    return env


def registered_envs() -> list[str]:
    return list(_ENV_DEFAULTS.keys())
