"""Gymnasium environment factory with sane wrapper defaults."""

import gymnasium as gym

from low_rank_rl.envs.wrappers import DiscreteActionWrapper, NormalizeObsWrapper

_ENV_DEFAULTS: dict[str, dict] = {
    "Acrobot-v1":               {"normalize_obs": False, "n_discrete_actions": None},
    "CartPole-v1":              {"normalize_obs": False, "n_discrete_actions": None},
    "LunarLander-v2":           {"normalize_obs": False, "n_discrete_actions": None},
    "MountainCarContinuous-v0": {"normalize_obs": True,  "n_discrete_actions": 21},
    "Pendulum-v1":              {"normalize_obs": True,  "n_discrete_actions": 15},
    "HalfCheetah-v4":           {"normalize_obs": True,  "n_discrete_actions": 11},
    "Hopper-v4":                {"normalize_obs": True,  "n_discrete_actions": 11},
    "Ant-v4":                   {"normalize_obs": True,  "n_discrete_actions": 11},
}


def make_env(
    env_id: str,
    render_mode: str | None = None,
    normalize_obs: bool | None = None,
    n_discrete_actions: int | None = None,
    **gym_kwargs,
) -> gym.Env:
    defaults   = _ENV_DEFAULTS.get(env_id, {"normalize_obs": False, "n_discrete_actions": None})
    _normalize = normalize_obs      if normalize_obs      is not None else defaults["normalize_obs"]
    _n_disc    = n_discrete_actions if n_discrete_actions is not None else defaults["n_discrete_actions"]

    env = gym.make(env_id, render_mode=render_mode, **gym_kwargs)

    if _n_disc:
        env = DiscreteActionWrapper(env, n_actions=_n_disc)
        env.metadata["action_type"] = "discrete_wrapped"
    else:
        env.metadata["action_type"] = "discrete_original"

    if _normalize:
        env = NormalizeObsWrapper(env)

    return env


def registered_envs() -> list[str]:
    return list(_ENV_DEFAULTS.keys())
