"""Tests for low_rank_rl.envs — registry and wrappers."""

import numpy as np
import pytest
from gymnasium import spaces

from low_rank_rl.envs import make_env, registered_envs, DiscreteActionWrapper, NormalizeObsWrapper


class TestRegisteredEnvs:
    def test_returns_list(self):
        envs = registered_envs()
        assert isinstance(envs, list)
        assert len(envs) > 0

    def test_known_envs_present(self):
        envs = registered_envs()
        assert "Acrobot-v1" in envs
        assert "MountainCarContinuous-v0" in envs


class TestMakeEnv:
    def test_acrobot_action_space_is_discrete(self):
        env = make_env("Acrobot-v1")
        assert isinstance(env.action_space, spaces.Discrete)
        env.close()

    def test_acrobot_obs_space_is_box(self):
        env = make_env("Acrobot-v1")
        assert isinstance(env.observation_space, spaces.Box)
        env.close()

    def test_acrobot_metadata_action_type(self):
        env = make_env("Acrobot-v1")
        assert env.metadata["action_type"] == "discrete_original"
        env.close()

    def test_mountaincar_wrapped_to_discrete(self):
        env = make_env("MountainCarContinuous-v0", n_discrete_actions=21)
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 21
        assert env.metadata["action_type"] == "discrete_wrapped"
        env.close()

    def test_normalize_obs_override(self):
        env = make_env("Acrobot-v1", normalize_obs=True)
        obs, _ = env.reset()
        assert obs.min() >= -1.0 - 1e-6
        assert obs.max() <= 1.0 + 1e-6
        env.close()

    def test_unknown_env_uses_safe_defaults(self):
        env = make_env("CartPole-v1")
        assert isinstance(env.action_space, spaces.Discrete)
        env.close()

    def test_env_step_returns_obs(self):
        env = make_env("Acrobot-v1")
        obs, _ = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert next_obs.shape == obs.shape
        assert isinstance(reward, float)
        env.close()


class TestDiscreteActionWrapper:
    def test_action_space_is_discrete(self):
        import gymnasium as gym
        base = gym.make("MountainCarContinuous-v0")
        env  = DiscreteActionWrapper(base, n_actions=11)
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 11
        env.close()

    def test_action_values_length(self):
        import gymnasium as gym
        base = gym.make("MountainCarContinuous-v0")
        env  = DiscreteActionWrapper(base, n_actions=11)
        vals = env.action_values
        assert len(vals) == 11
        env.close()

    def test_action_values_span_range(self):
        import gymnasium as gym
        base = gym.make("MountainCarContinuous-v0")
        env  = DiscreteActionWrapper(base, n_actions=11)
        vals = env.action_values
        low  = float(base.action_space.low[0])
        high = float(base.action_space.high[0])
        assert np.isclose(vals[0],  low)
        assert np.isclose(vals[-1], high)
        env.close()

    def test_rejects_non_box_action_space(self):
        import gymnasium as gym
        base = gym.make("CartPole-v1")
        with pytest.raises(AssertionError):
            DiscreteActionWrapper(base)
        base.close()


class TestNormalizeObsWrapper:
    def test_obs_in_unit_range(self):
        import gymnasium as gym
        base = gym.make("MountainCarContinuous-v0")
        env  = NormalizeObsWrapper(base)
        for _ in range(20):
            obs, _ = env.reset()
            assert obs.min() >= -1.0 - 1e-5
            assert obs.max() <= 1.0  + 1e-5
        env.close()

    def test_obs_space_bounds_updated(self):
        import gymnasium as gym
        base = gym.make("MountainCarContinuous-v0")
        env  = NormalizeObsWrapper(base)
        assert env.observation_space.low.min()  >= -1.0
        assert env.observation_space.high.max() <=  1.0
        env.close()

    def test_infinite_bounds_handled(self):
        import gymnasium as gym
        # Pendulum has obs bounds of [-inf, inf] for some dims
        base = gym.make("Pendulum-v1")
        env  = NormalizeObsWrapper(base)
        obs, _ = env.reset()
        assert np.all(np.isfinite(obs))
        env.close()
