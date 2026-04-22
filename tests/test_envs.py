"""Tests for low_rank_rl.envs — registry and wrappers."""

import numpy as np
import pytest
from gymnasium import spaces

from low_rank_rl.envs import (
    make_env,
    registered_envs,
    GridWorldEnv,
    DiscreteActionWrapper,
    DiscretizeObsWrapper,
    NormalizeObsWrapper,
    find_obs_discretizer,
)


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


class TestGridWorldEnv:
    def test_reset_returns_start(self):
        env = GridWorldEnv(size=(4, 4), start=(0, 0), goal=(3, 3))
        obs, _ = env.reset()
        assert tuple(obs.astype(int)) == (0, 0)

    def test_step_moves_correctly(self):
        env = GridWorldEnv(size=(4, 4), start=(1, 1), goal=(3, 3))
        env.reset()
        obs, _, _, _, _ = env.step(1)  # right
        assert tuple(obs.astype(int)) == (1, 2)

    def test_wall_blocks_move(self):
        env = GridWorldEnv(size=(3, 3), start=(0, 0), goal=(2, 2), walls=((0, 1),))
        env.reset()
        obs, _, _, _, _ = env.step(1)  # right into wall
        assert tuple(obs.astype(int)) == (0, 0)

    def test_boundary_blocks_move(self):
        env = GridWorldEnv(size=(3, 3), start=(0, 0), goal=(2, 2))
        env.reset()
        obs, _, _, _, _ = env.step(0)  # up off-grid
        assert tuple(obs.astype(int)) == (0, 0)

    def test_goal_terminates_with_zero_reward(self):
        env = GridWorldEnv(size=(2, 2), start=(0, 0), goal=(0, 1))
        env.reset()
        obs, reward, terminated, _, _ = env.step(1)
        assert terminated
        assert reward == 0.0

    def test_step_reward_is_minus_one(self):
        env = GridWorldEnv(size=(3, 3), start=(0, 0), goal=(2, 2))
        env.reset()
        _, reward, _, _, _ = env.step(1)
        assert reward == -1.0

    def test_truncation_at_max_steps(self):
        env = GridWorldEnv(size=(3, 3), start=(0, 0), goal=(2, 2), max_steps=3)
        env.reset()
        for _ in range(2):
            _, _, _, trunc, _ = env.step(3)  # left into wall, stays in place
            assert not trunc
        _, _, _, trunc, _ = env.step(3)
        assert trunc

    def test_state_grid_shape(self):
        env = GridWorldEnv(size=(4, 5), start=(0, 0), goal=(3, 4))
        grid = env.state_grid()
        assert grid.shape == (20, 2)

    def test_make_env_returns_gridworld(self):
        env = make_env("GridWorld-v0")
        assert env.action_space.n == 4
        obs, _ = env.reset()
        assert obs.shape == (2,)
        env.close()


class TestDiscretizeObsWrapper:
    def test_observation_is_a_bin_center(self):
        import gymnasium as gym
        base = gym.make("MountainCarContinuous-v0")
        env  = DiscretizeObsWrapper(base, n_bins=[10, 10])
        obs, _ = env.reset(seed=0)
        centers_0 = env.bin_centers[0]
        centers_1 = env.bin_centers[1]
        assert np.any(np.isclose(obs[0], centers_0))
        assert np.any(np.isclose(obs[1], centers_1))
        env.close()

    def test_obs_to_index_bounds(self):
        import gymnasium as gym
        base = gym.make("MountainCarContinuous-v0")
        env  = DiscretizeObsWrapper(base, n_bins=[10, 10])
        env.reset()
        low  = env.observation_space.low
        high = env.observation_space.high
        assert env.obs_to_index(low)       == (0, 0)
        assert env.obs_to_index(high)      == (9, 9)
        env.close()

    def test_clips_out_of_range(self):
        import gymnasium as gym
        base = gym.make("MountainCarContinuous-v0")
        env  = DiscretizeObsWrapper(base, n_bins=[10, 10])
        very_high = env.observation_space.high + 100.0
        very_low  = env.observation_space.low  - 100.0
        assert env.obs_to_index(very_high) == (9, 9)
        assert env.obs_to_index(very_low)  == (0, 0)
        env.close()

    def test_bin_edges_length(self):
        import gymnasium as gym
        base = gym.make("MountainCarContinuous-v0")
        env  = DiscretizeObsWrapper(base, n_bins=[7, 12])
        assert len(env.bin_edges[0]) == 8
        assert len(env.bin_edges[1]) == 13
        env.close()

    def test_infinite_bounds_handled(self):
        import gymnasium as gym
        base = gym.make("Pendulum-v1")
        env  = DiscretizeObsWrapper(base, n_bins=[11, 11, 11])
        obs, _ = env.reset()
        assert np.all(np.isfinite(obs))
        env.close()

    def test_scalar_n_bins_broadcasts(self):
        import gymnasium as gym
        base = gym.make("MountainCarContinuous-v0")
        env  = DiscretizeObsWrapper(base, n_bins=15)
        assert env.n_bins == [15, 15]
        env.close()

    def test_make_env_applies_discretizer_by_default(self):
        env = make_env("MountainCarContinuous-v0")
        assert env.metadata.get("obs_type") == "discretized"
        env.close()

    def test_make_env_discretize_off_by_override(self):
        env = make_env("MountainCarContinuous-v0", discretize_obs=False)
        assert env.metadata.get("obs_type") != "discretized"
        env.close()

    def test_discretizer_preserves_obs_shape(self):
        env = make_env("Pendulum-v1")
        obs, _ = env.reset()
        assert obs.shape == (3,)
        env.close()

    def test_acrobot_obs_is_6d(self):
        env = make_env("Acrobot-v1")
        obs, _ = env.reset()
        assert obs.shape == (6,)
        env.close()


class TestFindObsDiscretizer:
    def test_finds_wrapped_discretizer(self):
        env = make_env("MountainCarContinuous-v0")
        found = find_obs_discretizer(env)
        assert isinstance(found, DiscretizeObsWrapper)
        env.close()

    def test_returns_none_when_absent(self):
        env = make_env("MountainCarContinuous-v0", discretize_obs=False)
        assert find_obs_discretizer(env) is None
        env.close()
