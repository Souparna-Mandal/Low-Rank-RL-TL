"""Gymnasium wrappers: continuous-to-discrete action mapping and obs normalisation."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, n_actions: int = 21):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Box), (
            "DiscreteActionWrapper requires a Box action space"
        )
        assert env.action_space.shape == (1,), (
            "DiscreteActionWrapper only supports 1-D action spaces"
        )
        self.n_actions    = n_actions
        low               = float(env.action_space.low[0])
        high              = float(env.action_space.high[0])
        self._action_map  = np.linspace(low, high, n_actions, dtype=np.float32)
        self.action_space = spaces.Discrete(n_actions)

    def action(self, action: int) -> np.ndarray:
        return np.array([self._action_map[action]], dtype=np.float32)

    @property
    def action_values(self) -> np.ndarray:
        return self._action_map.copy()


class NormalizeObsWrapper(gym.ObservationWrapper):
    _CLIP_FALLBACK = 5.0

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box), (
            "NormalizeObsWrapper requires a Box observation space"
        )
        low  = env.observation_space.low.copy().astype(np.float32)
        high = env.observation_space.high.copy().astype(np.float32)

        inf_mask       = ~np.isfinite(low) | ~np.isfinite(high)
        low[inf_mask]  = -self._CLIP_FALLBACK
        high[inf_mask] =  self._CLIP_FALLBACK

        self._low   = low
        self._range = (high - low).clip(min=1e-8)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs = np.clip(obs, self._low, self._low + self._range).astype(np.float32)
        return 2.0 * (obs - self._low) / self._range - 1.0
