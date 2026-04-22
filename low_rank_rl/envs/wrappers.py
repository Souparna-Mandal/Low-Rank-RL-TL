"""Gymnasium wrappers: continuous-to-discrete action mapping and normalisation of observations."""

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


class DiscretizeObsWrapper(gym.ObservationWrapper):
    """Snap each observation dimension to the centre of its uniform bin.

    Training and analysis then share a common finite state grid.
    Observations keep their float dtype and shape (bin centres, not indices)
    so neural-net agents still train, while tabular agents and
    tensor/successor analysis can index the grid via ``obs_to_index``.
    """

    _CLIP_FALLBACK = 5.0

    def __init__(self, env: gym.Env, n_bins: int | list[int] | tuple[int, ...] = 21):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box), (
            "DiscretizeObsWrapper requires a Box observation space like something continuous between -20 to 5 for example"
        )
        assert len(env.observation_space.shape) == 1, (
            "DiscretizeObsWrapper only supports 1-D observation spaces, i.e. it has to be a flat vector"
        )
        d = env.observation_space.shape[0] #dims

        if isinstance(n_bins, int):
            n_bins = [n_bins] * d
        n_bins = [int(b) for b in n_bins]
        assert len(n_bins) == d and all(b >= 2 for b in n_bins), ("n_bins must have one entry per dim and each >= 2")
        self.n_bins = n_bins

        low  = env.observation_space.low.astype(np.float32).copy()
        high = env.observation_space.high.astype(np.float32).copy()
        inf_mask       = ~np.isfinite(low) | ~np.isfinite(high)
        low[inf_mask]  = -self._CLIP_FALLBACK
        high[inf_mask] =  self._CLIP_FALLBACK
        self._low  = low
        self._high = high

        self._edges   = [np.linspace(low[i], high[i], b + 1, dtype=np.float32) for i, b in enumerate(n_bins)]
        self._centers = [0.5 * (e[:-1] + e[1:]) for e in self._edges]

        self.observation_space = spaces.Box(low=low, high=high, shape=env.observation_space.shape, dtype=np.float32,)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        idx = self.obs_to_index(obs)
        return np.array([self._centers[i][idx[i]] for i in range(len(idx))],dtype=np.float32,)

    def obs_to_index(self, obs: np.ndarray) -> tuple[int, ...]:
        obs = np.clip(np.asarray(obs, dtype=np.float32), self._low, self._high)
        return tuple(
            min(int(np.digitize(obs[i], self._edges[i][1:-1])), self.n_bins[i] - 1)
            for i in range(len(self.n_bins))
        )

    @property
    def bin_edges(self) -> list[np.ndarray]:
        return [e.copy() for e in self._edges]

    @property
    def bin_centers(self) -> list[np.ndarray]:
        return [c.copy() for c in self._centers]


def find_obs_discretizer(env: gym.Env) -> DiscretizeObsWrapper | None:
    cur = env
    while cur is not None:
        if isinstance(cur, DiscretizeObsWrapper):
            return cur
        cur = getattr(cur, "env", None)
    return None
