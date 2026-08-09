import numpy as np
import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.spaces import Box


class DiscreteStateWrapper(Wrapper):
    """Discretisation service over a finite Box observation space.

    Observations pass through unchanged (so tensor-based consumers keep
    working); the wrapper owns the bin geometry and exposes `discretise` for
    tabular agents. Applied last in make_environment, after clip/normalise,
    so the bounds it bins over are the limited ones.
    """

    def __init__(self, env: gym.Env, state_bins: list):
        super().__init__(env)
        space = env.observation_space
        assert isinstance(space, Box), "DiscreteStateWrapper requires a Box observation space"
        assert space.shape == (len(state_bins),), "one bin count per observation dim"
        self.low = space.low.astype(np.float64)
        self.high = space.high.astype(np.float64)
        assert np.isfinite(self.low).all() and np.isfinite(self.high).all(), (
            "observation bounds must be finite — clip/normalise the state space first")

        self.state_bins = np.asarray(state_bins, dtype=np.int64)
        self.n_states = int(np.prod(self.state_bins))
        widths = (self.high - self.low) / self.state_bins
        per_dim = [self.low[d] + (np.arange(b) + 0.5) * widths[d]
                   for d, b in enumerate(self.state_bins)]
        # C-order grid: row i of bin_centres is the state with flat index i.
        self.bin_centres = np.stack(np.meshgrid(*per_dim, indexing="ij"),
                                    axis=-1).reshape(self.n_states, -1)

    def discretise(self, obs) -> int:
        """Flat bin index of an observation; out-of-bounds values clip to edge bins."""
        ratios = (np.asarray(obs, dtype=np.float64) - self.low) / (self.high - self.low)
        idxs = np.clip((ratios * self.state_bins).astype(np.int64), 0, self.state_bins - 1)
        return int(np.ravel_multi_index(idxs, self.state_bins))