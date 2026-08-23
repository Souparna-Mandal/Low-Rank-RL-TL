import gymnasium as gym
import numpy as np
from gymnasium import ActionWrapper
from gymnasium.spaces import Discrete


class DiscretiseActionWrapper(ActionWrapper):
    """Expose a 1-D Box action space as Discrete(n_bins) with evenly spaced
    actions, so discrete-action agents can drive continuous-torque classics
    (Pendulum)."""

    def __init__(self, env: gym.Env, n_bins: int):
        super().__init__(env)
        box = env.action_space
        assert isinstance(box, gym.spaces.Box) and box.shape == (1,), (
            f"DiscretiseActionWrapper needs a 1-D Box action space, got {box}")
        low, high = float(box.low[0]), float(box.high[0])
        assert np.isfinite(low) and np.isfinite(high) and low < high, (
            f"DiscretiseActionWrapper needs finite low < high bounds, "
            f"got [{low}, {high}]")
        assert int(n_bins) >= 2, f"n_bins must be >= 2, got {n_bins}"
        self.values = np.linspace(low, high, int(n_bins), dtype=box.dtype)
        self.action_space = Discrete(int(n_bins))

    def action(self, act):
        return np.array([self.values[int(act)]], dtype=self.values.dtype)
