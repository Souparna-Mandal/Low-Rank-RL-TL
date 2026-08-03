import gymnasium as gym
import numpy as np
from gymnasium import ActionWrapper
from gymnasium.spaces import Discrete


class DiscretiseActionWrapper(ActionWrapper):
    """Expose a 1-D Box action space as Discrete(n_bins) with evenly spaced
    actions, so DQN-family agents can drive continuous-torque classics
    (Pendulum)."""

    def __init__(self, env: gym.Env, n_bins: int):
        super().__init__(env)
        box = env.action_space
        assert isinstance(box, gym.spaces.Box) and box.shape == (1,)
        self.values = np.linspace(box.low[0], box.high[0], n_bins, dtype=box.dtype)
        self.action_space = Discrete(n_bins)

    def action(self, act):
        return np.array([self.values[int(act)]], dtype=self.values.dtype)
