import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box


class OneHotObservationWrapper(ObservationWrapper):
    """Discrete(n) observations as one-hot float32 vectors, so MLP agents can
    consume tabular envs (CliffWalking, FrozenLake)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.n = int(env.observation_space.n)
        self.observation_space = Box(0.0, 1.0, (self.n,), dtype=np.float32)

    def observation(self, obs):
        vec = np.zeros(self.n, dtype=np.float32)
        vec[int(obs)] = 1.0
        return vec
