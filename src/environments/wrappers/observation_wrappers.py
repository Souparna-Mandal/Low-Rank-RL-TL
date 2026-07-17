import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box


class UnderlyingStateWrapper(ObservationWrapper):
    """Expose an env's full internal `unwrapped.state` as the observation.

    For envs whose observation is a lossy encoding of the true state (Acrobot:
    a 6-dim cos/sin embedding of the 4-dim [theta1, theta2, dtheta1, dtheta2]),
    tabular PI must discretise and teleport in the native state coordinates.
    This wrapper replaces the observation with `unwrapped.state` over the finite
    `low`/`high` box, so DiscreteStateWrapper bins — and GenerativeStateWrapper
    teleports to — the same coordinates. Applied right after the generative
    wrapper and before clip/discretise.
    """

    def __init__(self, env: gym.Env, low: list, high: list):
        super().__init__(env)
        low = np.asarray(low, dtype=np.float32)
        high = np.asarray(high, dtype=np.float32)
        assert low.shape == high.shape and np.isfinite(low).all() and np.isfinite(high).all(), (
            "UnderlyingStateWrapper needs finite low/high matching the native state dim")
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs) -> np.ndarray:
        return np.asarray(self.unwrapped.state, dtype=np.float32)
