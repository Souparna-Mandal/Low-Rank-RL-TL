import numpy as np
import gymnasium as gym
from gymnasium import Wrapper


class GenerativeStateWrapper(Wrapper):
    """Generative-model access for classic-control envs that keep their full
    state in `unwrapped.state` (CartPole, Acrobot, MountainCar, ...).

    `teleport(state)` resets the env (fresh TimeLimit / termination
    bookkeeping), then overwrites the underlying state so the next `step`
    continues from it. The returned obs is the state itself: observation
    wrappers applied outside this one are not re-run, so teleport targets
    must already be valid observations (e.g. bin centres within clip bounds).
    """

    def teleport(self, state) -> np.ndarray:
        self.env.reset()
        state = np.asarray(state, dtype=np.float64).copy()
        self.unwrapped.state = state
        return state