from __future__ import annotations
from abc import ABC, abstractmethod
import gymnasium as gym
import pathlib
import numpy as np
import torch

class BaseAgent(ABC):
    def __init__(self, env: gym.Env):
        self.agent_env = env
    @abstractmethod
    def pi(self, state: np.ndarray) -> np.ndarray | torch.tensor :
        """Policy pi: Returns an action or distribution over actions from env.action_space given the state.

        Args:
            state ( np.ndarray| torch.tensor ): Current State the agent is observing given by the environment.
        """
        pass
    
    @abstractmethod
    def save(self, path: str | pathlib.Path) -> None:
        """Save the current Model Params, like Neural Network Parms for PPO and DQN to
        the path specified. Good to also include optimiser state which may be useful when
        we want to retrain agents or continue from crashed points. """
        raise NotImplementedError(f"{type(self).__name__} has not implemented save()")
    
    @abstractmethod
    def load(self, path: str | pathlib.Path) -> None:
        """Loads respective state dicts from the given path and restores them to the agent
        instance. """
        raise NotImplementedError(f"{type(self).__name__} has not implemented load()")


class EpsilonGreedyExplorer():
    def __init__(self, eps_start, eps_min, decay_rate):
        self.epsilon = eps_start
        self.eps_min = eps_min
        self.decay_rate = decay_rate
        
    def decay_epsilon(self):
        # We follow an exponential decay scheme
        self.epsilon = max(self.epsilon * self.decay_rate, self.eps_min)
        
    def get_epsilon(self):
        return self.epsilon

    def is_random_step(self):
        return np.random.random() < self.epsilon