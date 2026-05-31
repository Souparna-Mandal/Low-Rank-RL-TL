from __future__ import annotations
from abc import ABC, abstractmethod
import pathlib
import numpy as np
import torch

class BaseAgent(ABC):
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
