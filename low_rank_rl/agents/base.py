"""Abstract base class shared by every RL agent."""

from __future__ import annotations

import abc
import pathlib

import numpy as np

class BaseAgent(abc.ABC):
    @abc.abstractmethod
    def act(self, state: np.ndarray, training: bool = True) -> int: ...

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> dict[str, float]: ...

    @abc.abstractmethod
    def q_matrix(self, states: np.ndarray) -> np.ndarray:
        """Return Q(s, a) as an (N, n_actions) array without changing any states."""

    def save(self, path: str | pathlib.Path) -> None:
        """Save the current Model Params, like Neural Network Parms for PPO and DQN to
        the path specified. Good to also include optimiser state which may be useful when
        we want to retrain agents or continue from crashed points. """
        raise NotImplementedError(f"{type(self).__name__} has not implemented save()")

    def load(self, path: str | pathlib.Path) -> None:
        """Loads respective state dicts from the given path and restores them to the agent
        instance. """
        raise NotImplementedError(f"{type(self).__name__} has not implemented load()")

    def value_vector(self, states: np.ndarray) -> np.ndarray:
        return self.q_matrix(states).max(axis=1)

    def policy_vector(self, states: np.ndarray) -> np.ndarray:
        return self.q_matrix(states).argmax(axis=1)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
