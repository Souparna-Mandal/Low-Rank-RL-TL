"""Tabular off-policy Q-learning. Requires an env wrapped with DiscretizeObsWrapper."""

from __future__ import annotations

import pathlib
import pickle
from collections import defaultdict

import gymnasium as gym
import numpy as np

from low_rank_rl.agents.base import BaseAgent
from low_rank_rl.envs.wrappers import find_obs_discretizer


class QLearningAgent(BaseAgent):
    def __init__(
        self,
        env: gym.Env,
        n_actions: int,
        lr: float = 0.1,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_decay: float = 0.999,
        eps_min: float = 0.001,
    ):
        disc = find_obs_discretizer(env)
        if disc is None:
            raise ValueError(
                "QLearningAgent requires an environment wrapped with DiscretizeObsWrapper; "
                "build it via make_env(..., discretize_obs=True)."
            )
        self._obs_to_index = disc.obs_to_index
        self.n_bins        = list(disc.n_bins)

        self.n_actions = n_actions
        self.lr        = lr
        self.gamma     = gamma
        self.epsilon   = eps_start
        self.eps_decay = eps_decay
        self.eps_min   = eps_min

        self.q_table: defaultdict = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[self._obs_to_index(state)]))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> dict[str, float]:
        s          = self._obs_to_index(state)
        s_next     = self._obs_to_index(next_state)
        best_next  = 0.0 if done else float(np.max(self.q_table[s_next]))
        td_target  = reward + self.gamma * best_next
        td_error   = td_target - self.q_table[s][action]
        self.q_table[s][action] += self.lr * td_error
        return {"td_error": abs(td_error), "epsilon": self.epsilon}

    def q_matrix(self, states: np.ndarray) -> np.ndarray:
        out = np.zeros((len(states), self.n_actions), dtype=np.float64)
        for i, s in enumerate(states):
            out[i] = self.q_table[self._obs_to_index(s)].copy()
        return out

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def save(self, path: str | pathlib.Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"q_table": dict(self.q_table), "epsilon": self.epsilon}, f)

    def load(self, path: str | pathlib.Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table.update(data["q_table"])
        self.epsilon = data["epsilon"]

    def __repr__(self) -> str:
        return (
            f"QLearningAgent(n_actions={self.n_actions}, n_bins={self.n_bins}, "
            f"eps={self.epsilon:.3f}, states_seen={len(self.q_table)})"
        )
