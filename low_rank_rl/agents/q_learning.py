"""Tabular off-policy Q-learning with discretised observations."""

from __future__ import annotations

import pathlib
import pickle
from collections import defaultdict

import numpy as np

from low_rank_rl.agents.base import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(
        self,
        n_actions: int,
        obs_low: np.ndarray,
        obs_high: np.ndarray,
        n_bins: int = 10,
        lr: float = 0.1,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_decay: float = 0.999,
        eps_min: float = 0.001,
    ):
        self.n_actions = n_actions
        self.n_bins    = n_bins
        self.lr        = lr
        self.gamma     = gamma
        self.epsilon   = eps_start
        self.eps_decay = eps_decay
        self.eps_min   = eps_min

        obs_low  = np.clip(np.asarray(obs_low,  dtype=np.float64), -1e4, 1e4)
        obs_high = np.clip(np.asarray(obs_high, dtype=np.float64), -1e4, 1e4)
        self._bins    = [np.linspace(obs_low[i], obs_high[i], n_bins) for i in range(len(obs_low))]
        self._obs_dim = len(obs_low)
        self.q_table: defaultdict = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[self._discretise(state)]))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> dict[str, float]:
        s          = self._discretise(state)
        s_next     = self._discretise(next_state)
        best_next  = 0.0 if done else float(np.max(self.q_table[s_next]))
        td_target  = reward + self.gamma * best_next
        td_error   = td_target - self.q_table[s][action]
        self.q_table[s][action] += self.lr * td_error
        return {"td_error": abs(td_error), "epsilon": self.epsilon}

    def q_matrix(self, states: np.ndarray) -> np.ndarray:
        out = np.zeros((len(states), self.n_actions), dtype=np.float64)
        for i, s in enumerate(states):
            out[i] = self.q_table[self._discretise(s)].copy()
        return out

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def save(self, path: str | pathlib.Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"q_table": dict(self.q_table), "epsilon": self.epsilon,
                 "n_bins": self.n_bins, "bins": self._bins},
                f,
            )

    def load(self, path: str | pathlib.Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table.update(data["q_table"])
        self.epsilon = data["epsilon"]

    def _discretise(self, state: np.ndarray) -> tuple:
        return tuple(int(np.digitize(state[i], self._bins[i])) for i in range(self._obs_dim))

    def __repr__(self) -> str:
        return (
            f"QLearningAgent(n_actions={self.n_actions}, n_bins={self.n_bins}, "
            f"eps={self.epsilon:.3f}, states_seen={len(self.q_table)})"
        )
