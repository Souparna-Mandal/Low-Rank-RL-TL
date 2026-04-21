"""First-visit on-policy Monte Carlo control with discretised observations."""

from __future__ import annotations

import pathlib
import pickle
from collections import defaultdict
from typing import Optional

import numpy as np

from low_rank_rl.agents.base import BaseAgent


class MonteCarloAgent(BaseAgent):
    def __init__(
        self,
        n_actions: int,
        obs_low: np.ndarray,
        obs_high: np.ndarray,
        n_bins: int = 10,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_decay: float = 0.999,
        eps_min: float = 0.01,
    ):
        self.n_actions = n_actions
        self.n_bins    = n_bins
        self.gamma     = gamma
        self.epsilon   = eps_start
        self.eps_decay = eps_decay
        self.eps_min   = eps_min

        obs_low  = np.clip(np.asarray(obs_low,  dtype=np.float64), -1e4, 1e4)
        obs_high = np.clip(np.asarray(obs_high, dtype=np.float64), -1e4, 1e4)
        self._bins    = [np.linspace(obs_low[i], obs_high[i], n_bins) for i in range(len(obs_low))]
        self._obs_dim = len(obs_low)

        self.q_table: defaultdict        = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))
        self._returns_count: defaultdict = defaultdict(lambda: np.zeros(n_actions, dtype=np.int64))
        self._episode: list[tuple]       = []

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[self._discretise(state)]))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: Optional[np.ndarray] = None,
        done: bool = False,
    ) -> dict[str, float]:
        self._episode.append((self._discretise(state), action, reward))
        return {}

    def end_episode(self) -> dict[str, float]:
        G       = 0.0
        visited: set[tuple] = set()
        updates = []

        for ds, action, reward in reversed(self._episode):
            G = reward + self.gamma * G
            sa = (ds, action)
            if sa not in visited:
                visited.add(sa)
                self._returns_count[ds][action] += 1
                n = self._returns_count[ds][action]
                delta = G - self.q_table[ds][action]
                self.q_table[ds][action] += delta / n
                updates.append(abs(delta))

        self._episode = []
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        mean_update = float(np.mean(updates)) if updates else 0.0
        return {"mean_update": mean_update, "episode_len": len(visited), "epsilon": self.epsilon}

    def q_matrix(self, states: np.ndarray) -> np.ndarray:
        out = np.zeros((len(states), self.n_actions), dtype=np.float64)
        for i, s in enumerate(states):
            out[i] = self.q_table[self._discretise(s)].copy()
        return out

    def save(self, path: str | pathlib.Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "q_table": dict(self.q_table),
                "returns_count": dict(self._returns_count),
                "epsilon": self.epsilon,
            }, f)

    def load(self, path: str | pathlib.Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table.update(data["q_table"])
        self._returns_count.update(data["returns_count"])
        self.epsilon = data["epsilon"]

    def _discretise(self, state: np.ndarray) -> tuple:
        return tuple(int(np.digitize(state[i], self._bins[i])) for i in range(self._obs_dim))

    def __repr__(self) -> str:
        return f"MonteCarloAgent(n_actions={self.n_actions}, eps={self.epsilon:.3f})"
