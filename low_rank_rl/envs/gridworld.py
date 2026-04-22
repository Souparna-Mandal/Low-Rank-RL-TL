"""Tabular grid world with deterministic-or-slippery transitions.

Obs is the (row, col) of the agent as float32. Actions are 0=up, 1=right,
2=down, 3=left. Stepping into a wall or off the grid keeps the agent in place.
Reward is -1 per step and 0 on reaching the goal; episodes truncate at
``max_steps``.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "action_type": "discrete_original"}

    _DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def __init__(
        self,
        size: tuple[int, int] = (5, 5),
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] = (4, 4),
        walls: tuple[tuple[int, int], ...] = (),
        slip_prob: float = 0.0,
        max_steps: int = 100,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.rows, self.cols = int(size[0]), int(size[1])
        self.start           = tuple(start)
        self.goal            = tuple(goal)
        self.walls           = {tuple(w) for w in walls}
        self.slip_prob       = float(slip_prob)
        self.max_steps       = int(max_steps)
        self.render_mode     = render_mode

        assert 0 <= self.start[0] < self.rows and 0 <= self.start[1] < self.cols
        assert 0 <= self.goal[0]  < self.rows and 0 <= self.goal[1]  < self.cols
        assert self.start not in self.walls and self.goal not in self.walls

        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([self.rows - 1, self.cols - 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        self._pos: tuple[int, int] = self.start
        self._t: int               = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._pos = self.start
        self._t   = 0
        return self._obs(), {}

    def step(self, action: int):
        if self.slip_prob > 0.0 and self.np_random.random() < self.slip_prob:
            action = int(self.np_random.integers(4))
        dr, dc = self._DELTAS[int(action)]
        nr, nc = self._pos[0] + dr, self._pos[1] + dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in self.walls:
            self._pos = (nr, nc)

        self._t   += 1
        terminated = self._pos == self.goal
        truncated  = self._t >= self.max_steps
        reward     = 0.0 if terminated else -1.0
        return self._obs(), reward, terminated, truncated, {}

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        rows = []
        for r in range(self.rows):
            line = []
            for c in range(self.cols):
                if (r, c) == self._pos:    line.append("A")
                elif (r, c) == self.goal:  line.append("G")
                elif (r, c) in self.walls: line.append("#")
                else:                      line.append(".")
            rows.append("".join(line))
        return "\n".join(rows)

    def _obs(self) -> np.ndarray:
        return np.array(self._pos, dtype=np.float32)

    @property
    def n_states(self) -> int:
        return self.rows * self.cols

    def state_grid(self) -> np.ndarray:
        r, c = np.meshgrid(np.arange(self.rows), np.arange(self.cols), indexing="ij")
        return np.stack([r.ravel(), c.ravel()], axis=-1).astype(np.float32)
