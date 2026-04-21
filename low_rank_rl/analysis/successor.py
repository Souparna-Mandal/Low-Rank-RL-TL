"""Successor-measure construction and shift analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import gymnasium as gym

from low_rank_rl.agents.base import BaseAgent
from low_rank_rl.analysis.rank import _metrics_from_matrix, RankMetrics


@dataclass
class SuccessorComparison:
    vanilla: RankMetrics
    shifted: RankMetrics

    def summary(self) -> str:
        return (
            "Successor matrix comparison\n"
            f"  Vanilla  -> {self.vanilla.summary()}\n"
            f"  Shifted  -> {self.shifted.summary()}"
        )


def build_successor_matrix(
    agent: BaseAgent,
    env: gym.Env,
    states: np.ndarray,
    gamma: float = 0.99,
    n_rollout_steps: int = 200,
    use_greedy: bool = True,
) -> np.ndarray:
    N = len(states)
    M = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        env.reset()
        current_obs = states[i].copy()
        discount    = 1.0

        for _ in range(n_rollout_steps):
            j = int(np.argmin(np.sum((states - current_obs) ** 2, axis=1)))
            M[i, j] += discount
            discount *= gamma

            action = agent.act(current_obs, training=not use_greedy)
            next_obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
            current_obs = next_obs

    return M


def shifted_successor_matrix(M: np.ndarray) -> np.ndarray:
    """M̃ = M - 1·μᵀ  where μ = column means of M."""
    mu   = M.mean(axis=0, keepdims=True)
    ones = np.ones((M.shape[0], 1))
    return M - ones @ mu


def compare_shift_rank(
    agent: BaseAgent,
    env: gym.Env,
    states: np.ndarray,
    gamma: float = 0.99,
    n_rollout_steps: int = 200,
    tol: float = 1e-5,
) -> SuccessorComparison:
    M         = build_successor_matrix(agent, env, states, gamma=gamma, n_rollout_steps=n_rollout_steps)
    M_shifted = shifted_successor_matrix(M)
    return SuccessorComparison(
        vanilla=_metrics_from_matrix(M,         tol),
        shifted=_metrics_from_matrix(M_shifted, tol),
    )


def successor_features(
    agent: BaseAgent,
    env: gym.Env,
    states: np.ndarray,
    feature_fn: Callable[[np.ndarray], np.ndarray],
    gamma: float = 0.99,
    n_rollout_steps: int = 200,
) -> np.ndarray:
    """ψ(s) = E[Σ_{t≥0} γᵗ φ(sₜ) | s₀ = s, π]."""
    d   = len(feature_fn(states[0]))
    psi = np.zeros((len(states), d), dtype=np.float64)

    for i, start in enumerate(states):
        current_obs = start.copy()
        discount    = 1.0
        env.reset()

        for _ in range(n_rollout_steps):
            psi[i]   += discount * feature_fn(current_obs)
            discount *= gamma
            action = agent.act(current_obs, training=False)
            next_obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
            current_obs = next_obs

    return psi
