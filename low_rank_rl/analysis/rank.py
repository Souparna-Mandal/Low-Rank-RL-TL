"""Rank metrics for Q-matrices (numerical, stable, effective)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import gymnasium as gym

from low_rank_rl.agents.base import BaseAgent
from low_rank_rl.envs.wrappers import find_obs_discretizer


@dataclass
class RankMetrics:
    matrix_shape:              tuple[int, int]
    singular_values:           np.ndarray
    numerical_rank:            int
    stable_rank:               float
    effective_rank:            float
    spectral_gap:              float
    normalised_numerical_rank: float

    def summary(self) -> str:
        m, n = self.matrix_shape
        return (
            f"Q-matrix {m}x{n}  |  "
            f"numerical rank = {self.numerical_rank}/{min(m,n)} "
            f"({self.normalised_numerical_rank:.2%})  |  "
            f"stable rank = {self.stable_rank:.2f}  |  "
            f"effective rank = {self.effective_rank:.2f}  |  "
            f"spectral gap = {self.spectral_gap:.4f}"
        )


def compute_rank_metrics(
    agent: BaseAgent,
    states: np.ndarray,
    tol: float = 1e-5,
) -> RankMetrics:
    return _metrics_from_matrix(agent.q_matrix(states), tol)


def compute_rank_metrics_from_matrix(Q: np.ndarray, tol: float = 1e-5) -> RankMetrics:
    return _metrics_from_matrix(Q, tol)


def sample_states(env: gym.Env, n: int) -> np.ndarray:
    grid = canonical_states(env)
    if grid is not None:
        return grid
    return _sample_states(env, n)


def canonical_states(env: gym.Env) -> np.ndarray | None:
    disc = find_obs_discretizer(env)
    if disc is None:
        return None
    centers = disc.bin_centers
    mesh    = np.meshgrid(*centers, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=-1).astype(np.float64)


def canonical_subsample(env: gym.Env, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Random subset of the canonical grid, or MC-sampled states if env is not discretised."""
    grid = canonical_states(env)
    if grid is None:
        return _sample_states(env, n)
    if len(grid) <= n:
        return grid
    rng = rng or np.random.default_rng(0)
    return grid[rng.choice(len(grid), size=n, replace=False)]


def _metrics_from_matrix(Q: np.ndarray, tol: float) -> RankMetrics:
    sigma = np.linalg.svd(Q, compute_uv=False)

    threshold = tol * sigma[0] if sigma[0] > 0 else tol
    num_rank  = int(np.sum(sigma > threshold))
    stable    = float(np.sum(sigma ** 2) / (sigma[0] ** 2 + 1e-12))

    p = sigma ** 2 / (sigma ** 2).sum()
    p = p[p > 1e-12]
    eff_rank = float(np.exp(-np.sum(p * np.log(p))))

    gap = float(sigma[0] - sigma[1]) if len(sigma) > 1 else float(sigma[0])

    return RankMetrics(
        matrix_shape=Q.shape,
        singular_values=sigma,
        numerical_rank=num_rank,
        stable_rank=stable,
        effective_rank=eff_rank,
        spectral_gap=gap,
        normalised_numerical_rank=num_rank / min(Q.shape),
    )


def _sample_states(env: gym.Env, n: int) -> np.ndarray:
    states = []
    obs, _ = env.reset()
    while len(states) < n:
        states.append(obs.copy())
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()
    return np.array(states[:n], dtype=np.float64)
