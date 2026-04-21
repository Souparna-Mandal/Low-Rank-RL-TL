"""HOSVD / Tucker analysis of value tensors."""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym

from low_rank_rl.agents.base import BaseAgent
from low_rank_rl.analysis.rank import _sample_states


def build_value_tensor(
    agent: BaseAgent,
    env: gym.Env,
    dims: Optional[list[int]] = None,
    n_bins: int = 20,
    n_samples: int = 50_000,
) -> np.ndarray:
    obs_space = env.observation_space
    obs_dim   = obs_space.shape[0]
    if dims is None:
        dims = list(range(min(obs_dim, 4)))

    low  = np.clip(obs_space.low.astype(np.float64),  -1e4, 1e4)
    high = np.clip(obs_space.high.astype(np.float64), -1e4, 1e4)
    edges = [np.linspace(low[d], high[d], n_bins + 1) for d in dims]

    states = _sample_states(env, n_samples)
    values = agent.value_vector(states)

    shape = tuple(n_bins for _ in dims)
    accum = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.int64)

    for i, s in enumerate(states):
        idx = tuple(
            int(np.clip(np.digitize(s[d], edges[j][1:-1]), 0, n_bins - 1))
            for j, d in enumerate(dims)
        )
        accum[idx] += values[i]
        count[idx] += 1

    fill_value = float(np.mean(values))
    return np.where(count > 0, accum / np.maximum(count, 1), fill_value)


def hosvd_spectra(tensor: np.ndarray) -> dict[int, np.ndarray]:
    return {
        mode: np.linalg.svd(_mode_unfold(tensor, mode), compute_uv=False)
        for mode in range(tensor.ndim)
    }


def hosvd_stable_ranks(tensor: np.ndarray) -> dict[int, float]:
    return {
        mode: float(np.sum(sigma ** 2) / (sigma[0] ** 2 + 1e-12))
        for mode, sigma in hosvd_spectra(tensor).items()
    }


def tucker_reconstruction_error(
    tensor: np.ndarray,
    ranks: list[int],
) -> tuple[np.ndarray, float]:
    try:
        import tensorly as tl
        from tensorly.decomposition import tucker
    except ImportError as e:
        raise ImportError("tucker_reconstruction_error requires tensorly") from e

    tl.set_backend("numpy")
    core, factors = tucker(tensor, rank=ranks)
    reconstructed = tl.tucker_to_tensor((core, factors))
    rel_error = float(
        np.linalg.norm(tensor - reconstructed) / (np.linalg.norm(tensor) + 1e-12)
    )
    return reconstructed, rel_error


def _mode_unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))
