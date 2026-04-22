"""HOSVD / Tucker analysis of value tensors."""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym

from low_rank_rl.agents.base import BaseAgent
from low_rank_rl.analysis.rank import _sample_states
from low_rank_rl.envs.wrappers import find_obs_discretizer


def build_value_tensor(
    agent: BaseAgent,
    env: gym.Env,
    dims: Optional[list[int]] = None,
    n_bins: Optional[int] = None,
    n_samples: int = 50_000,
) -> np.ndarray:
    obs_space = env.observation_space
    obs_dim   = obs_space.shape[0]
    if dims is None:
        dims = list(range(min(obs_dim, 4)))

    disc = find_obs_discretizer(env)

    if disc is not None and n_bins is None:
        edges        = [disc.bin_edges[d]   for d in dims]
        bins_per_dim = [disc.n_bins[d]      for d in dims]
        if len(dims) == obs_dim:
            centers    = [disc.bin_centers[d] for d in dims]
            mesh       = np.meshgrid(*centers, indexing="ij")
            all_states = np.stack([m.ravel() for m in mesh], axis=-1).astype(np.float64)
            values     = agent.value_vector(all_states)
            return values.reshape(tuple(bins_per_dim)).astype(np.float64)
    else:
        bins         = n_bins if n_bins is not None else 20
        low  = np.clip(obs_space.low.astype(np.float64),  -1e4, 1e4)
        high = np.clip(obs_space.high.astype(np.float64), -1e4, 1e4)
        edges        = [np.linspace(low[d], high[d], bins + 1) for d in dims]
        bins_per_dim = [bins] * len(dims)

    states = _sample_states(env, n_samples)
    values = agent.value_vector(states)

    shape = tuple(bins_per_dim)
    accum = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.int64)

    for i, s in enumerate(states):
        idx = tuple(
            int(np.clip(np.digitize(s[d], edges[j][1:-1]), 0, bins_per_dim[j] - 1))
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
