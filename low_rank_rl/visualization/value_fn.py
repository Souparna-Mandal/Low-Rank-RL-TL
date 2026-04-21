"""Value-function and Q-function 2-D slice heatmaps."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import gymnasium as gym

from low_rank_rl.agents.base import BaseAgent


def _grid_states(
    env: gym.Env,
    dims: tuple[int, int],
    n_bins: int,
    fixed_values: dict[int, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_space = env.observation_space
    low       = np.clip(obs_space.low.astype(float),  -1e4, 1e4)
    high      = np.clip(obs_space.high.astype(float), -1e4, 1e4)
    d1, d2    = dims
    xs        = np.linspace(low[d1], high[d1], n_bins)
    ys        = np.linspace(low[d2], high[d2], n_bins)

    base = (low + high) / 2.0
    if fixed_values:
        for d, v in fixed_values.items():
            base[d] = v

    grid = np.tile(base, (n_bins * n_bins, 1))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            grid[i * n_bins + j, d1] = x
            grid[i * n_bins + j, d2] = y
    return grid, xs, ys


def plot_value_heatmap(
    agent: BaseAgent,
    env: gym.Env,
    dims: tuple[int, int] = (0, 1),
    n_bins: int = 50,
    fixed_values: dict[int, float] | None = None,
    title: str = "Value Function V(s)",
    save_path: str | None = None,
) -> Figure:
    grid, xs, ys = _grid_states(env, dims, n_bins, fixed_values)
    V            = agent.value_vector(grid).reshape(n_bins, n_bins)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        V.T, origin="lower", aspect="auto",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, label="V(s)")
    ax.set_xlabel(f"dim {dims[0]}")
    ax.set_ylabel(f"dim {dims[1]}")
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_q_heatmap(
    agent: BaseAgent,
    env: gym.Env,
    action: int = 0,
    dims: tuple[int, int] = (0, 1),
    n_bins: int = 50,
    fixed_values: dict[int, float] | None = None,
    title: str | None = None,
    save_path: str | None = None,
) -> Figure:
    grid, xs, ys = _grid_states(env, dims, n_bins, fixed_values)
    Q            = agent.q_matrix(grid)[:, action].reshape(n_bins, n_bins)

    if title is None:
        title = f"Q(s, a={action})"

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        Q.T, origin="lower", aspect="auto",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        cmap="plasma",
    )
    plt.colorbar(im, ax=ax, label=f"Q(s, a={action})")
    ax.set_xlabel(f"dim {dims[0]}")
    ax.set_ylabel(f"dim {dims[1]}")
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
