"""Plots for training progress: episode durations and reward curves."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_episode_durations(
    durations: list[int] | np.ndarray,
    window: int = 100,
    title: str = "Episode Duration",
    save_path: str | None = None,
) -> Figure:
    durations = np.asarray(durations, dtype=float)
    fig, ax   = plt.subplots(figsize=(10, 4))

    ax.plot(durations, alpha=0.35, color="steelblue", label="Duration")
    if len(durations) >= window:
        rolling = np.convolve(durations, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(durations)), rolling,
                color="steelblue", linewidth=2, label=f"{window}-ep mean")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_episode_rewards(
    rewards: list[float] | np.ndarray,
    window: int = 20,
    title: str = "Reward vs Episode",
    save_path: str | None = None,
) -> Figure:
    rewards = np.asarray(rewards, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(rewards, alpha=0.35, color="crimson", label="Reward")
    if len(rewards) >= window:
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), rolling,
                color="crimson", linewidth=2, label=f"{window}-ep mean")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_learning_curves(
    curves: dict[str, list[float] | np.ndarray],
    window: int = 50,
    title: str = "Learning Curves",
    ylabel: str = "Total Reward",
    save_path: str | None = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 5))

    for label, rewards in curves.items():
        rewards = np.asarray(rewards, dtype=float)
        ax.plot(rewards, alpha=0.25)
        if len(rewards) >= window:
            rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(rewards)), rolling, linewidth=2, label=label)
        else:
            ax.lines[-1].set_label(label)

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
