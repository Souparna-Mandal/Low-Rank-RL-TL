"""Plots for rank / spectral analysis results."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from low_rank_rl.analysis.rank import RankMetrics
from low_rank_rl.analysis.hankel import HankelMetrics
from low_rank_rl.analysis.successor import SuccessorComparison


def plot_singular_value_spectrum(
    metrics: RankMetrics,
    log_scale: bool = True,
    title: str = "Q-matrix Singular Value Spectrum",
    save_path: str | None = None,
) -> Figure:
    sigma   = metrics.singular_values
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(sigma, "o-", markersize=4, color="steelblue", label="sigma_i")
    if log_scale:
        ax.set_yscale("log")

    threshold = sigma[0] * 1e-5
    ax.axhline(threshold, linestyle="--", color="red", alpha=0.6,
               label="rank threshold (sigma_1 x 1e-5)")
    ax.axvline(metrics.numerical_rank - 0.5, linestyle=":", color="orange", alpha=0.8,
               label=f"numerical rank = {metrics.numerical_rank}")

    ax.set_xlabel("Index i")
    ax.set_ylabel("sigma_i" + (" (log)" if log_scale else ""))
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    info = (f"stable rank = {metrics.stable_rank:.2f}  |  "
            f"effective rank = {metrics.effective_rank:.2f}  |  "
            f"spectral gap = {metrics.spectral_gap:.4f}")
    ax.text(0.02, 0.03, info, transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom", bbox=dict(boxstyle="round", alpha=0.1))

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_hosvd_spectra(
    spectra: dict[int, np.ndarray],
    dim_labels: list[str] | None = None,
    log_scale: bool = True,
    title: str = "HOSVD Mode Spectra (Value Tensor)",
    save_path: str | None = None,
) -> Figure:
    n_modes    = len(spectra)
    fig, axes  = plt.subplots(1, n_modes, figsize=(4 * n_modes, 4), sharey=False)
    if n_modes == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors

    for idx, (mode, sigma) in enumerate(sorted(spectra.items())):
        ax     = axes[idx]
        label  = dim_labels[mode] if dim_labels and mode < len(dim_labels) else f"mode {mode}"
        stable = float(np.sum(sigma ** 2) / (sigma[0] ** 2 + 1e-12))

        ax.plot(sigma, "o-", markersize=4, color=colors[idx % 10])
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(f"{label}\nstable rank={stable:.2f}")
        ax.set_xlabel("Index i")
        ax.set_ylabel("sigma_i")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_rank_vs_episode(
    history: list[dict],
    metrics: list[str] | None = None,
    title: str = "Rank Metrics vs Training Episode",
    save_path: str | None = None,
) -> Figure:
    if metrics is None:
        metrics = ["stable_rank", "effective_rank", "normalised_rank"]

    episodes  = [h["episode"] for h in history]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    labels = {
        "stable_rank":     "Stable Rank",
        "effective_rank":  "Effective Rank",
        "normalised_rank": "Numerical Rank / min(m,n)",
        "numerical_rank":  "Numerical Rank",
    }
    colors = ["steelblue", "darkorange", "seagreen"]

    for i, (metric, ax) in enumerate(zip(metrics, axes)):
        values = [h[metric] for h in history]
        ax.plot(episodes, values, "o-", color=colors[i % 3], markersize=4)
        ax.set_xlabel("Episode")
        ax.set_ylabel(labels.get(metric, metric))
        ax.set_title(labels.get(metric, metric))
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_hankel_spectrum(
    metrics: HankelMetrics,
    log_scale: bool = True,
    title: str | None = None,
    save_path: str | None = None,
) -> Figure:
    sigma = metrics.singular_values
    if title is None:
        title = f"Hankel Spectrum ({metrics.sequence_type})"

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sigma, "s-", markersize=4, color="purple", label="sigma_i")
    if log_scale:
        ax.set_yscale("log")
    ax.axvline(metrics.numerical_rank - 0.5, linestyle=":", color="orange",
               label=f"numerical rank = {metrics.numerical_rank}")
    ax.set_xlabel("Index i")
    ax.set_ylabel("sigma_i" + (" (log)" if log_scale else ""))
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    info = (f"stable rank = {metrics.stable_rank:.2f}  |  "
            f"effective rank = {metrics.effective_rank:.2f}")
    ax.text(0.02, 0.03, info, transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom", bbox=dict(boxstyle="round", alpha=0.1))

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_shift_comparison(
    comparison: SuccessorComparison,
    log_scale: bool = True,
    title: str = "Successor Matrix: Vanilla vs Shifted",
    save_path: str | None = None,
) -> Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metrics, label, color in [
        (ax1, comparison.vanilla, "Vanilla M",          "steelblue"),
        (ax2, comparison.shifted, "Shifted M - 1 mu^T", "darkorange"),
    ]:
        ax.plot(metrics.singular_values, "o-", markersize=4, color=color)
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Index i")
        ax.set_ylabel("sigma_i")
        ax.set_title(f"{label}\nstable rank={metrics.stable_rank:.2f}, "
                     f"eff rank={metrics.effective_rank:.2f}")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
