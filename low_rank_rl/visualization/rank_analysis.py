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
    log_scale: bool = False,
    title: str = "Q-matrix Singular Value Spectrum",
    save_path: str | None = None,
) -> Figure:
    sigma   = metrics.singular_values
    idx     = np.arange(len(sigma))
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(idx, sigma, color="steelblue", edgecolor="black", linewidth=0.5)
    if log_scale:
        ax.set_yscale("log")

    ax.axvline(metrics.numerical_rank - 0.5, linestyle=":", color="orange", alpha=0.8,
               label=f"numerical rank = {metrics.numerical_rank}")

    ax.set_xlabel("Index i")
    ax.set_ylabel("sigma_i" + (" (log)" if log_scale else ""))
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_hosvd_spectra(
    spectra: dict[int, np.ndarray],
    dim_labels: list[str] | None = None,
    log_scale: bool = False,
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
    title: str = "Numerical Rank vs Training Episode",
    save_path: str | None = None,
) -> Figure:
    if metrics is None:
        metrics = ["numerical_rank"]

    episodes  = [h["episode"] for h in history]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    labels = {"numerical_rank": "Numerical Rank"}
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
    log_scale: bool = False,
    title: str | None = None,
    save_path: str | None = None,
) -> Figure:
    sigma = metrics.singular_values
    m, n  = metrics.hankel_shape
    if title is None:
        title = (f"Hankel Spectrum ({metrics.sequence_type})  "
                 f"{m}x{n}  seq_len={metrics.sequence_length}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(sigma)), sigma, color="purple", edgecolor="black", linewidth=0.5)
    if log_scale:
        ax.set_yscale("log")
    ax.axvline(metrics.numerical_rank - 0.5, linestyle=":", color="orange",
               label=f"numerical rank = {metrics.numerical_rank}/{min(m, n)}")
    ax.set_xlabel("Index i")
    ax.set_ylabel("sigma_i" + (" (log)" if log_scale else ""))
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_hankel_spectra_over_training(
    history: list[dict],
    sequence_type: str = "value",
    log_scale: bool = False,
    title: str | None = None,
    save_path: str | None = None,
) -> Figure:
    """Overlay Hankel singular-value spectra captured at each checkpoint.

    ``history`` is ``TrainingLog.rank_history``; entries that include a
    ``"hankel"`` sub-dict (populated when ``analysis.hankel_sequence_types``
    is set in the config) are plotted, colour-coded by episode.
    """
    entries = [h for h in history
               if "hankel" in h and sequence_type in h["hankel"]]
    if not entries:
        raise ValueError(
            f"No checkpoints with Hankel data for sequence_type={sequence_type!r}. "
            f"Set analysis.hankel_sequence_types in the config to include it."
        )

    fig, (ax_spec, ax_rank) = plt.subplots(1, 2, figsize=(13, 4.5))
    cmap = plt.cm.viridis
    n    = len(entries)

    for i, h in enumerate(entries):
        hm    = h["hankel"][sequence_type]
        sigma = hm.singular_values
        color = cmap(i / max(1, n - 1))
        ax_spec.plot(sigma, "-", color=color, linewidth=1.2,
                     label=f"ep {h['episode']}  (rank={hm.numerical_rank})")

    if log_scale:
        ax_spec.set_yscale("log")
    ax_spec.set_xlabel("Index i")
    ax_spec.set_ylabel("sigma_i" + (" (log)" if log_scale else ""))
    ax_spec.set_title(f"Hankel spectra ({sequence_type}) across training")
    ax_spec.grid(True, alpha=0.3)
    ax_spec.legend(fontsize=7, ncol=max(1, n // 8), loc="upper right")

    eps   = [h["episode"] for h in entries]
    ranks = [h["hankel"][sequence_type].numerical_rank for h in entries]
    cap   = min(entries[-1]["hankel"][sequence_type].hankel_shape)
    ax_rank.plot(eps, ranks, "o-", color="purple", markersize=5)
    ax_rank.axhline(cap, linestyle=":", color="grey", alpha=0.6,
                    label=f"max rank = {cap}")
    ax_rank.set_xlabel("Episode")
    ax_rank.set_ylabel("Hankel numerical rank")
    ax_rank.set_title(f"Hankel rank vs episode ({sequence_type})")
    ax_rank.legend(fontsize=9)
    ax_rank.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_shift_comparison(
    comparison: SuccessorComparison,
    log_scale: bool = False,
    title: str = "Successor Matrix: Vanilla vs Shifted",
    save_path: str | None = None,
) -> Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metrics, label, color in [
        (ax1, comparison.vanilla, "Vanilla M",          "steelblue"),
        (ax2, comparison.shifted, "Shifted M - 1 mu^T", "darkorange"),
    ]:
        sigma = metrics.singular_values
        ax.bar(np.arange(len(sigma)), sigma, color=color, edgecolor="black", linewidth=0.5)
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Index i")
        ax.set_ylabel("sigma_i")
        ax.set_title(f"{label}\nnumerical rank = {metrics.numerical_rank}")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
