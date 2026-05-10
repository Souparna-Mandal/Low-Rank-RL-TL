"""Tests for low_rank_rl.visualization — all functions return valid figures."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # headless backend for CI
import matplotlib.pyplot as plt

from low_rank_rl.analysis.rank import _metrics_from_matrix, RankMetrics
from low_rank_rl.analysis.hankel import HankelMetrics
from low_rank_rl.analysis.successor import SuccessorComparison
from low_rank_rl.visualization.training import plot_episode_durations, plot_learning_curves
from low_rank_rl.visualization.rank_analysis import (
    plot_singular_value_spectrum,
    plot_hosvd_spectra,
    plot_rank_vs_episode,
    plot_hankel_spectrum,
    plot_shift_comparison,
)
from low_rank_rl.visualization.value_fn import plot_value_heatmap, plot_q_heatmap
from low_rank_rl.agents import DQNAgent
from low_rank_rl.envs import make_env


def _make_rank_metrics(shape=(50, 3)) -> RankMetrics:
    Q = np.random.randn(*shape)
    return _metrics_from_matrix(Q, tol=1e-5)


def _make_hankel_metrics() -> HankelMetrics:
    seq   = np.sin(np.linspace(0, 4 * np.pi, 60))
    from low_rank_rl.analysis.hankel import build_hankel_matrix
    H     = build_hankel_matrix(seq, n_rows=30)
    sigma = np.linalg.svd(H, compute_uv=False)
    return HankelMetrics(
        sequence_type="value",
        sequence_length=60,
        hankel_shape=H.shape,
        singular_values=sigma,
        numerical_rank=int(np.sum(sigma > 1e-5 * sigma[0])),
    )


class TestTrainingPlots:
    def test_episode_durations_returns_figure(self):
        durations = list(range(1, 101))
        fig = plot_episode_durations(durations)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_episode_durations_short_series(self):
        fig = plot_episode_durations([10, 20, 30])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_learning_curves_single(self):
        rewards = np.random.randn(100).tolist()
        fig = plot_learning_curves({"DQN": rewards})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_learning_curves_multi(self):
        r = np.random.randn(100).tolist()
        fig = plot_learning_curves({"DQN": r, "PPO": r})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestRankAnalysisPlots:
    def test_singular_value_spectrum(self):
        fig = plot_singular_value_spectrum(_make_rank_metrics())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_singular_value_spectrum_linear_scale(self):
        fig = plot_singular_value_spectrum(_make_rank_metrics(), log_scale=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_hosvd_spectra_single_mode(self):
        spectra = {0: np.array([3.0, 1.0, 0.1])}
        fig = plot_hosvd_spectra(spectra)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_hosvd_spectra_multi_mode(self):
        spectra = {0: np.array([3.0, 1.0]), 1: np.array([2.0, 0.5]), 2: np.array([1.0, 0.1])}
        fig = plot_hosvd_spectra(spectra, dim_labels=["θ₁", "θ₂", "ω₁"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_rank_vs_episode(self):
        history = [
            {"episode": ep, "numerical_rank": 3}
            for ep in range(0, 500, 50)
        ]
        fig = plot_rank_vs_episode(history)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_hankel_spectrum(self):
        fig = plot_hankel_spectrum(_make_hankel_metrics())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_shift_comparison(self):
        M   = np.random.rand(8, 8)
        Ms  = M - M.mean(axis=0, keepdims=True)
        cmp = SuccessorComparison(
            vanilla=_metrics_from_matrix(M,  1e-5),
            shifted=_metrics_from_matrix(Ms, 1e-5),
        )
        fig = plot_shift_comparison(cmp)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestValueFnPlots:
    def setup_method(self):
        self.env   = make_env("Acrobot-v1")
        self.agent = DQNAgent(
            self.env.observation_space.shape[0],
            self.env.action_space.n,
            hidden=16, device="cpu",
        )

    def teardown_method(self):
        self.env.close()

    def test_value_heatmap_returns_figure(self):
        fig = plot_value_heatmap(self.agent, self.env, dims=(0, 1), n_bins=10)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_q_heatmap_returns_figure(self):
        fig = plot_q_heatmap(self.agent, self.env, action=0, dims=(0, 1), n_bins=10)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_q_heatmap_different_actions(self):
        for a in range(3):
            fig = plot_q_heatmap(self.agent, self.env, action=a, dims=(0, 1), n_bins=8)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
