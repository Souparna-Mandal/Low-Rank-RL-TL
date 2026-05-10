"""Tests for low_rank_rl.analysis — rank, tensor, Hankel, successor."""

import numpy as np
import pytest

from low_rank_rl.envs import make_env
from low_rank_rl.agents import DQNAgent, QLearningAgent
from low_rank_rl.analysis.rank import (
    RankMetrics, compute_rank_metrics, compute_rank_metrics_from_matrix,
    sample_states, canonical_states, canonical_subsample, _metrics_from_matrix,
)
from low_rank_rl.analysis.tensor import (
    build_value_tensor, hosvd_spectra, hosvd_stable_ranks,
    tucker_reconstruction_error, _mode_unfold,
)
from low_rank_rl.analysis.hankel import (
    build_hankel_matrix, collect_trajectory, hankel_rank_metrics, dmd_from_hankel,
)
from low_rank_rl.analysis.successor import (
    shifted_successor_matrix, build_successor_matrix, successor_features,
    SuccessorComparison,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

def make_agent(env=None, n_obs: int = 6, n_actions: int = 3):
    if env is not None:
        n_obs     = env.observation_space.shape[0]
        n_actions = env.action_space.n
    return DQNAgent(n_obs, n_actions, hidden=16, device="cpu")


def make_tabular_agent(env):
    return QLearningAgent(env=env, n_actions=env.action_space.n)


# ── rank.py ───────────────────────────────────────────────────────────────────

class TestRankMetrics:
    def test_rank1_matrix(self):
        u = np.random.randn(20, 1)
        v = np.random.randn(1, 5)
        Q = u @ v
        m = _metrics_from_matrix(Q, tol=1e-5)
        assert m.numerical_rank == 1

    def test_full_rank_matrix(self):
        Q = np.random.randn(20, 5)
        m = _metrics_from_matrix(Q, tol=1e-10)
        assert m.numerical_rank == 5

    def test_singular_values_descending(self):
        Q = np.random.randn(20, 5)
        m = _metrics_from_matrix(Q, tol=1e-10)
        assert np.all(np.diff(m.singular_values) <= 0)

    def test_summary_string(self):
        Q = np.random.randn(10, 3)
        m = _metrics_from_matrix(Q, tol=1e-5)
        s = m.summary()
        assert "numerical rank" in s

    def test_compute_rank_metrics_shape_consistency(self):
        env   = make_env("Acrobot-v1", discretize_obs=False)
        agent = make_agent(env)
        states = sample_states(env, 32)
        m = compute_rank_metrics(agent, states)
        assert m.matrix_shape == (32, env.action_space.n)
        env.close()

    def test_sample_states_returns_full_canonical_grid_when_discretised(self):
        env = make_env("MountainCarContinuous-v0")  # 40 x 40 discretised by default
        states = sample_states(env, 64)
        assert states.shape == (40 * 40, env.observation_space.shape[0])
        env.close()

    def test_sample_states_continuous_env_uses_mc(self):
        env    = make_env("MountainCarContinuous-v0", discretize_obs=False)
        states = sample_states(env, 20)
        assert states.shape == (20, env.observation_space.shape[0])
        env.close()

    def test_canonical_states_none_when_not_discretised(self):
        env = make_env("MountainCarContinuous-v0", discretize_obs=False)
        assert canonical_states(env) is None
        env.close()

    def test_canonical_subsample_caps_at_n(self):
        env = make_env("MountainCarContinuous-v0")  # 1600 canonical states
        sub = canonical_subsample(env, 64)
        assert sub.shape == (64, env.observation_space.shape[0])
        env.close()

    def test_canonical_subsample_returns_full_grid_when_n_exceeds(self):
        env = make_env("MountainCarContinuous-v0")
        sub = canonical_subsample(env, 10_000)
        assert sub.shape == (1600, env.observation_space.shape[0])
        env.close()

    def test_compute_from_matrix(self):
        Q = np.eye(5)
        m = compute_rank_metrics_from_matrix(Q, tol=1e-10)
        assert m.numerical_rank == 5

    def test_sample_states_shape(self):
        env    = make_env("Acrobot-v1", discretize_obs=False)
        obs_dim = env.observation_space.shape[0]
        states = sample_states(env, 64)
        assert states.shape == (64, obs_dim)
        env.close()


# ── tensor.py ────────────────────────────────────────────────────────────────

class TestTensorAnalysis:
    def test_mode_unfold_shape(self):
        T = np.random.randn(4, 5, 6)
        assert _mode_unfold(T, 0).shape == (4, 30)
        assert _mode_unfold(T, 1).shape == (5, 24)
        assert _mode_unfold(T, 2).shape == (6, 20)

    def test_hosvd_spectra_keys(self):
        T       = np.random.randn(4, 5, 6)
        spectra = hosvd_spectra(T)
        assert set(spectra.keys()) == {0, 1, 2}

    def test_hosvd_spectra_descending(self):
        T       = np.random.randn(4, 5, 6)
        spectra = hosvd_spectra(T)
        for sigma in spectra.values():
            assert np.all(sigma[:-1] >= sigma[1:] - 1e-10)

    def test_hosvd_stable_ranks_positive(self):
        T      = np.random.randn(4, 5, 6)
        ranks  = hosvd_stable_ranks(T)
        for r in ranks.values():
            assert r >= 1.0

    def test_build_value_tensor_shape(self):
        env   = make_env("Acrobot-v1")
        agent = make_agent(env)
        T     = build_value_tensor(agent, env, dims=[0, 1], n_bins=5, n_samples=200)
        assert T.shape == (5, 5)
        env.close()

    def test_build_value_tensor_no_nans(self):
        env   = make_env("Acrobot-v1")
        agent = make_agent(env)
        T     = build_value_tensor(agent, env, dims=[0, 1], n_bins=5, n_samples=200)
        assert np.all(np.isfinite(T))
        env.close()

    def test_tucker_reconstruction_full_rank_is_exact(self):
        tensorly = pytest.importorskip("tensorly")
        T = np.random.randn(4, 5, 6)
        rec, err = tucker_reconstruction_error(T, ranks=[4, 5, 6])
        assert rec.shape == T.shape
        assert err < 1e-6

    def test_tucker_reconstruction_low_rank_has_error(self):
        pytest.importorskip("tensorly")
        T = np.random.randn(6, 6, 6)
        _, err = tucker_reconstruction_error(T, ranks=[1, 1, 1])
        assert err > 0.0


# ── hankel.py ────────────────────────────────────────────────────────────────

class TestHankelMatrix:
    def test_shape(self):
        seq = np.arange(10, dtype=float)
        H   = build_hankel_matrix(seq, n_rows=4)
        assert H.shape == (4, 7)

    def test_antidiagonal_structure(self):
        seq = np.arange(10, dtype=float)
        H   = build_hankel_matrix(seq, n_rows=4)
        # Every anti-diagonal should be constant: H[i,j] = i+j
        for i in range(4):
            for j in range(7):
                assert H[i, j] == seq[i + j]

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            build_hankel_matrix(np.arange(3, dtype=float), n_rows=5)

    def test_rank1_signal_gives_rank1_hankel(self):
        """A constant sequence is rank-1."""
        seq   = np.ones(20)
        H     = build_hankel_matrix(seq, n_rows=10)
        sigma = np.linalg.svd(H, compute_uv=False)
        assert (sigma[1:] < 1e-10).all()

    def test_sinusoid_low_rank(self):
        """A pure sinusoid generates a rank-2 Hankel matrix."""
        t     = np.linspace(0, 4 * np.pi, 100)
        seq   = np.sin(t)
        H     = build_hankel_matrix(seq, n_rows=50)
        sigma = np.linalg.svd(H, compute_uv=False)
        significant = np.sum(sigma > 1e-3 * sigma[0])
        assert significant <= 3  # rank ≤ 2 plus numerical noise

    def test_collect_trajectory_keys(self):
        env   = make_env("Acrobot-v1")
        agent = make_agent(env)
        traj  = collect_trajectory(agent, env, n_steps=20)
        assert set(traj.keys()) == {"states", "actions", "value", "q_taken", "policy"}
        env.close()

    def test_collect_trajectory_lengths_consistent(self):
        env   = make_env("Acrobot-v1")
        agent = make_agent(env)
        traj  = collect_trajectory(agent, env, n_steps=20)
        T     = len(traj["states"])
        for k in ("actions", "value", "q_taken", "policy"):
            assert len(traj[k]) == T
        env.close()

    def test_hankel_rank_metrics_returns_dataclass(self):
        from low_rank_rl.analysis.hankel import HankelMetrics
        env     = make_env("Acrobot-v1")
        agent   = make_agent(env)
        metrics = hankel_rank_metrics(agent, env, sequence_type="value", n_steps=40)
        assert isinstance(metrics, HankelMetrics)
        assert metrics.numerical_rank >= 1
        env.close()

    def test_dmd_from_hankel_shapes(self):
        seq   = np.sin(np.linspace(0, 4 * np.pi, 60))
        H     = build_hankel_matrix(seq, n_rows=30)
        modes, eigs = dmd_from_hankel(H, rank=4)
        assert modes.shape[1] == 4
        assert len(eigs) == 4

    def test_dmd_eigenvalues_complex(self):
        seq   = np.sin(np.linspace(0, 4 * np.pi, 60))
        H     = build_hankel_matrix(seq, n_rows=30)
        _, eigs = dmd_from_hankel(H, rank=4)
        assert eigs.dtype in (np.complex64, np.complex128)


# ── successor.py ─────────────────────────────────────────────────────────────

class TestSuccessorAnalysis:
    def test_shifted_matrix_shape(self):
        M   = np.random.rand(8, 8)
        Ms  = shifted_successor_matrix(M)
        assert Ms.shape == M.shape

    def test_shifted_column_means_near_zero(self):
        """After shifting, column means should be approximately zero."""
        M   = np.random.rand(10, 10)
        Ms  = shifted_successor_matrix(M)
        np.testing.assert_allclose(Ms.mean(axis=0), 0.0, atol=1e-10)

    def test_shift_does_not_increase_numerical_rank(self):
        """Since M̃ = M(I - 1·1ᵀ/n) is a projection, rank(M̃) ≤ rank(M)."""
        np.random.seed(0)
        M    = np.random.rand(20, 20)
        Ms   = shifted_successor_matrix(M)
        r_M  = _metrics_from_matrix(M,  tol=1e-8).numerical_rank
        r_Ms = _metrics_from_matrix(Ms, tol=1e-8).numerical_rank
        assert r_Ms <= r_M

    def test_shift_removes_pure_stationary(self):
        """A matrix whose rows are all identical (pure stationary) shifts to zero."""
        mu = np.random.rand(20)
        M  = np.ones((20, 1)) @ mu[None, :]
        Ms = shifted_successor_matrix(M)
        np.testing.assert_allclose(Ms, 0.0, atol=1e-10)

    def test_shift_reduces_rank_on_stationary_plus_low_rank(self):
        """M = stationary (rank 1) + low-rank-2 signal  →  shifted is rank ≤ 2."""
        np.random.seed(42)
        A          = np.random.randn(20, 2) @ np.random.randn(2, 20)
        stationary = np.ones((20, 1)) @ np.random.rand(1, 20)
        M          = stationary + A
        Ms         = shifted_successor_matrix(M)
        r_M        = _metrics_from_matrix(M,  tol=1e-8).numerical_rank
        r_Ms       = _metrics_from_matrix(Ms, tol=1e-8).numerical_rank
        assert r_Ms <= 2
        assert r_Ms < r_M

    def test_successor_comparison_summary(self):
        from low_rank_rl.analysis.rank import _metrics_from_matrix
        M   = np.random.rand(8, 8)
        Ms  = shifted_successor_matrix(M)
        cmp = SuccessorComparison(
            vanilla=_metrics_from_matrix(M,  1e-5),
            shifted=_metrics_from_matrix(Ms, 1e-5),
        )
        s = cmp.summary()
        assert "Vanilla" in s
        assert "Shifted" in s

    def test_build_successor_matrix_shape_and_nonneg(self):
        env   = make_env("Acrobot-v1", discretize_obs=False)
        agent = make_agent(env)
        states = sample_states(env, 8)
        M      = build_successor_matrix(agent, env, states, gamma=0.9, n_rollout_steps=20)
        assert M.shape == (8, 8)
        assert (M >= 0).all()
        env.close()

    def test_successor_features_shape(self):
        env    = make_env("Acrobot-v1", discretize_obs=False)
        agent  = make_agent(env)
        states = sample_states(env, 4)
        psi    = successor_features(
            agent, env, states,
            feature_fn=lambda s: s[:3],
            gamma=0.9, n_rollout_steps=20,
        )
        assert psi.shape == (4, 3)
        env.close()
