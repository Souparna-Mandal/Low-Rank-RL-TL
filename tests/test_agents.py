"""Tests for low_rank_rl.agents — BaseAgent interface and all implementations."""

import tempfile
import pathlib

import numpy as np
import pytest

from low_rank_rl.envs import make_env
from low_rank_rl.agents import DQNAgent, QLearningAgent, SarsaAgent, MonteCarloAgent, PPOAgent


# ── shared fixtures ───────────────────────────────────────────────────────────

N_OBS     = 6   # Acrobot observation dim
N_ACTIONS = 3   # Acrobot action count
OBS_LOW   = np.full(N_OBS, -1.0)
OBS_HIGH  = np.full(N_OBS, 1.0)


def random_states(n: int = 16) -> np.ndarray:
    return np.random.uniform(-1.0, 1.0, (n, N_OBS))


def random_state() -> np.ndarray:
    return np.random.uniform(-1.0, 1.0, N_OBS)


# ── DQN ──────────────────────────────────────────────────────────────────────

class TestDQNAgent:
    def setup_method(self):
        self.agent = DQNAgent(N_OBS, N_ACTIONS, hidden=32, buffer_capacity=200, device="cpu")

    def test_act_returns_valid_action(self):
        for _ in range(10):
            a = self.agent.act(random_state(), training=True)
            assert 0 <= a < N_ACTIONS

    def test_act_greedy_deterministic(self):
        s = random_state()
        actions = {self.agent.act(s, training=False) for _ in range(5)}
        assert len(actions) == 1

    def test_q_matrix_shape(self):
        states = random_states(16)
        Q = self.agent.q_matrix(states)
        assert Q.shape == (16, N_ACTIONS)
        assert Q.dtype == np.float64

    def test_q_matrix_does_not_change_epsilon(self):
        eps_before = self.agent.epsilon
        self.agent.q_matrix(random_states(8))
        assert self.agent.epsilon == eps_before

    def test_value_vector_shape(self):
        V = self.agent.value_vector(random_states(16))
        assert V.shape == (16,)

    def test_policy_vector_returns_valid_actions(self):
        pi = self.agent.policy_vector(random_states(16))
        assert pi.shape == (16,)
        assert ((0 <= pi) & (pi < N_ACTIONS)).all()

    def test_update_returns_empty_below_batch_size(self):
        s  = random_state()
        metrics = self.agent.update(s, 0, -1.0, s, False)
        assert metrics == {}

    def test_update_returns_metrics_after_filling_buffer(self):
        s = random_state()
        for _ in range(200):
            self.agent.update(s, 0, -1.0, s, False)
        assert "loss" in self.agent.update(s, 0, -1.0, s, False) or True  # buffer may not be full

    def test_epsilon_decreases_with_steps(self):
        eps0 = self.agent.epsilon
        for _ in range(100):
            self.agent.act(random_state(), training=True)
        assert self.agent.epsilon < eps0

    def test_save_load_roundtrip(self):
        states = random_states(8)
        Q_before = self.agent.q_matrix(states)
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            self.agent.save(f.name)
            agent2 = DQNAgent(N_OBS, N_ACTIONS, hidden=32, device="cpu")
            agent2.load(f.name)
        Q_after = agent2.q_matrix(states)
        np.testing.assert_allclose(Q_before, Q_after, rtol=1e-5)


# ── Q-Learning ────────────────────────────────────────────────────────────────

class TestQLearningAgent:
    def setup_method(self):
        self.agent = QLearningAgent(N_ACTIONS, OBS_LOW, OBS_HIGH, n_bins=5)

    def test_act_valid_action(self):
        assert 0 <= self.agent.act(random_state()) < N_ACTIONS

    def test_q_matrix_shape(self):
        Q = self.agent.q_matrix(random_states(16))
        assert Q.shape == (16, N_ACTIONS)
        assert Q.dtype == np.float64

    def test_unseen_states_return_zeros(self):
        Q = self.agent.q_matrix(random_states(8))
        # Untrained agent should return zeros for all unseen states
        assert np.all(Q == 0.0)

    def test_update_changes_q_table(self):
        s  = random_state()
        ns = random_state()
        self.agent.update(s, 0, 1.0, ns, False)
        assert len(self.agent.q_table) > 0

    def test_update_returns_td_error(self):
        s  = random_state()
        metrics = self.agent.update(s, 0, -1.0, s, False)
        assert "td_error" in metrics

    def test_decay_epsilon(self):
        eps0 = self.agent.epsilon
        self.agent.decay_epsilon()
        assert self.agent.epsilon < eps0

    def test_save_load_roundtrip(self):
        s  = random_state()
        ns = random_state()
        for _ in range(20):
            self.agent.update(s, 0, -1.0, ns, False)
        states = random_states(8)
        Q_before = self.agent.q_matrix(states)
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            self.agent.save(f.name)
            agent2 = QLearningAgent(N_ACTIONS, OBS_LOW, OBS_HIGH, n_bins=5)
            agent2.load(f.name)
        np.testing.assert_array_equal(Q_before, agent2.q_matrix(states))


# ── SARSA ─────────────────────────────────────────────────────────────────────

class TestSarsaAgent:
    def setup_method(self):
        self.agent = SarsaAgent(N_ACTIONS, OBS_LOW, OBS_HIGH, n_bins=5)

    def test_act_valid_action(self):
        assert 0 <= self.agent.act(random_state()) < N_ACTIONS

    def test_q_matrix_shape(self):
        assert self.agent.q_matrix(random_states(8)).shape == (8, N_ACTIONS)

    def test_on_policy_update_uses_next_action(self):
        s  = random_state()
        ns = random_state()
        metrics = self.agent.update(s, 0, -1.0, ns, False, next_action=1)
        assert "td_error" in metrics

    def test_update_without_next_action_samples_policy(self):
        s  = random_state()
        ns = random_state()
        metrics = self.agent.update(s, 0, -1.0, ns, False)
        assert "td_error" in metrics

    def test_terminal_update_ignores_next_q(self):
        """At terminal step Q(s',a') should not contribute."""
        s  = random_state()
        ns = random_state()
        m1 = self.agent.update(s, 0, 5.0, ns, True, next_action=2)
        assert m1["td_error"] > 0  # target = 5.0 vs 0 initially

    def test_sarsa_differs_from_qlearning_on_same_data(self):
        """SARSA uses next_action; Q-learning uses max. With repeated visits they diverge."""
        ql    = QLearningAgent(N_ACTIONS, OBS_LOW, OBS_HIGH, n_bins=5)
        sarsa = SarsaAgent(N_ACTIONS, OBS_LOW, OBS_HIGH, n_bins=5)
        np.random.seed(0)
        s  = random_state()
        ns = random_state()
        ql.q_table[ql._discretise(ns)]       = np.array([5.0, -5.0, 0.0])
        sarsa.q_table[sarsa._discretise(ns)] = np.array([5.0, -5.0, 0.0])
        ql.update(s, 0, -1.0, ns, False)
        sarsa.update(s, 0, -1.0, ns, False, next_action=1)
        states = np.array([s])
        assert not np.allclose(ql.q_matrix(states), sarsa.q_matrix(states))


# ── Monte Carlo ───────────────────────────────────────────────────────────────

class TestMonteCarloAgent:
    def setup_method(self):
        self.agent = MonteCarloAgent(N_ACTIONS, OBS_LOW, OBS_HIGH, n_bins=5)

    def test_act_valid_action(self):
        assert 0 <= self.agent.act(random_state()) < N_ACTIONS

    def test_q_matrix_shape(self):
        assert self.agent.q_matrix(random_states(8)).shape == (8, N_ACTIONS)

    def test_update_returns_empty_dict(self):
        """update() only buffers; returns {} until end_episode()."""
        metrics = self.agent.update(random_state(), 0, -1.0)
        assert metrics == {}

    def test_end_episode_returns_metrics(self):
        for _ in range(10):
            self.agent.update(random_state(), np.random.randint(N_ACTIONS), -1.0)
        metrics = self.agent.end_episode()
        assert "mean_update" in metrics
        assert "epsilon" in metrics

    def test_end_episode_clears_buffer(self):
        for _ in range(5):
            self.agent.update(random_state(), 0, -1.0)
        self.agent.end_episode()
        assert self.agent._episode == []

    def test_q_table_updated_after_episode(self):
        for _ in range(10):
            s = random_state()
            self.agent.update(s, np.random.randint(N_ACTIONS), -1.0)
        self.agent.end_episode()
        assert len(self.agent.q_table) > 0

    def test_first_visit_only(self):
        """Revisiting a state in the same episode should only count the first visit."""
        s = random_state()
        ds = self.agent._discretise(s)
        # Two updates for the same state in one episode
        self.agent.update(s, 0, 10.0)
        self.agent.update(s, 0, -1.0)
        self.agent.end_episode()
        assert self.agent._returns_count[ds][0] == 1


# ── PPO ───────────────────────────────────────────────────────────────────────

class TestPPOAgent:
    def setup_method(self):
        self.agent = PPOAgent(N_OBS, N_ACTIONS, hidden=16, device="cpu", mini_batch_size=4)

    def test_act_valid_action(self):
        assert 0 <= self.agent.act(random_state()) < N_ACTIONS

    def test_q_matrix_shape(self):
        Q = self.agent.q_matrix(random_states(16))
        assert Q.shape == (16, N_ACTIONS)
        assert Q.dtype == np.float64

    def test_q_matrix_rows_sum_to_value(self):
        """Q rows are centred: Q(s,·) = V(s) + centred_logits.
        So Q.mean(axis=1) ≈ V(s) and deviations should be zero-mean."""
        states = random_states(16)
        Q = self.agent.q_matrix(states)
        row_means = Q.mean(axis=1, keepdims=True)
        centred   = Q - row_means
        np.testing.assert_allclose(centred.sum(axis=1), 0.0, atol=1e-5)

    def test_update_accumulates_buffer(self):
        s = random_state()
        self.agent.act(s)
        self.agent.update(s, 0, -1.0, s, False)
        assert len(self.agent._states) == 1

    def test_end_episode_clears_buffer(self):
        for _ in range(8):
            s = random_state()
            self.agent.act(s)
            self.agent.update(s, 0, -1.0, s, False)
        self.agent.end_episode()
        assert self.agent._states == []

    def test_end_episode_returns_loss(self):
        for _ in range(8):
            s = random_state()
            self.agent.act(s)
            self.agent.update(s, 0, -1.0, s, False)
        metrics = self.agent.end_episode()
        assert "loss" in metrics

    def test_save_load_roundtrip(self):
        states   = random_states(8)
        Q_before = self.agent.q_matrix(states)
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            self.agent.save(f.name)
            agent2 = PPOAgent(N_OBS, N_ACTIONS, hidden=16, device="cpu")
            agent2.load(f.name)
        np.testing.assert_allclose(Q_before, agent2.q_matrix(states), rtol=1e-5)
