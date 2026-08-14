"""Smoke tests for FHR-DQN (recurrence penalty, episodic-buffer handles,
agent). Run as `python tests/test_fhrdqn.py` or via pytest from repo root."""
import pathlib
import random
import sys
import tempfile
import warnings

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import gymnasium as gym

from agents.fhrdqn_agent import FHRDQNAgent
from agents.hankel_dqn_agent import EpisodicReplayBuffer
from agents.q_agent import QAgent
from analysis.run_logger import RunLogger
from training import dqn_training_loop


class TinyQNet(nn.Module):
    def __init__(self, obs_dim=4, n_actions=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


class FirstCoordQNet(nn.Module):
    """Q(s, a) = s[0] for every action — lets a test dictate the exact value
    sequence along an episode through the stored states. The zero-weighted
    linear term keeps the output attached to parameters so backward() works."""
    def __init__(self, obs_dim=2, n_actions=2):
        super().__init__()
        self.lin = nn.Linear(obs_dim, n_actions)

    def forward(self, x):
        return x[:, :1].repeat(1, 2) + 0.0 * self.lin(x)


def _fill_buffer(buf, ep_lens, terminated_flags):
    for e, (L, term) in enumerate(zip(ep_lens, terminated_flags)):
        for t in range(L):
            s = torch.tensor([[float(e), float(t)]])
            a = torch.tensor([t % 2])
            r = torch.tensor([float(t)])
            last = t == L - 1
            nxt = None if (last and term) else torch.tensor([[float(e), float(t + 1)]])
            buf.append(s, a, r, nxt)
        buf.close(term)


def test_handles_and_predecessors():
    random.seed(3)
    buf = EpisodicReplayBuffer(capacity=10_000)
    ep_lens, terms = [12, 5, 30, 7], [True, False, True, False]
    _fill_buffer(buf, ep_lens, terms)

    r = 3
    for _ in range(200):
        states, actions, rewards, nexts, handles = buf.sample_transitions(16, with_handles=True)
        assert len(handles) == 16
        for i, (e, t) in enumerate(handles):
            # Handle agrees with the (e, t) encoded in the state itself.
            assert int(states[i, 0]) == e and int(states[i, 1]) == t
        keep = [h for h in handles if h[1] >= r]
        if not keep:
            continue
        ps, pa, pr = buf.gather_predecessors(keep, r)
        assert ps.shape == (len(keep), r, 2) and pa.shape == (len(keep), r)
        for i, (e, t) in enumerate(keep):
            for j in range(r):  # output[:, j] = transition at t-(j+1), same episode
                assert int(ps[i, j, 0]) == e, "predecessor crossed episodes"
                assert int(ps[i, j, 1]) == t - (j + 1), "wrong lag ordering"
                assert float(pr[i, j]) == float(t - (j + 1)), "rewards misaligned"

    # Handles work for the open episode too, and t < r is rejected.
    for t in range(5):
        buf.append(torch.tensor([[9.0, float(t)]]), torch.tensor([0]),
                   torch.tensor([0.0]), torch.tensor([[9.0, float(t + 1)]]))
    ps, _, _ = buf.gather_predecessors([(len(buf._episodes), 4)], r)
    assert int(ps[0, 0, 1]) == 3 and int(ps[0, 0, 0]) == 9
    try:
        buf.gather_predecessors([(0, r - 1)], r)
        raise AssertionError("expected ValueError for t < r")
    except ValueError:
        pass

    # with_handles must not change the RNG stream or the sample.
    random.seed(7)
    plain = buf.sample_transitions(8)
    random.seed(7)
    with_h = buf.sample_transitions(8, with_handles=True)
    assert torch.equal(plain[0], with_h[0]) and torch.equal(plain[1], with_h[1])


def _make_fhr(weight, warmup=0, seed=0, **overrides):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    kwargs = dict(fhr_weight=weight, fhr_order=2, warmup_grad_steps=warmup,
                  replay_buffer_capacity=1000, q_network=TinyQNet, batch_size=16,
                  nn_learning_rate=1e-3, nn_extra_kwargs={}, env=gym.make("CartPole-v1"),
                  eps_start=1.0, eps_min=0.05, decay_rate=0.999, discount_factor=0.99,
                  device="cpu", TD_LR=0.05, buffer_util=1, gd_steps_ceil=1, double=True)
    kwargs.update(overrides)
    return FHRDQNAgent(**kwargs)


def _fill_agent(agent, n_eps=3, ep_len=12, reward=1.0):
    rng = np.random.RandomState(7)
    for ep in range(n_eps):
        states = rng.randn(ep_len + 1, 4).astype(np.float32)
        for t in range(ep_len):
            terminated = (t == ep_len - 1) and (ep % 2 == 0)
            truncated = (t == ep_len - 1) and not terminated
            agent.update_buffer(states[t], t % 2, reward, states[t + 1], terminated, truncated)


def test_lambda0_matches_qagent_exactly():
    """fhr_weight = 0 must reproduce plain QAgent training bit-for-bit: same
    nets at init, same replay draws, same parameters after training."""
    def _mk(cls, **extra):
        torch.manual_seed(0); np.random.seed(0); random.seed(0)
        return cls(replay_buffer_capacity=1000, q_network=TinyQNet, batch_size=16,
                   nn_learning_rate=1e-3, nn_extra_kwargs={}, env=gym.make("CartPole-v1"),
                   eps_start=1.0, eps_min=0.05, decay_rate=0.999, discount_factor=0.99,
                   device="cpu", TD_LR=0.05, buffer_util=1, gd_steps_ceil=2, double=True,
                   **extra)
    q = _mk(QAgent)
    f = _mk(FHRDQNAgent, fhr_weight=0.0)
    for k, v in q.policy_net.state_dict().items():
        assert torch.equal(v, f.policy_net.state_dict()[k])
    _fill_agent(q), _fill_agent(f)
    for _ in range(3):
        random.seed(123), torch.manual_seed(123)
        q.train()
        random.seed(123), torch.manual_seed(123)
        f.train()
    for k, v in q.policy_net.state_dict().items():
        assert torch.equal(v, f.policy_net.state_dict()[k]), f"params diverge at {k}"


def test_bh_empty_guard():
    # r larger than every episode: no sample ever has r predecessors, so the
    # penalty term must be skipped without NaNs and TD must still train.
    a = _make_fhr(0.01, fhr_order=10)
    _fill_agent(a, n_eps=4, ep_len=8)  # episodes shorter than r+1
    before = [p.clone() for p in a.policy_net.parameters()]
    d = a.train()
    assert d is not None and a.nan_skips == 0
    assert d["b_h"] == 0.0 and np.isnan(d["penalty_raw"])
    assert any(not torch.equal(b, p) for b, p in zip(before, a.policy_net.parameters()))


def _recurrence_episode(agent, coeffs, v0, v1, ep_len, reward=0.0):
    """Feed one episode whose FirstCoordQNet value sequence follows
    v_t = coeffs[0] v_{t-1} + coeffs[1] v_{t-2} exactly."""
    v = [v0, v1]
    for _ in range(ep_len - 2):
        v.append(coeffs[0] * v[-1] + coeffs[1] * v[-2])
    for t in range(ep_len):
        s = np.array([v[t], 0.0], dtype=np.float32)
        nxt = np.array([v[t + 1] if t + 1 < ep_len else 0.0, 0.0], dtype=np.float32)
        agent.update_buffer(s, t % 2, reward, nxt, terminated=False,
                            truncated=(t == ep_len - 1))


def test_zero_residual_on_exact_recurrence():
    gamma = 0.99
    a = _make_fhr(0.01, fhr_order=2, q_network=FirstCoordQNet,
                  discount_factor=gamma, batch_size=16, gd_steps_ceil=1)
    c1, c2 = 1.0 + 1.0 / gamma, -1.0 / gamma  # the agent's own init
    _recurrence_episode(a, (c1, c2), v0=5.0, v1=5.1, ep_len=40)
    d = a.train()
    assert d is not None and d["b_h"] > 0
    assert d["penalty_raw"] < 1e-6, f"exact recurrence gave penalty {d['penalty_raw']}"
    # Perturbed sequence must give a strictly positive penalty.
    b = _make_fhr(0.01, fhr_order=2, q_network=FirstCoordQNet,
                  discount_factor=gamma, batch_size=16, gd_steps_ceil=1)
    _recurrence_episode(b, (c1 + 0.3, c2), v0=5.0, v1=5.1, ep_len=40)
    d2 = b.train()
    assert d2["penalty_raw"] > 1e-4


def test_arx_exact_bellman_order1():
    # Bellman-consistent constant-reward values: v_t = (v_{t-1} - rew)/gamma.
    # ARX(1) at the init c_1 = 1/gamma, d_1 = -1/gamma has zero residual.
    gamma, rew = 0.99, 1.0
    a = _make_fhr(0.01, fhr_order=1, reward_lags=True, q_network=FirstCoordQNet,
                  discount_factor=gamma, batch_size=16, gd_steps_ceil=1)
    v = [50.0]
    ep_len = 40
    for _ in range(ep_len):
        v.append((v[-1] - rew) / gamma)
    for t in range(ep_len):
        s = np.array([v[t], 0.0], dtype=np.float32)
        nxt = np.array([v[t + 1], 0.0], dtype=np.float32)
        a.update_buffer(s, t % 2, rew, nxt, terminated=False, truncated=(t == ep_len - 1))
    d = a.train()
    assert d is not None and d["b_h"] > 0
    assert d["penalty_raw"] < 1e-6, f"Bellman-consistent ARX(1) penalty {d['penalty_raw']}"
    assert "d_1" in d and abs(d["d_1"] + 1.0 / gamma) < 1e-6


def test_r1_pure_ar_warns_arx_does_not():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _make_fhr(0.01, fhr_order=1)
        assert any("fhr_order=1" in str(x.message) for x in w)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _make_fhr(0.01, fhr_order=1, reward_lags=True)
        assert not any("fhr_order=1" in str(x.message) for x in w)


def test_coeff_param_group_and_init():
    gamma = 0.99
    a = _make_fhr(0.01, c_learning_rate=0.007)
    group = a.optimiser.param_groups[-1]
    assert group["weight_decay"] == 0.0 and group["lr"] == 0.007
    assert any(p is a.c for p in group["params"])
    assert all(p is not a.c for p in a.policy_net.parameters())  # excluded from clip
    assert all(p is not a.c for p in a.target_net.parameters())  # excluded from target
    # Pure-AR init annihilates constant-reward Bellman sequences: roots {1, 1/gamma}.
    assert abs(float(a.c[0].detach()) - (1 + 1 / gamma)) < 1e-6
    assert abs(float(a.c[1].detach()) + 1 / gamma) < 1e-6
    assert abs(a._companion_radius() - 1 / gamma) < 1e-4
    # theta's group keeps AdamW's default decay; the coefficient group has none.
    assert a.optimiser.param_groups[0]["weight_decay"] != 0.0


def test_hard_warmup_no_ramp():
    a = _make_fhr(0.05, warmup=10)
    a._grad_steps = 0
    assert a._lambda_eff() == 0.0
    a._grad_steps = 9
    assert a._lambda_eff() == 0.0
    a._grad_steps = 10
    assert a._lambda_eff() == 0.05  # full strength immediately, no ramp


def test_atari_path_raises():
    a = _make_fhr(0.0)
    try:
        a.update_buffer_atari(np.zeros(4), 0, 0.0, np.zeros(4), False)
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError:
        pass


def test_save_load_roundtrip():
    a = _make_fhr(0.01)
    with torch.no_grad():
        a.c += 0.25
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "ckpt.pt"
        a.save(path)
        b = _make_fhr(0.01, seed=1)
        b.load(path)
        assert torch.equal(a.c.detach(), b.c.detach())
        for k, v in a.policy_net.state_dict().items():
            assert torch.equal(v, b.policy_net.state_dict()[k])


def test_tiny_e2e_cartpole():
    torch.manual_seed(3), np.random.seed(3), random.seed(3)
    env = gym.make("CartPole-v1")
    agent = _make_fhr(1e-2, warmup=10, seed=3, batch_size=32, gd_steps_ceil=2, env=env)
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(tmp)
        rewards = dqn_training_loop(
            agent, env, no_episodes=25, target_network_update_steps=50,
            train_frequency_steps=1, use_episode_training=True,
            solved_reward=10 ** 9, warmup_steps=50,
            early_stopping_patience_eps=10 ** 9,
            analysis_config={"ep_freq": 10 ** 9, "methods": []},
            run_logger=logger)
        assert len(rewards) == 25
        assert agent.nan_skips == 0 and agent._grad_steps > 0
        diag_csv = logger.dir / "train_diagnostics.csv"
        assert diag_csv.exists()
        header = diag_csv.read_text().splitlines()[0].split(",")
        for col in ("td_loss", "penalty_raw", "b_h", "c_1", "c_2",
                    "companion_radius", "residual_rms", "sum_c"):
            assert col in header, f"missing diagnostics column {col}"


def test_rampdown_trigger_and_linear_ramp():
    a = _make_fhr(0.2, warmup=0, rampdown_reward_threshold=100.0,
                  rampdown_penalty_threshold=1.0, rampdown_patience_eps=3,
                  rampdown_episodes=4)
    # reward below threshold: high residual alone never triggers
    for ep in range(3):
        a._ep_penalty_vals = [5.0]
        a.notify_episode_end(ep, 50.0)
    assert a._rd_k is None and a._lambda_eff() == 0.2
    # one low-residual episode inside the window blocks the trigger
    for ep, pen in ((3, 5.0), (4, 0.1), (5, 5.0)):
        a._ep_penalty_vals = [pen]
        a.notify_episode_end(ep, 150.0)
    assert a._rd_k is None
    a._ep_penalty_vals = [5.0]
    a.notify_episode_end(6, 150.0)          # window (4, 5, 6) still holds the 0.1
    assert a._rd_k is None
    a._ep_penalty_vals = [5.0]
    a.notify_episode_end(7, 150.0)          # window (5, 6, 7): all high + reward high
    assert a._rd_k == 1
    # linear ramp over 4 episodes: lambda = 0.2 * (0.75, 0.5, 0.25, 0), then stays 0
    got = []
    for ep in (8, 9, 10, 11):
        got.append(a._lambda_eff())
        a._ep_penalty_vals = [5.0]
        a.notify_episode_end(ep, 150.0)
    assert np.allclose(got, [0.15, 0.10, 0.05, 0.0]), got
    assert a._lambda_eff() == 0.0
    # an episode with NO penalty samples reads as NaN and can never trigger
    b = _make_fhr(0.2, warmup=0, rampdown_reward_threshold=10.0,
                  rampdown_penalty_threshold=0.0, rampdown_patience_eps=1,
                  rampdown_episodes=0)
    b.notify_episode_end(0, 100.0)          # _ep_penalty_vals empty -> NaN
    assert b._rd_k is None and b._lambda_eff() == 0.2


def test_rampdown_immediate_disabled_and_warmup_gated():
    # rampdown_episodes = 0: lambda cut to 0 the moment it triggers; with no
    # penalty threshold the reward window alone gates
    a = _make_fhr(0.3, warmup=0, rampdown_reward_threshold=10.0,
                  rampdown_patience_eps=2, rampdown_episodes=0)
    a.notify_episode_end(0, 20.0)
    assert a._rd_k is None and a._lambda_eff() == 0.3
    a.notify_episode_end(1, 20.0)
    assert a._rd_k == 1 and a._lambda_eff() == 0.0
    # disabled (reward threshold None): the hook never schedules anything
    b = _make_fhr(0.3, warmup=0)
    for ep in range(5):
        b._ep_penalty_vals = [9.9]
        b.notify_episode_end(ep, 1e9)
    assert b._rd_k is None and b._lambda_eff() == 0.3
    # warm-up gates the trigger: conditions met but lambda still in warm-up
    c = _make_fhr(0.2, warmup=10 ** 9, rampdown_reward_threshold=10.0,
                  rampdown_patience_eps=1, rampdown_episodes=0)
    c._ep_penalty_vals = [5.0]
    c.notify_episode_end(0, 100.0)
    assert c._rd_k is None and c._lambda_eff() == 0.0
    # constructor validation
    for bad in (dict(rampdown_episodes=-1), dict(rampdown_patience_eps=0)):
        try:
            _make_fhr(0.1, **bad)
            assert False, f"expected ValueError for {bad}"
        except ValueError:
            pass


def test_rampdown_window_ignores_warmup_and_baseline_arm():
    # warm-up era episodes must not pre-fill the trigger window — otherwise a
    # good-enough policy fires the trigger on the first post-warm-up hook and
    # the penalty phase is skipped entirely
    a = _make_fhr(0.3, warmup=5, rampdown_reward_threshold=50.0,
                  rampdown_patience_eps=3, rampdown_episodes=0)
    for ep in range(4):                      # high-reward episodes in warm-up
        a.notify_episode_end(ep, 100.0)
    assert a._rd_window == [] and a._rd_k is None
    a._grad_steps = 5                        # warm-up over
    a.notify_episode_end(4, 100.0)
    a.notify_episode_end(5, 100.0)
    assert a._rd_k is None                   # only 2 post-warm-up episodes
    a.notify_episode_end(6, 100.0)
    assert a._rd_k == 1                      # 3rd qualifying episode triggers
    # the baseline arm (fhr_weight 0) stays inert even in reward-only mode:
    # no spurious trigger line / rampdown_scale ramp in its diagnostics
    b = _make_fhr(0.0, warmup=0, rampdown_reward_threshold=10.0,
                  rampdown_patience_eps=1, rampdown_episodes=0)
    b.notify_episode_end(0, 100.0)
    assert b._rd_k is None and b._rampdown_scale() == 1.0


def test_rampdown_relative_penalty_threshold():
    # "50%" bar = half the mean of the topk largest per-episode residuals seen
    # so far; the bar is undefined (condition fails) until topk episodes exist
    a = _make_fhr(0.2, warmup=0, rampdown_reward_threshold=100.0,
                  rampdown_penalty_threshold="50%", rampdown_penalty_topk=3,
                  rampdown_patience_eps=2, rampdown_episodes=0)
    for ep, pen in enumerate((4.0, 6.0, 8.0)):   # low reward: only builds history
        a._ep_penalty_vals = [pen]
        a.notify_episode_end(ep, 0.0)
    assert a._penalty_bar() == 0.5 * (4 + 6 + 8) / 3
    a._ep_penalty_vals = [3.5]; a.notify_episode_end(3, 150.0)
    assert a._rd_k is None                       # window mean reward still low
    a._ep_penalty_vals = [3.5]; a.notify_episode_end(4, 150.0)
    assert a._rd_k == 1                          # 3.5 >= bar 3.0, reward high
    # bar undefined until topk history entries exist
    b = _make_fhr(0.2, warmup=0, rampdown_reward_threshold=10.0,
                  rampdown_penalty_threshold="50%", rampdown_penalty_topk=5,
                  rampdown_patience_eps=1, rampdown_episodes=0)
    for ep in range(4):
        b._ep_penalty_vals = [9.0]
        b.notify_episode_end(ep, 100.0)
    assert b._rd_k is None                       # only 4 of 5 history entries
    b._ep_penalty_vals = [9.0]
    b.notify_episode_end(4, 100.0)
    assert b._rd_k == 1                          # bar 4.5, residual 9
    # residuals that collapsed relative to their historical highs never trigger
    c = _make_fhr(0.2, warmup=0, rampdown_reward_threshold=10.0,
                  rampdown_penalty_threshold="50%", rampdown_penalty_topk=2,
                  rampdown_patience_eps=2, rampdown_episodes=0)
    for ep, (pen, rew) in enumerate(((10.0, 0.0), (10.0, 0.0),
                                     (1.0, 100.0), (1.0, 100.0))):
        c._ep_penalty_vals = [pen]
        c.notify_episode_end(ep, rew)
    assert c._rd_k is None                       # bar 5.0, recent residuals 1.0
    # malformed strings are rejected at construction
    for bad in ("high", "40", "40 percent"):
        try:
            _make_fhr(0.1, rampdown_penalty_threshold=bad)
            assert False, f"expected ValueError for {bad!r}"
        except ValueError:
            pass


def test_rampdown_e2e_training_loop():
    """The dqn_training_loop hook drives the trigger; lambda_eff falls to 0 in
    the diagnostics once an easy trigger fires."""
    torch.manual_seed(5), np.random.seed(5), random.seed(5)
    env = gym.make("CartPole-v1")
    agent = _make_fhr(1e-2, warmup=0, seed=5, batch_size=16, gd_steps_ceil=2,
                      env=env, rampdown_reward_threshold=5.0,
                      rampdown_penalty_threshold=0.0, rampdown_patience_eps=2,
                      rampdown_episodes=3)
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(tmp)
        dqn_training_loop(
            agent, env, no_episodes=20, target_network_update_steps=50,
            train_frequency_steps=1, use_episode_training=True,
            solved_reward=10 ** 9, warmup_steps=0,
            early_stopping_patience_eps=10 ** 9,
            analysis_config={"ep_freq": 10 ** 9, "methods": []},
            run_logger=logger)
        assert agent._rd_k is not None, "easy trigger never fired in 20 episodes"
        assert agent._lambda_eff() == 0.0
        diag_csv = (logger.dir / "train_diagnostics.csv").read_text().splitlines()
        header = diag_csv[0].split(",")
        assert "rampdown_scale" in header and "lambda_eff" in header
        li = header.index("lambda_eff")
        assert float(diag_csv[-1].split(",")[li]) == 0.0


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in fns:
        fn()
        print(f"{name} OK")
    print(f"all {len(fns)} tests passed")
