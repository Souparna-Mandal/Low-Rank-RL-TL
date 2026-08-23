"""Smoke tests for the Hankel-rank-regularised DQN (penalty, episodic buffer,
agent). Run as `python tests/test_hankel_dqn.py` or via pytest from repo root."""
import pathlib
import random
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import gymnasium as gym

from agents.hankel_regulariser import HankelRankPenalty, _energy_rank
from agents.hankel_dqn_agent import EpisodicReplayBuffer, HankelDQNAgent
from analysis.run_logger import RunLogger
from training import dqn_training_loop


class TinyQNet(nn.Module):
    def __init__(self, obs_dim=4, n_actions=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


def _rank2_batch(B=6, T=16):
    """Damped oscillations: q_t = rho^t (a cos wt + b sin wt), an exact order-2
    recurrence, so Hankel rank <= 2."""
    t = torch.arange(T, dtype=torch.float64)
    seqs = []
    for i in range(B):
        rho, om = 0.95 + 0.004 * i, 0.3 + 0.05 * i
        seqs.append((rho ** t) * (1.5 * torch.cos(om * t) + 0.5 * torch.sin(om * t)))
    return torch.stack(seqs).float()


def test_penalty_zero_on_rank2_positive_on_noise():
    pen = HankelRankPenalty(order=2)
    p2, d2 = pen(_rank2_batch())
    assert float(p2) < 1e-5, f"rank-2 penalty {float(p2)}"
    assert d2["batch_eff_rank"] <= 2.5
    assert d2["converged_frac"] == 1.0 and d2["gate_frac"] == 0.0
    torch.manual_seed(0)
    pn, dn = pen(torch.randn(6, 16))
    assert float(pn) > 0.05, f"noise penalty {float(pn)}"
    assert dn["batch_eff_rank"] > 3
    assert int(_energy_rank(torch.zeros(1, 8))[0]) == 0  # reference convention


def test_penalty_minimizable_and_finite_grads():
    torch.manual_seed(1)
    x = (_rank2_batch(8, 16) + 0.3 * torch.randn(8, 16)).clone().requires_grad_(True)
    pen = HankelRankPenalty(order=2)
    opt = torch.optim.Adam([x], lr=0.02)
    p0 = None
    for _ in range(200):
        opt.zero_grad()
        p, _ = pen(x)
        p.backward()
        assert torch.isfinite(x.grad).all()
        opt.step()
        p0 = float(p) if p0 is None else p0
    pT, _ = pen(x)
    assert float(pT) < 0.5 * p0, f"penalty did not decrease: {p0} -> {float(pT)}"
    # Degenerate spectrum (constant seq -> rank 1, zero tail): no NaN, zero grad.
    c = torch.ones(4, 16, requires_grad=True)
    p, _ = pen(c)
    p.backward()
    assert float(p) == 0.0 and torch.isfinite(c.grad).all()


def test_log_transform_penalty():
    # Windows whose signed log sign(v)*log1p(|v|) is an exact rank-2 sequence:
    # low tail after the transform, high tail on the raw values.
    y = _rank2_batch(6, 16)
    v = torch.sign(y) * torch.expm1(y.abs())  # symlog^{-1}(y)
    p_log, d_log = HankelRankPenalty(order=2, log_transform=True)(v)
    p_raw, _ = HankelRankPenalty(order=2)(v)
    assert float(p_log) < 1e-5, f"log-domain penalty {float(p_log)}"
    assert d_log["converged_frac"] == 1.0
    assert float(p_raw) > 1e-3, f"raw penalty should see high rank: {float(p_raw)}"
    # Gradients stay finite through the transform, including at negative values.
    x = (v + 0.1 * torch.randn(6, 16)).requires_grad_(True)
    p, _ = HankelRankPenalty(order=2, log_transform=True)(x)
    p.backward()
    assert float(p) > 0 and torch.isfinite(x.grad).all()


def test_gate_excludes_offmanifold_windows():
    torch.manual_seed(2)
    pen = HankelRankPenalty(order=2, gate_threshold=0.05)
    noise = torch.randn(8, 16, requires_grad=True)
    p, d = pen(noise)
    assert d["gate_frac"] == 1.0 and d["converged_frac"] == 0.0 and float(p.detach()) == 0.0
    p.backward()
    assert torch.isfinite(noise.grad).all() and float(noise.grad.abs().sum()) == 0.0
    # Slightly-noised rank-2 windows pass the gate and give signal.
    near = (_rank2_batch(8, 16) + 0.01 * torch.randn(8, 16)).requires_grad_(True)
    p, d = pen(near)
    assert d["gate_frac"] == 0.0 and float(p) > 0
    p.backward()
    assert torch.isfinite(near.grad).all() and float(near.grad.abs().sum()) > 0


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


def test_buffer_windows_and_transitions():
    random.seed(4)
    buf = EpisodicReplayBuffer(capacity=10_000)
    ep_lens, terms = [12, 5, 30, 7], [True, False, True, False]
    _fill_buffer(buf, ep_lens, terms)
    assert len(buf) == sum(ep_lens)

    T = 5
    for _ in range(1000):
        states, actions, rewards = buf.sample_windows(1, T, exclude_terminal=True)
        e = states[0, :, 0]
        ts = states[0, :, 1]
        assert (e == e[0]).all(), "window crosses episodes"
        assert torch.equal(ts, torch.arange(ts[0], ts[0] + T, dtype=ts.dtype)), "non-consecutive"
        assert torch.equal(rewards[0], ts), "rewards misaligned"
        L, term = ep_lens[int(e[0])], terms[int(e[0])]
        if term:
            assert int(ts[-1]) < L - 1, "window ends on a terminal step"

    seen_terminal_end = False
    for _ in range(500):
        states, actions, rewards = buf.sample_windows(1, T, exclude_terminal=False)
        if int(states[0, -1, 1]) == ep_lens[int(states[0, 0, 0])] - 1:
            seen_terminal_end = True
    assert seen_terminal_end

    for _ in range(200):
        states, actions, rewards, nexts = buf.sample_transitions(16)
        for i in range(16):
            e, t = int(states[i, 0]), int(states[i, 1])
            L, term = ep_lens[e], terms[e]
            assert int(actions[i]) == t % 2 and float(rewards[i]) == float(t)
            if t == L - 1 and term:
                assert nexts[i] is None
            else:
                assert nexts[i] is not None
                assert int(nexts[i][0, 0]) == e and int(nexts[i][0, 1]) == t + 1

    # Open (in-progress) episode is sampleable for transitions, not windows.
    buf2 = EpisodicReplayBuffer(capacity=100)
    for t in range(3):
        buf2.append(torch.tensor([[9.0, float(t)]]), torch.tensor([0]),
                    torch.tensor([0.0]), torch.tensor([[9.0, float(t + 1)]]))
    assert len(buf2) == 3
    assert buf2.sample_windows(1, 2) is None
    states, actions, rewards, nexts = buf2.sample_transitions(3)
    for i in range(3):
        t = int(states[i, 1])
        assert int(nexts[i][0, 1]) == t + 1


def test_buffer_eviction():
    buf = EpisodicReplayBuffer(capacity=20)
    _fill_buffer(buf, [10, 10, 10], [True, True, True])
    assert len(buf) == 20, "oldest episode should have been evicted"
    states, _, _ = buf.sample_windows(50, 4, exclude_terminal=True)
    assert (states[:, 0, 0] >= 1).all(), "evicted episode still sampled"
    # Open episode growth also evicts closed episodes, keeping len <= capacity.
    for t in range(15):
        buf.append(torch.tensor([[9.0, float(t)]]), torch.tensor([0]),
                   torch.tensor([0.0]), torch.tensor([[9.0, float(t + 1)]]))
    assert len(buf) <= 20


def test_recency_biased_windows():
    random.seed(11)
    buf = EpisodicReplayBuffer(capacity=10_000)
    _fill_buffer(buf, [20] * 10, [False] * 10)
    from_new = sum(int(buf.sample_windows(1, 5, half_life=1.0)[0][0, 0, 0]) >= 7
                   for _ in range(1000))
    assert from_new > 800, f"half_life=1 drew only {from_new}/1000 from newest 3 episodes"
    from_new_uniform = sum(int(buf.sample_windows(1, 5)[0][0, 0, 0]) >= 7
                           for _ in range(1000))
    assert from_new_uniform < 500, f"uniform drew {from_new_uniform}/1000 from newest 3"


def test_lambda_schedules_and_v_signal():
    a = _make_agent(0.05, decay_grad_steps=100)
    a._grad_steps = 0
    assert abs(a._lambda_eff() - 0.05) < 1e-12
    a._grad_steps = 50
    assert abs(a._lambda_eff() - 0.025) < 1e-9
    a._grad_steps = 200
    assert a._lambda_eff() == 0.0
    b = _make_agent(0.05, warmup=10, ramp_grad_steps=10, decay_grad_steps=100)
    b._grad_steps = 5
    assert b._lambda_eff() == 0.0
    b._grad_steps = 20  # ramp done (k=10), decay factor 0.9
    assert abs(b._lambda_eff() - 0.05 * 0.9) < 1e-9
    v = _make_agent(0.01, seed=2, hankel_signal="v")
    _fill_agent(v)
    d = v.train()
    assert d is not None and not np.isnan(d["batch_eff_rank"])
    assert v.nan_skips == 0


def test_td_consistency_gate():
    # keep_mask plumbing: all-False mask kills the penalty and reports ext_gate_frac.
    pen = HankelRankPenalty(order=2)
    torch.manual_seed(6)
    x = torch.randn(6, 16, requires_grad=True)
    p, d = pen(x, keep_mask=torch.zeros(6, dtype=torch.bool))
    assert float(p.detach()) == 0.0 and d["ext_gate_frac"] == 1.0
    p, d = pen(x, keep_mask=torch.ones(6, dtype=torch.bool))
    assert float(p.detach()) > 0 and d["ext_gate_frac"] == 0.0
    # Agent-level: huge scale keeps everything, zero scale masks everything.
    a = _make_agent(0.01, seed=4, td_gate_scale=1e9)
    _fill_agent(a)
    d = a.train()
    assert d["ext_gate_frac"] == 0.0
    b = _make_agent(0.01, seed=4, td_gate_scale=0.0)
    _fill_agent(b)
    d = b.train()
    assert d["ext_gate_frac"] == 1.0 and d["penalty_raw"] == 0.0


def test_progress_conditioned_engagement():
    a = _make_agent(0.01, engage_reward_threshold=50.0, engage_reward_window=2)
    _fill_agent(a, n_eps=2, ep_len=12)  # episode returns 12 < 50 -> stay off
    assert a._engaged_at is None and a._lambda_eff() == 0.0
    a._grad_steps = 500
    _fill_agent(a, n_eps=2, ep_len=60)  # returns 60 -> rolling mean crosses
    assert a._engaged_at == 500
    assert a._lambda_eff() == 0.01  # no ramp: full weight from engagement
    b = _make_agent(0.01, engage_reward_threshold=50.0, engage_reward_window=2,
                    ramp_grad_steps=100)
    _fill_agent(b, n_eps=2, ep_len=60)
    assert b._engaged_at == 0 and abs(b._lambda_eff()) < 1e-12  # ramp starts at 0
    b._grad_steps = 50
    assert abs(b._lambda_eff() - 0.005) < 1e-9


def test_windows_td_includes_terminal_anchor():
    a = _make_agent(0.01, seed=1, td_source="windows")
    _fill_agent(a, n_eps=2, ep_len=12)  # episode 0 terminates
    random.seed(0)
    found_terminal = False
    for _ in range(200):
        _, _, _, nexts = a._td_batch()
        if any(x is None for x in nexts):
            found_terminal = True
            break
    assert found_terminal, "windows-mode TD never saw a terminal (target=r) anchor"


def test_atari_path_raises():
    a = _make_agent(0.0)
    try:
        a.update_buffer_atari(np.zeros(4), 0, 0.0, np.zeros(4), False)
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError:
        pass


def test_penalty_on_mps_if_available():
    if not torch.backends.mps.is_available():
        print("  (mps unavailable, skipped)")
        return
    torch.manual_seed(5)
    x = torch.randn(8, 16, device="mps", requires_grad=True)
    p, d = HankelRankPenalty(order=2)(x)
    assert p.device.type == "mps" and p.dtype == torch.float32
    p.backward()
    assert torch.isfinite(x.grad).all() and float(x.grad.abs().sum()) > 0


def _make_agent(weight, warmup=0, seed=0, **overrides):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    kwargs = dict(hankel_weight=weight, hankel_order=2, window_len=8, n_windows=4,
                  warmup_grad_steps=warmup,
                  replay_buffer_capacity=1000, q_network=TinyQNet, batch_size=16,
                  nn_learning_rate=1e-3, nn_extra_kwargs={}, env=gym.make("CartPole-v1"),
                  eps_start=1.0, eps_min=0.05, decay_rate=0.999, discount_factor=0.99,
                  device="cpu", TD_LR=0.05, buffer_util=1, gd_steps_ceil=1, double=True)
    kwargs.update(overrides)
    return HankelDQNAgent(**kwargs)


def _fill_agent(agent, n_eps=3, ep_len=12):
    rng = np.random.RandomState(7)
    for ep in range(n_eps):
        states = rng.randn(ep_len + 1, 4).astype(np.float32)
        for t in range(ep_len):
            terminated = (t == ep_len - 1) and (ep % 2 == 0)
            truncated = (t == ep_len - 1) and not terminated
            agent.update_buffer(states[t], t % 2, 1.0, states[t + 1], terminated, truncated)


def test_lambda0_matches_disabled_penalty():
    a0 = _make_agent(0.0)
    _fill_agent(a0)
    a1 = _make_agent(0.01, warmup=10 ** 9)  # penalty machinery on, lambda_eff = 0
    _fill_agent(a1)
    for k, v in a0.policy_net.state_dict().items():
        assert torch.equal(v, a1.policy_net.state_dict()[k])
    random.seed(123), torch.manual_seed(123)
    d0 = a0.train()
    random.seed(123), torch.manual_seed(123)
    d1 = a1.train()
    assert d0 is not None and d1 is not None
    for k, v in a0.policy_net.state_dict().items():
        assert torch.equal(v, a1.policy_net.state_dict()[k]), f"params diverge at {k}"
    assert np.isnan(d0["penalty_raw"]) and not np.isnan(d1["batch_eff_rank"])


def test_tiny_e2e_cartpole():
    torch.manual_seed(3), np.random.seed(3), random.seed(3)
    env = gym.make("CartPole-v1")
    agent = _make_agent(1e-2, seed=3, gate_threshold=0.5, batch_size=32,
                        gd_steps_ceil=2, env=env)
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
        assert "td_loss" in header and "penalty_raw" in header


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in fns:
        fn()
        print(f"{name} OK")
    print(f"all {len(fns)} tests passed")
