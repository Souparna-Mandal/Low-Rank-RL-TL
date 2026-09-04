"""Tests for the EfficientRainbowAgent stack: sample-time n-step targets on the
episodic buffer, DrQ augmentation, the noisy-free fixed-tau IQN head, FHR
penalty integration (offset sharing, lambda=0 inertness), and the optimizer
kwargs. CPU-only; CartPole and a tiny fake image env stand in for Atari."""
import pathlib
import random
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from agents import efficient_rainbow_agent as era_module          # noqa: E402
from agents.augmentation import intensity, random_shift           # noqa: E402
from agents.efficient_rainbow_agent import EfficientRainbowAgent  # noqa: E402
from agents.hankel_dqn_agent import EpisodicReplayBuffer          # noqa: E402
from agents.q_agent import QAgent                                 # noqa: E402
from agents.rainbow_agent import (NoisyLinear, RainbowIQNNetwork)  # noqa: E402


# --------------------------------------------------------------- helpers
class MLPEncoder(nn.Module):
    def __init__(self, in_dim, feature_dim=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, feature_dim), nn.ReLU())
        self.feature_dim = feature_dim

    def forward(self, x):
        return self.net(x)


class ImageEncoder(nn.Module):
    """Flatten + Linear over (1, 84, 84) frames — conv-free for CPU speed."""
    def __init__(self, feature_dim=16):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(84 * 84, feature_dim), nn.ReLU())
        self.feature_dim = feature_dim

    def forward(self, x):
        return self.net(x.float() / 255.0)


class FakeImageEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 255, shape=(1, 84, 84), dtype=np.uint8)
    action_space = gym.spaces.Discrete(3)

    def __init__(self, ep_len=12):
        self.ep_len = ep_len
        self._t = 0
        self._rng = np.random.default_rng(0)

    def _obs(self):
        return self._rng.integers(0, 256, size=(1, 84, 84), dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        return self._obs(), 1.0, self._t >= self.ep_len, False, {}


def _agent(env, encoder_cls, encoder_kwargs, **overrides):
    kwargs = dict(
        replay_buffer_capacity=1000, q_network=encoder_cls, batch_size=8,
        nn_learning_rate=1e-3, nn_extra_kwargs=encoder_kwargs, env=env,
        eps_start=1.0, eps_min=0.05, decay_rate=0.999, discount_factor=0.99,
        device="cpu", TD_LR=1.0, buffer_util=1, gd_steps_ceil=1, double=True,
        n_step=3, n_quantiles=4, n_quantiles_target=4, n_quantiles_act=8,
        n_quantiles_fhr=4, n_cos=8, head_hidden=32,
        fhr_weight=0.0, fhr_order=2, warmup_grad_steps=0)
    kwargs.update(overrides)
    return EfficientRainbowAgent(**kwargs)


def _cartpole_agent(**overrides):
    env = gym.make("CartPole-v1")
    return _agent(env, MLPEncoder,
                  {"in_dim": env.observation_space.shape[0]}, **overrides), env


def _fill_buffer(agent, env, steps=60, atari=False):
    push = agent.update_buffer_atari if atari else agent.update_buffer
    env.action_space.seed(0)          # the space RNG is separate from reset's
    state, _ = env.reset(seed=0)
    for _ in range(steps):
        action = env.action_space.sample()
        nxt, r, term, trunc, _ = env.step(action)
        push(state, action, r, None if term else nxt, term, trunc)
        state = nxt
        if term or trunc:
            state, _ = env.reset()


# --------------------------------------------- n-step sampling on the buffer
def _toy_buffer(gamma=0.5):
    """One closed terminated episode with rewards r_t = t (t = 0..4)."""
    buf = EpisodicReplayBuffer(100)
    for t in range(5):
        s = torch.full((1, 2), float(t))
        nxt = None if t == 4 else torch.full((1, 2), float(t + 1))
        buf.append(s, torch.tensor([t % 2]), torch.tensor([float(t)]), nxt)
    buf.close(terminated=True)
    return buf


def test_nstep_mid_episode_and_tails():
    gamma = 0.5
    buf = _toy_buffer(gamma)
    # mid-episode full window at t=0, n=3: R = 0 + 0.5*1 + 0.25*2, boot s_3
    s, a, R, nxt, disc = buf._nstep_transition(0, 0, 3, gamma)
    assert R.item() == pytest.approx(0 + 0.5 * 1 + 0.25 * 2)
    assert disc == pytest.approx(gamma ** 3)
    assert torch.equal(nxt, torch.full((1, 2), 3.0))
    # terminal tail at t=3, n=3: only 2 steps remain -> R = 3 + 0.5*4, next None
    s, a, R, nxt, disc = buf._nstep_transition(0, 3, 3, gamma)
    assert R.item() == pytest.approx(3 + 0.5 * 4)
    assert nxt is None and disc == pytest.approx(gamma ** 2)


def test_nstep_truncated_and_open_tails():
    gamma = 0.5
    buf = EpisodicReplayBuffer(100)
    final = torch.full((1, 2), 99.0)
    for t in range(3):
        s = torch.full((1, 2), float(t))
        buf.append(s, torch.tensor([0]), torch.tensor([1.0]),
                   final if t == 2 else torch.full((1, 2), float(t + 1)))
    buf.close(terminated=False)          # truncated: bootstrap survives
    s, a, R, nxt, disc = buf._nstep_transition(0, 1, 5, gamma)
    assert R.item() == pytest.approx(1 + 0.5)      # two remaining unit rewards
    assert torch.equal(nxt, final) and disc == pytest.approx(gamma ** 2)
    # open episode: newest window bootstraps from _pending_next
    pending = torch.full((1, 2), 7.0)
    buf.append(torch.zeros(1, 2), torch.tensor([0]), torch.tensor([2.0]), pending)
    s, a, R, nxt, disc = buf._nstep_transition(1, 0, 4, gamma)
    assert R.item() == pytest.approx(2.0)
    assert torch.equal(nxt, pending) and disc == pytest.approx(gamma)


def test_nstep1_matches_sample_transitions_rng_and_values():
    buf = _toy_buffer()
    random.seed(123)
    s1, a1, r1, n1, h1 = buf.sample_transitions(4, with_handles=True)
    random.seed(123)
    s2, a2, R2, n2, d2, h2 = buf.sample_nstep_transitions(
        4, n_step=1, gamma=0.5, with_handles=True)
    assert h1 == h2                                   # identical RNG draw
    assert torch.equal(s1, s2) and torch.equal(a1, a2)
    assert torch.allclose(r1, R2)                     # 1-step return = reward
    for x, y in zip(n1, n2):
        assert (x is None and y is None) or torch.equal(x, y)
    assert torch.allclose(d2, torch.full((4,), 0.5))  # gamma^1 everywhere


# ----------------------------------------------------------- augmentation
def test_random_shift_bounds_border_and_replay():
    torch.manual_seed(0)
    x = torch.rand(6, 2, 10, 10) * 255
    aug, offs = random_shift(x, pad=3)
    assert aug.shape == x.shape
    assert offs.shape == (6, 2) and offs.min() >= 0 and offs.max() <= 6
    replay, _ = random_shift(x, pad=3, offsets=offs)
    assert torch.equal(aug, replay)                   # offsets replay exactly
    ident, _ = random_shift(x, pad=3, offsets=torch.full((6, 2), 3, dtype=torch.long))
    assert torch.equal(ident, x)                      # offset == pad is identity
    # shifted content matches a manual crop of the replicate-padded input
    shifted, _ = random_shift(x, pad=3, offsets=torch.zeros(6, 2, dtype=torch.long))
    padded = torch.nn.functional.pad(x, (3, 3, 3, 3), mode="replicate")
    assert torch.equal(shifted, padded[:, :, 0:10, 0:10])
    with pytest.raises(ValueError):
        random_shift(x, pad=3, offsets=torch.full((6, 2), 7, dtype=torch.long))


def test_intensity_replay_and_clip():
    torch.manual_seed(0)
    x = torch.ones(64, 1, 4, 4)
    aug, factors = intensity(x, scale=0.1)
    assert aug.shape == x.shape
    assert factors.min() >= 1 - 0.2 - 1e-6 and factors.max() <= 1 + 0.2 + 1e-6
    replay, _ = intensity(x, scale=0.1, factors=factors)
    assert torch.equal(aug, replay)


# ------------------------------------------------------------------- head
def test_head_noisy_free_fixed_taus_contract():
    torch.manual_seed(0)
    net = RainbowIQNNetwork(MLPEncoder(4), 16, 3, n_cos=8, head_hidden=32,
                            dueling=True, noisy=False, fixed_act_taus=True,
                            n_quantiles_act=8)
    assert not any(isinstance(m, NoisyLinear) for m in net.modules())
    x = torch.randn(5, 4)
    q1, q2 = net(x), net(x)
    assert q1.shape == (5, 3)
    assert torch.equal(q1, q2)                        # deterministic given weights
    grid = (torch.arange(8, dtype=torch.float32) + 0.5) / 8
    manual = net.quantiles(x, grid.expand(5, 8)).mean(dim=1)
    assert torch.allclose(q1, manual)
    v, a = net.value_advantage(x)
    assert v.shape == (5, 1) and a.shape == (5, 3)
    assert torch.allclose(a.mean(dim=1), torch.zeros(5), atol=1e-6)
    # default construction keeps the noisy Rainbow behaviour
    noisy_net = RainbowIQNNetwork(MLPEncoder(4), 16, 3, n_cos=8, head_hidden=32)
    assert any(isinstance(m, NoisyLinear) for m in noisy_net.modules())
    assert noisy_net.fixed_act_taus is False


def test_target_quantiles_per_sample_discount():
    torch.manual_seed(0)
    agent, env = _cartpole_agent(batch_size=2, n_quantiles_target=3)
    agent.policy_net._sample_taus = lambda b, n, d: torch.full((b, n), 0.5)
    rewards = torch.tensor([1.0, 2.0])
    discounts = torch.tensor([0.9, 0.5])
    non_final = torch.tensor([True, False])
    next_states = torch.randn(1, env.observation_space.shape[0])
    target = agent._target_quantiles(rewards, discounts, non_final, next_states)
    taus = torch.full((1, 3), 0.5)
    if agent.double:
        na = agent.policy_net(next_states, n_taus=agent.n_quantiles_select).argmax(1)
    nf_q = agent.target_net.quantiles(next_states, taus)
    boot = nf_q.gather(2, na.view(-1, 1, 1).expand(-1, 3, 1)).squeeze(2)
    assert torch.allclose(target[0], 1.0 + 0.9 * boot[0])
    assert torch.allclose(target[1], torch.full((3,), 2.0))   # terminal row: R only
    env.close()


# ---------------------------------------------------- FHR penalty integration
def test_penalty_sequences_share_augmentation_offsets(monkeypatch):
    torch.manual_seed(0)
    env = FakeImageEnv()
    agent = _agent(env, ImageEncoder, {}, fhr_weight=0.5, fhr_order=2,
                   use_augmentation=True, aug_pad=4, aug_intensity=0.05,
                   batch_size=8, warmup_grad_steps=0)
    _fill_buffer(agent, env, steps=40, atari=True)

    calls = []
    real_shift = era_module.random_shift

    def spy(x, pad=4, offsets=None):
        out = real_shift(x, pad, offsets=offsets)
        calls.append((x.shape[0], offsets, out[1]))
        return out

    monkeypatch.setattr(era_module, "random_shift", spy)
    random.seed(0)
    diag = agent._train_step()
    assert diag["b_h"] > 0                       # the penalty actually ran
    # calls: TD states (offsets drawn), TD next states (drawn), penalty (given)
    penalty_calls = [c for c in calls if c[1] is not None]
    assert len(penalty_calls) == 1
    n_rows, given, _ = penalty_calls[0]
    r = agent.fhr_order
    assert n_rows % (r + 1) == 0
    groups = given.view(-1, r + 1, 2)
    # every anchor and its r lags got the SAME offset
    assert bool((groups == groups[:, :1, :]).all())
    # TD calls drew their own independent offsets
    td_calls = [c for c in calls if c[1] is None]
    assert len(td_calls) == 2


def test_lambda0_is_inert_and_matches_pure_iqn_loss():
    torch.manual_seed(0)
    agent, env = _cartpole_agent(fhr_weight=0.0)
    _fill_buffer(agent, env, steps=50)
    c0 = agent.c.detach().clone()

    # deterministic taus so the loss can be recomputed independently
    agent.policy_net._sample_taus = lambda b, n, d: (
        (torch.arange(n, dtype=torch.float32) + 0.5) / n).expand(b, n)
    captured = {}
    orig = agent.replay_buffer.sample_nstep_transitions

    def record(*args, **kw):
        out = orig(*args, **kw)
        captured["batch"] = out
        return out

    agent.replay_buffer.sample_nstep_transitions = record
    # lr 0 everywhere: the step leaves the weights untouched, so the loss can
    # be recomputed exactly from the captured batch afterwards
    for g in agent.optimiser.param_groups:
        g["lr"] = 0.0
    diag = agent._train_step()

    # penalty machinery untouched
    assert np.isnan(diag["penalty_raw"]) and diag["penalty_weighted"] == 0.0
    assert np.isnan(diag["b_h"])
    assert torch.equal(agent.c.detach(), c0)
    assert agent.c.grad is None

    # recompute the pure IQN n-step loss from the captured batch
    states, actions, returns, next_list, discounts, handles = captured["batch"]
    non_final = torch.tensor([s is not None for s in next_list])
    with torch.no_grad():
        next_states = torch.cat([s for s in next_list if s is not None])
        target = agent._target_quantiles(returns, discounts, non_final, next_states)
        taus = agent.policy_net._sample_taus(agent.batch_size, agent.n_quantiles, "cpu")
        theta = agent.policy_net.quantiles(states, taus)
        theta_a = theta.gather(
            2, actions.view(-1, 1, 1).expand(-1, agent.n_quantiles, 1)).squeeze(2)
        td = target.unsqueeze(1) - theta_a.unsqueeze(2)
        expected = agent._quantile_huber(td, taus).mean(dim=2).sum(dim=1).mean()
    assert diag["td_loss"] == pytest.approx(float(expected), rel=1e-6)
    env.close()


def test_lambda0_matches_penalty_branch_disabled():
    def run(disable_penalty):
        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        agent, env = _cartpole_agent(fhr_weight=0.0)
        _fill_buffer(agent, env, steps=50)
        if disable_penalty:
            # fhr_weight=0 must never enter the penalty block: make entering fatal
            agent.replay_buffer.gather_predecessors = lambda *a, **k: (
                (_ for _ in ()).throw(AssertionError("penalty branch entered")))
        random.seed(11); torch.manual_seed(11)
        for _ in range(3):
            agent.train()
        state = {k: v.clone() for k, v in agent.policy_net.state_dict().items()}
        env.close()
        return state

    a, b = run(False), run(True)
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_c_group_isolated_from_clipping_and_decay():
    agent, env = _cartpole_agent(c_learning_rate=0.02)
    last = agent.optimiser.param_groups[-1]
    assert last["lr"] == 0.02 and last["weight_decay"] == 0.0
    assert last["params"][0] is agent.c
    policy_params = {id(p) for p in agent.policy_net.parameters()}
    assert id(agent.c) not in policy_params
    env.close()


def test_diagnostics_keys_stable_and_match_fhrdqn():
    from agents.fhrdqn_agent import FHRDQNAgent

    class TinyNet(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(in_dim, 8), nn.ReLU(),
                                     nn.Linear(8, out_dim))

        def forward(self, x):
            return self.net(x)

    env = gym.make("CartPole-v1")
    fhr = FHRDQNAgent(
        replay_buffer_capacity=500, q_network=TinyNet, batch_size=8,
        nn_learning_rate=1e-3,
        nn_extra_kwargs={"in_dim": 4, "out_dim": env.action_space.n}, env=env,
        eps_start=1.0, eps_min=0.05, decay_rate=0.999, discount_factor=0.99,
        device="cpu", TD_LR=1.0, buffer_util=1, gd_steps_ceil=1, double=True,
        fhr_weight=0.3, fhr_order=2, warmup_grad_steps=0)
    era, env2 = _cartpole_agent(fhr_weight=0.3, fhr_order=2)
    _fill_buffer(fhr, env, steps=40)
    _fill_buffer(era, env2, steps=40)
    d_fhr, d_era = fhr._train_step(), era._train_step()
    assert set(d_fhr) == set(d_era)                  # byte-identical key sets
    assert set(era._train_step()) == set(d_era)      # stable across calls
    env.close(); env2.close()


def test_save_load_roundtrip(tmp_path):
    agent, env = _cartpole_agent(fhr_weight=0.2)
    _fill_buffer(agent, env, steps=40)
    agent.train()
    with torch.no_grad():
        agent.c += 0.123
    agent.save(tmp_path / "ckpt.pt")
    agent2, env2 = _cartpole_agent(fhr_weight=0.2)
    agent2.load(tmp_path / "ckpt.pt")
    for k, v in agent.policy_net.state_dict().items():
        assert torch.equal(v, agent2.policy_net.state_dict()[k])
    assert torch.equal(agent.c.detach(), agent2.c.detach())
    env.close(); env2.close()


def test_prioritized_replay_flag_reserved():
    with pytest.raises(NotImplementedError):
        _cartpole_agent(prioritized_replay=True)


def test_e2e_smoke_with_penalty_and_augmentation():
    """Image env end-to-end: uint8 episodic storage, n-step, aug on, penalty on."""
    torch.manual_seed(0); random.seed(0)
    env = FakeImageEnv()
    agent = _agent(env, ImageEncoder, {}, fhr_weight=0.5, fhr_order=2,
                   use_augmentation=True, warmup_grad_steps=2, batch_size=8)
    _fill_buffer(agent, env, steps=50, atari=True)
    stored = agent.replay_buffer._episodes[0]["states"]
    assert stored.dtype == torch.uint8 and stored.device.type == "cpu"
    for _ in range(4):
        diag = agent.train()
    assert np.isfinite(diag["td_loss"]) and agent.nan_skips == 0
    assert np.isfinite(diag["penalty_raw"])          # penalty active post warm-up
    assert agent.pi(env.reset()[0]) in range(env.action_space.n)


# ------------------------------------------------------------ optimizer kwargs
def test_qagent_adam_kwargs_and_defaults():
    class TinyNet(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.net = nn.Linear(in_dim, out_dim)

        def forward(self, x):
            return self.net(x)

    env = gym.make("CartPole-v1")
    base = dict(replay_buffer_capacity=100, q_network=TinyNet, batch_size=8,
                nn_learning_rate=1e-3,
                nn_extra_kwargs={"in_dim": 4, "out_dim": 2}, env=env,
                eps_start=1.0, eps_min=0.05, decay_rate=0.999,
                discount_factor=0.99, device="cpu", TD_LR=1.0)
    legacy = QAgent(**base)
    g = legacy.optimiser.param_groups[0]
    assert g["weight_decay"] == 0.01 and g["eps"] == 1e-8 and g["amsgrad"] is True
    tuned = QAgent(**base, weight_decay=0.0, adam_eps=1.5e-4, amsgrad=False)
    g = tuned.optimiser.param_groups[0]
    assert g["weight_decay"] == 0.0 and g["eps"] == pytest.approx(1.5e-4)
    assert g["amsgrad"] is False
    env.close()


def test_fhr_lag_source_split_path_matches_fused_at_init():
    """detached/target run the split anchor/lag forward; at init (nets synced,
    no augmentation) the first-step penalty value must match the fused online
    path, and the step must complete cleanly."""
    diags = {}
    for src in ("online", "detached", "target"):
        torch.manual_seed(0), np.random.seed(0), random.seed(0)
        agent, env = _cartpole_agent(fhr_weight=0.5, fhr_lag_source=src)
        _fill_buffer(agent, env)
        torch.manual_seed(123), np.random.seed(123), random.seed(123)
        diags[src] = agent.train()
        assert agent.nan_skips == 0
        env.close()
    assert diags["online"]["b_h"] > 0
    p0 = diags["online"]["penalty_raw"]
    assert np.isfinite(p0)
    assert abs(diags["detached"]["penalty_raw"] - p0) < 1e-5
    assert abs(diags["target"]["penalty_raw"] - p0) < 1e-5
