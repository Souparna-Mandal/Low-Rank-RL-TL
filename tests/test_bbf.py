"""Tests for the BBFAgent stack: Impala encoder + renormalisation, n-step/gamma
annealing with reset restarts, per-gradient-step EMA target, bias-exempt
weight decay groups, shrink-and-perturb resets, linear epsilon decay,
target-network action selection, and FHR lambda=0 inertness. CPU-only."""
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

from agents.atari_networks import ImpalaCNNEncoder     # noqa: E402
from agents.bbf_agent import BBFAgent                  # noqa: E402


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
        eps_start=1.0, eps_min=0.0, decay_rate=0.999, discount_factor=0.997,
        device="cpu", TD_LR=1.0, buffer_util=1, gd_steps_ceil=1, double=True,
        weight_decay=0.1, adam_eps=1.5e-4, amsgrad=False,
        n_step=10, n_quantiles=4, n_quantiles_target=4, n_quantiles_act=8,
        n_quantiles_fhr=4, n_cos=8, head_hidden=32,
        n_step_final=3, gamma_start=0.97, anneal_grad_steps=10,
        reset_interval_grad_steps=0, eps_decay_steps=4,
        fhr_weight=0.0, fhr_order=2, warmup_grad_steps=0)
    kwargs.update(overrides)
    return BBFAgent(**kwargs)


def _cartpole_agent(**overrides):
    env = gym.make("CartPole-v1")
    return _agent(env, MLPEncoder,
                  {"in_dim": env.observation_space.shape[0]}, **overrides), env


def _fill_buffer(agent, env, steps=60, atari=False):
    push = agent.update_buffer_atari if atari else agent.update_buffer
    env.action_space.seed(0)
    state, _ = env.reset(seed=0)
    for _ in range(steps):
        action = env.action_space.sample()
        nxt, r, term, trunc, _ = env.step(action)
        push(state, action, r, None if term else nxt, term, trunc)
        state = nxt
        if term or trunc:
            state, _ = env.reset()


# ------------------------------------------------------------- Impala encoder
def test_impala_encoder_shapes_and_renormalize():
    enc = ImpalaCNNEncoder(in_channels=1, width_scale=1)
    assert enc.feature_dim == 32 * 11 * 11          # 84 -> 42 -> 21 -> 11
    x = torch.randint(0, 256, (3, 1, 84, 84), dtype=torch.uint8)
    out = enc(x)
    assert out.shape == (3, enc.feature_dim)
    # per-sample min-max renormalisation to [0, 1]
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    for row in out:
        assert float(row.max()) == pytest.approx(1.0, abs=1e-3)
    raw = ImpalaCNNEncoder(in_channels=1, width_scale=1, renormalize=False)
    assert raw(x.float()).shape == (3, raw.feature_dim)
    # BBF width: 4x -> (64, 128, 128) channels, 128 * 11 * 11 features
    assert ImpalaCNNEncoder(in_channels=4).feature_dim == 128 * 11 * 11
    # 15 conv layers total (3 stages x [1 + 2 blocks x 2])
    n_convs = sum(1 for m in enc.modules() if isinstance(m, nn.Conv2d))
    assert n_convs == 15


# ----------------------------------------------------------------- schedules
def test_schedules_anneal_and_restart():
    agent, env = _cartpole_agent(anneal_grad_steps=10)
    assert agent._current_gamma() == pytest.approx(0.97)
    assert agent._current_n_step() == 10
    agent._grad_steps = 5
    mid_gamma, mid_n = agent._current_gamma(), agent._current_n_step()
    assert 0.97 < mid_gamma < 0.997 and 3 <= mid_n <= 10
    # log-linear in (1 - gamma): halfway = geometric mean of 0.03 and 0.003
    assert 1 - mid_gamma == pytest.approx((0.03 * 0.003) ** 0.5)
    assert mid_n == round(10 * (3 / 10) ** 0.5)
    agent._grad_steps = 10
    assert agent._current_gamma() == pytest.approx(0.997)
    assert agent._current_n_step() == 3
    agent._grad_steps = 500                        # stays at final values
    assert agent._current_gamma() == pytest.approx(0.997)
    # a reset restarts the cycle
    agent._last_reset = 500
    assert agent._current_gamma() == pytest.approx(0.97)
    assert agent._current_n_step() == 10
    env.close()


# --------------------------------------------------------------- EMA target
def test_ema_target_per_grad_step_and_noop_loop_tick():
    agent, env = _cartpole_agent(target_ema_tau=0.1)
    _fill_buffer(agent, env, steps=50)
    t0 = {k: v.clone() for k, v in agent.target_net.state_dict().items()}
    # loop tick must NOT touch the target (EMA rides the gradient step)
    agent.update_target_network()
    for k, v in agent.target_net.state_dict().items():
        assert torch.equal(v, t0[k])
    # lr 0: policy stays fixed, so the EMA formula is exactly checkable
    for g in agent.optimiser.param_groups:
        g["lr"] = 0.0
    agent._train_step()
    p = agent.policy_net.state_dict()
    for k, v in agent.target_net.state_dict().items():
        expected = (1 - 0.1) * t0[k] + 0.1 * p[k]
        assert torch.allclose(v, expected, atol=1e-6), k
    env.close()


# ----------------------------------------------------------- optimizer groups
def test_bias_exempt_weight_decay_groups():
    agent, env = _cartpole_agent(c_learning_rate=0.02)
    g_decay, g_plain, g_c = agent.optimiser.param_groups
    assert g_decay["weight_decay"] == pytest.approx(0.1)
    assert all(p.ndim >= 2 for p in g_decay["params"])
    assert g_plain["weight_decay"] == 0.0
    assert all(p.ndim < 2 for p in g_plain["params"])
    grouped = {id(p) for p in g_decay["params"]} | {id(p) for p in g_plain["params"]}
    assert grouped == {id(p) for p in agent.policy_net.parameters()}
    # c/d group stays last: own lr, no decay, excluded from the policy params
    assert g_c["lr"] == 0.02 and g_c["weight_decay"] == 0.0
    assert g_c["params"][0] is agent.c
    assert g_decay["eps"] == pytest.approx(1.5e-4) and g_decay["amsgrad"] is False
    env.close()


# ------------------------------------------------------------------- resets
def test_shrink_perturb_reset_semantics():
    agent, env = _cartpole_agent()
    _fill_buffer(agent, env, steps=50)
    agent._train_step()                            # optimizer state exists
    old = {k: v.clone() for k, v in agent.policy_net.state_dict().items()}
    fresh = agent._net_factory()
    for p in fresh.parameters():
        torch.nn.init.zeros_(p)
    agent._net_factory = lambda: fresh
    agent._grad_steps = 7
    agent._shrink_perturb_reset()
    for name, p in agent.policy_net.named_parameters():
        if name.startswith("encoder."):
            assert torch.allclose(p, 0.5 * old[name], atol=1e-7), name
        else:
            assert torch.equal(p, torch.zeros_like(p)), name
    # target got the identical treatment -> reset head agrees exactly
    for name, p in agent.target_net.named_parameters():
        if not name.startswith("encoder."):
            assert torch.equal(p, torch.zeros_like(p)), name
    # Adam moments kept for the encoder, zeroed for the head
    head = {id(p) for n, p in agent.policy_net.named_parameters()
            if not n.startswith("encoder.")}
    enc = {id(p) for n, p in agent.policy_net.named_parameters()
           if n.startswith("encoder.")}
    state_ids = {id(p) for p in agent.optimiser.state}
    assert not (state_ids & head) and (state_ids & enc)
    assert agent._last_reset == 7 and agent.reset_count == 1
    env.close()


def test_reset_fires_during_training_and_restarts_schedule():
    agent, env = _cartpole_agent(reset_interval_grad_steps=3,
                                 anneal_grad_steps=100)
    _fill_buffer(agent, env, steps=50)
    for _ in range(7):
        agent._train_step()
    assert agent.reset_count >= 2
    assert agent._grad_steps - agent._last_reset < 3
    assert agent._current_n_step() >= 9            # cycle recently restarted
    env.close()


def test_no_resets_after_grad_steps_blocks_late_resets():
    agent, env = _cartpole_agent(reset_interval_grad_steps=3,
                                 no_resets_after_grad_steps=4)
    _fill_buffer(agent, env, steps=50)
    for _ in range(12):
        agent._train_step()
    assert agent.reset_count == 1                  # only the reset at step 3
    assert agent._last_reset == 3
    env.close()


# ------------------------------------------------------------- exploration
def test_linear_epsilon_decay():
    agent, env = _cartpole_agent(eps_decay_steps=4)
    seen = []
    for _ in range(6):
        agent.decay_epsilon()
        seen.append(agent.epsilon)
    assert seen == pytest.approx([0.75, 0.5, 0.25, 0.0, 0.0, 0.0])
    env.close()


def test_target_action_selection():
    agent, env = _cartpole_agent(target_action_selection=True)
    agent.policy_net.forward = lambda x, n_taus=None: torch.tensor([[9.0, 0.0]])
    agent.target_net.forward = lambda x, n_taus=None: torch.tensor([[0.0, 9.0]])
    state = torch.zeros(1, env.observation_space.shape[0])
    assert agent.act_greedy(state) == 1            # target net decides
    agent.target_action_selection = False
    assert agent.act_greedy(state) == 0
    env.close()


# ---------------------------------------------------- FHR penalty integration
def test_lambda0_matches_penalty_branch_disabled():
    def run(disable_penalty):
        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        agent, env = _cartpole_agent(fhr_weight=0.0,
                                     reset_interval_grad_steps=0)
        _fill_buffer(agent, env, steps=50)
        if disable_penalty:
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


# ------------------------------------------------------------------ e2e smoke
def test_e2e_smoke_with_penalty_augmentation_and_diagnostics():
    torch.manual_seed(0); random.seed(0)
    env = FakeImageEnv()
    agent = _agent(env, ImageEncoder, {}, fhr_weight=0.5, fhr_order=2,
                   use_augmentation=True, warmup_grad_steps=2, batch_size=8,
                   reset_interval_grad_steps=5, anneal_grad_steps=6)
    _fill_buffer(agent, env, steps=50, atari=True)
    stored = agent.replay_buffer._episodes[0]["states"]
    assert stored.dtype == torch.uint8 and stored.device.type == "cpu"
    for _ in range(8):
        diag = agent.train()
    assert np.isfinite(diag["td_loss"]) and agent.nan_skips == 0
    assert np.isfinite(diag["penalty_raw"])        # penalty active post warm-up
    for key in ("gamma_eff", "n_step_eff", "resets"):
        assert key in diag
    assert diag["resets"] >= 1.0                   # a reset happened mid-run
    assert agent.pi(env.reset()[0]) in range(env.action_space.n)


def test_save_load_roundtrip_carries_reset_state(tmp_path):
    agent, env = _cartpole_agent(fhr_weight=0.2)
    _fill_buffer(agent, env, steps=40)
    agent.train()
    agent._last_reset, agent.reset_count = 5, 2
    agent.save(tmp_path / "ckpt.pt")
    agent2, env2 = _cartpole_agent(fhr_weight=0.2)
    agent2.load(tmp_path / "ckpt.pt")
    for k, v in agent.policy_net.state_dict().items():
        assert torch.equal(v, agent2.policy_net.state_dict()[k])
    assert torch.equal(agent.c.detach(), agent2.c.detach())
    assert agent2._grad_steps == agent._grad_steps
    assert agent2._last_reset == 5 and agent2.reset_count == 2
    env.close(); env2.close()
