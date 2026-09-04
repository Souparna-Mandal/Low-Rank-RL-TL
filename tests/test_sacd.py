"""Tests for the native SAC-Discrete stack: SACDiscreteAgent (exact-expectation
soft targets, temperature machinery, FHR penalty on twin critics, ccond head,
gradient routing), and PrioritizedEpisodicReplayBuffer (sum tree over the
episodic store, whole-episode eviction, handle validity). CPU-only, tiny nets."""
import math
import pathlib
import random
import sys
import tempfile

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from agents.hankel_dqn_agent import EpisodicReplayBuffer          # noqa: E402
from agents.per_episodic_buffer import PrioritizedEpisodicReplayBuffer  # noqa: E402
from agents.sac_discrete_agent import (CCondHead,                 # noqa: E402
                                       SACDiscreteAgent,
                                       SACDiscreteNetwork)
from analysis.run_logger import RunLogger                         # noqa: E402
from training import dqn_training_loop, evaluate_policy_atari     # noqa: E402


# --------------------------------------------------------------- helpers
class MLPEncoder(nn.Module):
    def __init__(self, in_dim, feature_dim=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, feature_dim), nn.ReLU())
        self.feature_dim = feature_dim

    def forward(self, x):
        return self.net(x)


class ImageEncoder(nn.Module):
    def __init__(self, feature_dim=16):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.Linear(84 * 84, feature_dim), nn.ReLU())
        self.feature_dim = feature_dim

    def forward(self, x):
        return self.net(x.float() / 255.0)


class FakeImageEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 255, shape=(1, 84, 84),
                                       dtype=np.uint8)
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


def _seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _agent(env=None, encoder_cls=MLPEncoder, encoder_kwargs=None, **over):
    env = env or gym.make("CartPole-v1")
    if encoder_kwargs is None:
        encoder_kwargs = {"in_dim": int(np.prod(env.observation_space.shape))}
    kwargs = dict(
        replay_buffer_capacity=1000, q_network=encoder_cls, batch_size=8,
        nn_learning_rate=1e-3, nn_extra_kwargs=encoder_kwargs, env=env,
        eps_start=0.0, eps_min=0.0, decay_rate=1.0, discount_factor=0.99,
        device="cpu", TD_LR=1.0, buffer_util=1, gd_steps_ceil=1,
        n_step=3, head_hidden=32, fhr_weight=0.0, fhr_order=2,
        warmup_grad_steps=0)
    kwargs.update(over)
    return SACDiscreteAgent(**kwargs)


def _fill(agent, env, n_steps=120, seed=1):
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    for _ in range(n_steps):
        a = env.action_space.sample()
        nxt, r, term, trunc, _ = env.step(a)
        agent.update_buffer(obs, a, r, nxt, term, trunc)
        obs = nxt
        if term or trunc:
            obs, _ = env.reset()


# ---------------------------------------------------- 1. construction
def test_optimizer_groups_and_target_entropy():
    _seed_all(0)
    agent = _agent(actor_learning_rate=2e-4, alpha_learning_rate=3e-4,
                   c_learning_rate=0.03, target_entropy_scale=0.89)
    groups = agent.optimiser.param_groups
    assert len(groups) == 4
    critic_ids = {id(p) for p in groups[0]["params"]}
    assert critic_ids == {id(p) for p in agent.policy_net.encoder.parameters()} \
        | {id(p) for p in agent.policy_net.q1.parameters()} \
        | {id(p) for p in agent.policy_net.q2.parameters()}
    assert {id(p) for p in groups[1]["params"]} == \
        {id(p) for p in agent.policy_net.actor.parameters()}
    assert groups[1]["lr"] == pytest.approx(2e-4)
    assert groups[2]["params"] == [agent.log_alpha]
    assert groups[2]["lr"] == pytest.approx(3e-4)
    assert groups[2]["weight_decay"] == 0.0
    # c group LAST — the invariant every FHR test pins
    assert any(p is agent.c for p in groups[3]["params"])
    assert groups[3]["lr"] == pytest.approx(0.03)
    assert groups[3]["weight_decay"] == 0.0
    assert agent.target_entropy == pytest.approx(0.89 * math.log(2))
    # log_alpha / c never reachable from grad clipping (policy_net params)
    net_ids = {id(p) for p in agent.policy_net.parameters()}
    assert id(agent.log_alpha) not in net_ids
    assert id(agent.c) not in net_ids


def test_network_forward_is_min_twin():
    _seed_all(1)
    net = SACDiscreteNetwork(MLPEncoder(4), 16, 2, head_hidden=8)
    x = torch.randn(5, 4)
    phi = net.features(x)
    q1, q2 = net.critic_values(phi)
    assert torch.equal(net(x), torch.min(q1, q2))


# ---------------------------------------------------- 2. exact soft target
def test_soft_target_exact_expectation():
    _seed_all(2)
    agent = _agent()
    B = 4
    returns = torch.tensor([1.0, 2.0, 0.5, -1.0])
    discounts = torch.tensor([0.99 ** 3, 0.99 ** 3, 0.99, 0.99 ** 2])
    non_final = torch.tensor([True, False, True, True])
    nxt = torch.randn(3, 4)          # non-final rows only
    y = agent._soft_target(returns, discounts, non_final, nxt)

    with torch.no_grad():
        alpha = float(agent.log_alpha.exp())
        logits = agent.policy_net.actor_logits(agent.policy_net.features(nxt))
        p = torch.softmax(logits, dim=1)
        logp = torch.log_softmax(logits, dim=1)
        q1t, q2t = agent.target_net.critic_values(
            agent.target_net.features(nxt))
        v = (p * (torch.min(q1t, q2t) - alpha * logp)).sum(1)
    expected = returns.clone()
    idx = 0
    for i in range(B):
        if non_final[i]:
            expected[i] += discounts[i] * v[idx]
            idx += 1
    assert torch.allclose(y, expected, atol=1e-6)


def test_alpha_loss_direction():
    def delta(bias):
        _seed_all(3)
        env = gym.make("CartPole-v1")
        agent = _agent(env)
        with torch.no_grad():
            final = agent.policy_net.actor[-1]
            final.weight.zero_()
            final.bias.copy_(torch.tensor(bias))
        _fill(agent, env, 60)
        before = float(agent.log_alpha.detach())
        agent._train_step()
        return float(agent.log_alpha.detach()) - before

    assert delta([8.0, -8.0]) > 0     # peaked: H ~ 0 < target -> alpha up
    assert delta([0.0, 0.0]) < 0      # uniform: H = log2 > target -> down


def test_gradient_routing():
    """Critic loss reaches the encoder; the actor loss cannot (phi.detach);
    alpha only moves from the alpha loss; c only when the penalty is live."""
    _seed_all(4)
    env = gym.make("CartPole-v1")
    agent = _agent(env, fhr_weight=1.0, warmup_grad_steps=0)
    _fill(agent, env, 60)
    agent._train_step()
    enc_grads = [p.grad for p in agent.policy_net.encoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in enc_grads)
    assert agent.log_alpha.grad is not None
    assert agent.c.grad is not None and float(agent.c.grad.abs().sum()) > 0

    # actor-only backward cannot reach the encoder
    _seed_all(4)
    agent2 = _agent(env, fhr_weight=0.0)
    _fill(agent2, env, 60)
    s = agent2.replay_buffer.sample_transitions(8)[0]
    phi = agent2.policy_net.features(s)
    logits = agent2.policy_net.actor_logits(phi.detach())
    probs, log_probs = SACDiscreteNetwork.policy(logits)
    min_q = agent2.policy_net(s).detach()
    actor_loss = (probs * (0.2 * log_probs - min_q)).sum(1).mean()
    actor_loss.backward()
    assert all(p.grad is None for p in agent2.policy_net.encoder.parameters())
    assert all(p.grad is not None for p in agent2.policy_net.actor.parameters())


# ---------------------------------------------------- 3. FHR integration
def test_fhr_warmup_and_engagement():
    _seed_all(5)
    env = gym.make("CartPole-v1")
    agent = _agent(env, fhr_weight=0.7, warmup_grad_steps=10 ** 6)
    _fill(agent, env, 80)
    c0 = agent.c.detach().clone()
    diag = agent._train_step()
    assert diag["lambda_eff"] == 0.0
    assert np.isfinite(diag["penalty_raw"])       # diagnosed, out of graph
    assert torch.equal(agent.c.detach(), c0)

    agent2 = _agent(env, fhr_weight=0.7, warmup_grad_steps=0)
    _fill(agent2, env, 80)
    c0 = agent2.c.detach().clone()
    diag = agent2._train_step()
    assert diag["lambda_eff"] == pytest.approx(0.7)
    assert diag["b_h"] > 0
    assert not torch.equal(agent2.c.detach(), c0)


def test_lambda0_matches_no_penalty_updates():
    """fhr_weight=0 must not touch c and must not add penalty terms."""
    _seed_all(6)
    env = gym.make("CartPole-v1")
    agent = _agent(env, fhr_weight=0.0)
    _fill(agent, env, 60)
    c0 = agent.c.detach().clone()
    for _ in range(3):
        diag = agent._train_step()
    assert torch.equal(agent.c.detach(), c0)
    assert np.isnan(diag["penalty_raw"])
    assert diag["penalty_weighted"] == 0.0


def test_ccond_head_init_and_training():
    c0_ref = torch.tensor([1.0 + 1.0 / 0.99, -1.0 / 0.99])
    head = CCondHead(2, 0.99, False, feature_dim=16, n_actions=2)
    c, d = head(torch.randn(9, 16), torch.zeros(9, dtype=torch.long))
    assert d is None
    assert torch.allclose(c, c0_ref.expand_as(c), atol=1e-6)

    _seed_all(7)
    env = gym.make("CartPole-v1")
    agent = _agent(env, fhr_weight=0.7, warmup_grad_steps=0,
                   c_predictor="shared")
    assert agent.c_head is not None
    # c group holds the HEAD's params, not the frozen global c
    grp = agent.optimiser.param_groups[-1]
    assert {id(p) for p in grp["params"]} == \
        {id(p) for p in agent.c_head.parameters()}
    _fill(agent, env, 80)
    w0 = agent.c_head.net.weight.detach().clone()
    c_global0 = agent.c.detach().clone()
    diag = agent._train_step()
    assert np.isfinite(diag["c_spread"])
    assert not torch.equal(agent.c_head.net.weight.detach(), w0)
    assert torch.equal(agent.c.detach(), c_global0)   # frozen reference


# ---------------------------------------------------- 4. acting / eval
def test_pi_honors_epsilon_and_sample_flag():
    _seed_all(8)
    env = gym.make("CartPole-v1")
    agent = _agent(env)
    obs, _ = env.reset(seed=0)

    agent.epsilon = 0.0
    agent.sample_actions = False
    greedy = [agent.pi(obs) for _ in range(5)]
    assert len(set(greedy)) == 1                     # deterministic argmax

    agent.sample_actions = True
    with torch.no_grad():                            # force a spread policy
        final = agent.policy_net.actor[-1]
        final.weight.zero_()
        final.bias.zero_()
    draws = {agent.pi(obs) for _ in range(50)}
    assert draws == {0, 1}                           # samples the categorical

    agent.epsilon = 1.0
    env.action_space.seed(3)
    assert agent.pi(obs) in (0, 1)                   # uniform override works


def test_evaluate_policy_atari_protocol():
    _seed_all(9)
    env = FakeImageEnv()
    agent = _agent(env, encoder_cls=ImageEncoder, encoder_kwargs={},
                   batch_size=4)
    agent.sample_actions = False
    scores = evaluate_policy_atari(agent, env, episodes=2, epsilon=0.001,
                                   base_seed=7)
    assert len(scores) == 2 and all(s == env.ep_len for s in scores)
    assert agent.epsilon == 0.0                      # restored after eval


# ---------------------------------------------------- 5. save / load
def test_save_load_roundtrip():
    _seed_all(10)
    env = gym.make("CartPole-v1")
    agent = _agent(env, fhr_weight=0.5, warmup_grad_steps=0,
                   c_predictor="shared")
    _fill(agent, env, 80)
    agent._train_step()
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "sacd.pt"
        agent.save(path)
        _seed_all(11)
        fresh = _agent(env, fhr_weight=0.5, warmup_grad_steps=0,
                       c_predictor="shared")
        fresh.load(path)
    assert torch.equal(fresh.log_alpha.detach(), agent.log_alpha.detach())
    assert torch.equal(fresh.c.detach(), agent.c.detach())
    for k, v in agent.c_head.state_dict().items():
        assert torch.equal(fresh.c_head.state_dict()[k], v)
    s = torch.zeros(1, 4)
    assert fresh.act_greedy(s) == agent.act_greedy(s)


# ---------------------------------------------------- 6. tiny e2e
def test_tiny_e2e_cartpole_loop():
    _seed_all(12)
    env = gym.make("CartPole-v1")
    agent = _agent(env, fhr_weight=0.2, warmup_grad_steps=5, batch_size=16,
                   gd_steps_ceil=2)
    with tempfile.TemporaryDirectory() as tmp:
        logger = RunLogger(tmp)
        rewards = dqn_training_loop(
            agent, env, no_episodes=15, target_network_update_steps=50,
            train_frequency_steps=1, use_episode_training=True,
            solved_reward=10 ** 9, warmup_steps=50,
            early_stopping_patience_eps=10 ** 9,
            analysis_config={"ep_freq": 10 ** 9, "methods": []},
            run_logger=logger)
        assert len(rewards) == 15
        assert agent.nan_skips == 0 and agent._grad_steps > 0
        header = (logger.dir / "train_diagnostics.csv").read_text() \
            .splitlines()[0].split(",")
        for col in ("td_loss", "penalty_raw", "b_h", "c_1", "c_2",
                    "actor_loss", "alpha_loss", "alpha", "entropy"):
            assert col in header, f"missing diagnostics column {col}"


def test_augmented_image_step():
    _seed_all(13)
    env = FakeImageEnv()
    agent = _agent(env, encoder_cls=ImageEncoder, encoder_kwargs={},
                   batch_size=4, fhr_weight=0.5, warmup_grad_steps=0,
                   use_augmentation=True, aug_pad=2, aug_intensity=0.05)
    obs, _ = env.reset()
    for _ in range(40):
        a = env.action_space.sample()
        nxt, r, term, trunc, _ = env.step(a)
        agent.update_buffer_atari(obs, a, r, nxt, term, trunc)
        obs = nxt
        if term or trunc:
            obs, _ = env.reset()
    diag = agent._train_step()
    assert agent.nan_skips == 0
    assert np.isfinite(diag["td_loss"]) and np.isfinite(diag["actor_loss"])
    assert diag["b_h"] >= 0


# ---------------------------------------------------- 7. PER buffer
def _feed_ep(buf, length, val0=0.0):
    for t in range(length):
        s = torch.full((1, 4), val0 + t)
        buf.append(s, torch.tensor([0]), torch.tensor([1.0]),
                   s + 1 if t < length - 1 else None)
    buf.close(True)


def test_per_max_priority_insert_and_uniform_weights():
    _seed_all(14)
    buf = PrioritizedEpisodicReplayBuffer(64, alpha=0.6)
    _feed_ep(buf, 10)
    _feed_ep(buf, 10)
    out = buf.sample_nstep_prioritized(8, 3, 0.99)
    handles, weights = out[5], out[6]
    assert len(handles) == 8
    assert torch.allclose(weights, torch.ones(8))    # all priorities equal
    for ep_idx, t in handles:
        assert ep_idx in (0, 1) and 0 <= t < 10


def test_per_update_skews_sampling_and_handles_valid():
    _seed_all(15)
    buf = PrioritizedEpisodicReplayBuffer(64, alpha=1.0, beta_start=0.4)
    _feed_ep(buf, 12)
    _feed_ep(buf, 12)
    target = (1, 7)
    buf.update_priorities([target], [500.0])
    hits = 0
    for _ in range(40):
        out = buf.sample_nstep_prioritized(8, 3, 0.99)
        hits += sum(1 for h in out[5] if h == target)
        assert (out[6] <= 1.0 + 1e-9).all()
    assert hits > 120
    assert buf.beta > 0.4                            # annealing advanced

    # predecessors work off prioritized handles exactly as uniform ones
    keep = [h for h in out[5] if h[1] >= 2]
    if keep:
        s, a, r = buf.gather_predecessors(keep, 2)
        assert s.shape[1] == 2


def test_per_eviction_zeroes_leaves():
    _seed_all(16)
    buf = PrioritizedEpisodicReplayBuffer(30, alpha=0.6)
    for k in range(5):                               # 5 x 10 > capacity 30
        _feed_ep(buf, 10, val0=100.0 * k)
    assert len(buf._episodes) <= 3
    total = buf.tree.total()
    live = len(buf)
    assert total == pytest.approx(live * buf.max_priority ** buf.alpha)
    # every sampled handle maps to a live episode
    out = buf.sample_nstep_prioritized(16, 3, 0.99)
    for ep_idx, t in out[5]:
        assert ep_idx < len(buf._episodes)
        assert t < len(buf._episodes[ep_idx]["states"])


def test_per_oversized_episode_raises():
    buf = PrioritizedEpisodicReplayBuffer(8)
    with pytest.raises(RuntimeError, match="longer than capacity"):
        _feed_ep(buf, 12)


def test_uniform_path_untouched_by_per_module():
    """The plain buffer's RNG stream and class are unchanged: a
    prioritized_replay=False agent gets the parent class, and seeded uniform
    sampling draws identical indices as before this module existed."""
    agent = _agent()
    assert type(agent.replay_buffer) is EpisodicReplayBuffer
    agent_per = _agent(prioritized_replay=True)
    assert type(agent_per.replay_buffer) is PrioritizedEpisodicReplayBuffer

    buf = EpisodicReplayBuffer(64)
    _feed_ep(buf, 20)
    random.seed(123)
    _, _, _, _, h1 = buf.sample_transitions(6, with_handles=True)
    random.seed(123)
    _, _, _, _, h2 = buf.sample_transitions(6, with_handles=True)
    assert h1 == h2


def test_per_train_step_updates_priorities():
    _seed_all(17)
    env = gym.make("CartPole-v1")
    agent = _agent(env, prioritized_replay=True, per_alpha=0.6,
                   fhr_weight=0.5, warmup_grad_steps=0)
    _fill(agent, env, 80)
    total0 = agent.replay_buffer.tree.total()
    diag = agent._train_step()
    assert agent.replay_buffer.tree.total() != total0
    assert "per_beta" in diag and "is_weight_mean" in diag
    assert 0 < diag["is_weight_mean"] <= 1.0


def test_fhr_lag_source_split_path_matches_fused_at_init():
    """detached/target split the fused feature pass; with synced nets and no
    augmentation the first-step penalty value matches the online path for
    both critic heads (penalty_raw is their mean)."""
    diags = {}
    for src in ("online", "detached", "target"):
        _seed_all(0)
        env = gym.make("CartPole-v1")
        agent = _agent(env, fhr_weight=0.5, fhr_lag_source=src)
        _fill(agent, env)
        _seed_all(123)
        diags[src] = agent.train()
        assert agent.nan_skips == 0
        env.close()
    assert diags["online"]["b_h"] > 0
    p0 = diags["online"]["penalty_raw"]
    assert np.isfinite(p0)
    assert abs(diags["detached"]["penalty_raw"] - p0) < 1e-5
    assert abs(diags["target"]["penalty_raw"] - p0) < 1e-5
