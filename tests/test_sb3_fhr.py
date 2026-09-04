"""Tests for agents.sb3_fhr: the SB3-hosted FHR penalty (FHRDQN / FHRQRDQN),
the (episode, t)-annotated ring buffer, warm-up / ramp-down scheduling, the
c-coefficient optimiser group, save/load, the RunLogger callback contract, and
the SB3QAgentAdapter surface the analysis stack consumes. CPU-only, tiny nets
and budgets; fhr_weight=0 runs are checked bit-for-bit against stock SB3."""
import csv
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

from stable_baselines3 import DQN                                  # noqa: E402
from stable_baselines3.common.logger import configure              # noqa: E402
from stable_baselines3.common.monitor import Monitor               # noqa: E402
from sb3_contrib import QRDQN                                      # noqa: E402

from agents.fhrdqn_agent import FHRDQNAgent                        # noqa: E402
from agents.sb3_fhr import (BoundedObservations, FHRDQN,           # noqa: E402
                            FHREpisodicReplayBuffer, FHRQRDQN,
                            FHRRecurrenceHead, FHRSB3Callback,
                            SB3QAgentAdapter)
from analysis.low_rank.hankel_policy import collect_hankel_sequences  # noqa: E402
from analysis.low_rank.tabular_q_matrix import q_matrix_dqn        # noqa: E402
from analysis.run_logger import RunLogger                          # noqa: E402


# --------------------------------------------------------------- helpers
def _seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _dqn_kwargs(**over):
    kw = dict(policy_kwargs=dict(net_arch=[32, 32]), buffer_size=1000,
              learning_starts=100, batch_size=32, seed=7, device="cpu",
              verbose=0)
    kw.update(over)
    return kw


def _fill_buffer(model, n_steps=200, seed=1):
    """Feed the model's replay buffer from a random CartPole rollout."""
    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    for _ in range(n_steps):
        action = env.action_space.sample()
        nxt, r, term, trunc, _ = env.step(action)
        model.replay_buffer.add(np.asarray([obs]), np.asarray([nxt]),
                                np.asarray([action]), np.asarray([r]),
                                np.asarray([term or trunc]), [{}])
        obs = nxt
        if term or trunc:
            obs, _ = env.reset()
    env.close()


def _filled_model(**over):
    """FHRDQN with a hand-filled buffer and a null logger, ready for train()."""
    model = FHRDQN("MlpPolicy", gym.make("CartPole-v1"), **_dqn_kwargs(**over))
    model.set_logger(configure(folder=None, format_strings=[]))
    _fill_buffer(model)
    return model


def _make_buffer(size=32):
    env = gym.make("CartPole-v1")
    buf = FHREpisodicReplayBuffer(size, env.observation_space,
                                  env.action_space, device="cpu", n_envs=1)
    env.close()
    return buf


def _feed_episode(buf, length):
    for t in range(length):
        obs = np.full((1, 4), float(t), dtype=np.float32)
        buf.add(obs, obs, np.array([0]), np.array([1.0]),
                np.array([t == length - 1]), [{}])


# ------------------------------------------- 1. lambda=0 bit-exactness
def test_lambda0_bit_exact_matches_stock_sb3():
    def final_net_state(cls, policy_kwargs, extra):
        _seed_all(0)
        model = cls("MlpPolicy", gym.make("CartPole-v1"),
                    **_dqn_kwargs(policy_kwargs=policy_kwargs, **extra))
        model.learn(total_timesteps=800)
        net = model.q_net if hasattr(model, "q_net") else model.quantile_net
        state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        model.env.close()
        return state

    mlp = dict(net_arch=[32, 32])
    stock = final_net_state(DQN, mlp, {})
    fhr = final_net_state(FHRDQN, mlp, dict(fhr_weight=0.0))
    assert stock.keys() == fhr.keys()
    for k in stock:
        assert torch.equal(stock[k], fhr[k]), k

    qmlp = dict(net_arch=[32, 32], n_quantiles=5)
    stock_qr = final_net_state(QRDQN, qmlp, {})
    fhr_qr = final_net_state(FHRQRDQN, qmlp, dict(fhr_weight=0.0))
    assert stock_qr.keys() == fhr_qr.keys()
    for k in stock_qr:
        assert torch.equal(stock_qr[k], fhr_qr[k]), k


# ----------------------------------- 2. ring-buffer predecessor bookkeeping
def test_predecessors_episodes_and_wraparound():
    _seed_all(0)
    buf = _make_buffer(32)
    # ep0 slots 0..9 (t 0..9), ep1 slots 10..19 (t 0..9),
    # ep2 slots 20..31 then wraps to 0..7 (t 0..19): 40 adds on 32 slots
    for length in (10, 10, 20):
        _feed_episode(buf, length)
    assert buf.full and buf.pos == 8

    inds = np.array([8, 10, 11, 12, 0, 1, 20, 9])
    keep, pred = buf.predecessors(inds, 2)
    # slot 8: ep0 t=8, but slots 7/6 were overwritten by ep2 -> dropped
    # slot 10/11: ep1 t=0/1 < order          slot 20: ep2 t=0 < order
    # slot 9: ep0 t=9, slot 8 still ep0 but slot 7 overwritten -> dropped
    assert keep.tolist() == [False, False, False, True, True, True,
                             False, False]
    # kept rows: pred[:, j-1] holds t-j of the same episode
    for row in np.flatnonzero(keep):
        i = inds[row]
        for j in (1, 2):
            slot = pred[row, j - 1]
            assert buf.episode_ids[slot] == buf.episode_ids[i]
            assert buf.t_in_episode[slot] == buf.t_in_episode[i] - j
    assert pred[4].tolist() == [31, 30]        # slot 0 wraps to the ring tail
    assert pred[5].tolist() == [0, 31]         # slot 1 straddles the wrap
    keep3, pred3 = buf.predecessors(np.array([0]), 3)
    assert keep3.tolist() == [True] and pred3[0].tolist() == [31, 30, 29]

    # unwritten-region wraparound before the buffer is full: never keep=True
    buf2 = _make_buffer(32)
    for t in range(5):                          # one open episode, 5 slots
        obs = np.full((1, 4), float(t), dtype=np.float32)
        buf2.add(obs, obs, np.array([0]), np.array([1.0]),
                 np.array([False]), [{}])
    assert not buf2.full
    assert (buf2.episode_ids[5:] == -1).all()   # unwritten slots stay -1
    k1, _ = buf2.predecessors(np.arange(5), 1)
    assert k1.tolist() == [False, True, True, True, True]
    k2, _ = buf2.predecessors(np.arange(5), 2)
    assert k2.tolist() == [False, False, True, True, True]
    for order in (1, 2, 3, 6):
        keep, pred = buf2.predecessors(np.arange(5), order)
        wraps = (pred >= 5).any(axis=1)         # points into unwritten slots
        assert not keep[wraps].any()

    # one episode longer than the buffer wraps over its own start: the seam
    # slots carry the SAME episode id at the wrong t, so the episode-id check
    # alone would pair anchors with future-time lags — the exact-t check in
    # predecessors() must drop them (review regression)
    buf3 = _make_buffer(8)
    _feed_episode(buf3, 20)                     # t 0..19 on 8 slots, pos = 4
    assert buf3.full and buf3.pos == 4
    keep, pred = buf3.predecessors(np.arange(8), 2)
    for row in range(8):
        if keep[row]:
            for j in (1, 2):
                slot = pred[row, j - 1]
                assert buf3.t_in_episode[slot] == buf3.t_in_episode[row] - j
    # the oldest surviving slot (pos) neighbours the newest data at the seam
    assert not keep[buf3.pos]


# ------------------------------------------------- 3. hard warm-up schedule
def test_warmup_lambda_transition():
    _seed_all(0)
    model = _filled_model(fhr_weight=0.4, warmup_grad_steps=5, fhr_order=2)
    assert model._lambda_eff() == 0.0
    for _ in range(8):
        model.train(gradient_steps=1, batch_size=32)
    rows = model.drain_diagnostics()
    assert [r["lambda_eff"] for r in rows] == [0.0] * 5 + [0.4] * 3
    assert model._fhr_grad_steps == 8
    assert model._lambda_eff() == 0.4
    assert model.drain_diagnostics() == []      # drain empties the queue


# ------------------------------------------------ 4. penalty gradient flow
def test_penalty_gradient_flow_and_c_param_group():
    _seed_all(0)
    model = _filled_model(fhr_weight=0.5, warmup_grad_steps=0,
                          c_learning_rate=0.02)
    group = model.policy.optimizer.param_groups[model._fhr_group_index]
    assert group["weight_decay"] == 0.0
    assert group["lr"] == 0.02
    assert group["params"][0] is model.fhr_head.c
    assert id(model.fhr_head.c) not in {id(p) for p in model.policy.parameters()}

    c0 = model.fhr_head.c.detach().clone()
    model.train(gradient_steps=8, batch_size=32)
    rows = model.drain_diagnostics()
    assert rows[-1]["b_h"] > 0                  # the penalty actually engaged
    assert not torch.equal(model.fhr_head.c.detach(), c0)
    # the schedule overwrote every group's lr in train(); c's was restored
    assert group["lr"] == 0.02

    baseline = _filled_model(fhr_weight=0.0)
    cb0 = baseline.fhr_head.c.detach().clone()
    baseline.train(gradient_steps=8, batch_size=32)
    assert torch.equal(baseline.fhr_head.c.detach(), cb0)
    assert baseline.fhr_head.c.grad is None


# ---------------------------------------------------- 5. Bellman-informed inits
def test_recurrence_head_inits_match_fhrdqn_agent():
    gamma = 0.9

    class TinyNet(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.net = nn.Linear(in_dim, out_dim)

        def forward(self, x):
            return self.net(x)

    def classic(**fhr_over):
        env = gym.make("CartPole-v1")
        agent = FHRDQNAgent(
            replay_buffer_capacity=100, q_network=TinyNet, batch_size=8,
            nn_learning_rate=1e-3,
            nn_extra_kwargs={"in_dim": 4, "out_dim": 2}, env=env,
            eps_start=1.0, eps_min=0.05, decay_rate=0.999,
            discount_factor=gamma, device="cpu", TD_LR=1.0, buffer_util=1,
            gd_steps_ceil=1, double=True, fhr_weight=0.1,
            warmup_grad_steps=0, **fhr_over)
        env.close()
        return agent

    # reward lags (ARX): c1 = 1/gamma, d1 = -1/gamma, rest 0
    head = FHRRecurrenceHead(3, gamma, reward_lags=True)
    ag = classic(fhr_order=3, reward_lags=True)
    assert torch.equal(head.c.detach(), ag.c.detach())
    assert torch.equal(head.d.detach(), ag.d.detach())
    assert head.c[0].item() == pytest.approx(1.0 / gamma)
    assert head.d[0].item() == pytest.approx(-1.0 / gamma)
    assert torch.equal(head.c[1:].detach(), torch.zeros(2))
    assert torch.equal(head.d[1:].detach(), torch.zeros(2))

    # pure AR, r >= 2: c = (1 + 1/gamma, -1/gamma, 0, ...)
    head = FHRRecurrenceHead(3, gamma, reward_lags=False)
    ag = classic(fhr_order=3, reward_lags=False)
    assert head.d is None and ag.d is None
    assert torch.equal(head.c.detach(), ag.c.detach())
    assert head.c[0].item() == pytest.approx(1.0 + 1.0 / gamma)
    assert head.c[1].item() == pytest.approx(-1.0 / gamma)
    assert head.c[2].item() == 0.0

    # pure AR, r = 1: c1 = 1/gamma (the classic agent warns about this config)
    head = FHRRecurrenceHead(1, gamma, reward_lags=False)
    with pytest.warns(UserWarning):
        ag = classic(fhr_order=1, reward_lags=False)
    assert head.d is None
    assert torch.equal(head.c.detach(), ag.c.detach())
    assert head.c[0].item() == pytest.approx(1.0 / gamma)


# ------------------------------------------------ 6. rampdown trigger + parsing
def _rd_model(**over):
    kw = dict(fhr_weight=0.5, warmup_grad_steps=0,
              rampdown_reward_threshold=100.0, rampdown_patience_eps=3,
              rampdown_episodes=2)
    kw.update(over)
    return FHRDQN("MlpPolicy", gym.make("CartPole-v1"), **_dqn_kwargs(**kw))


def test_notify_episode_end_rampdown():
    _seed_all(0)
    model = _rd_model()
    model._fhr_grad_steps = 10        # bypass training: post-warm-up state
    scales, lams, rd_ks = [], [], []
    for ep, rew in enumerate([10.0, 10.0, 10.0, 200.0, 200.0, 200.0, 200.0]):
        model._ep_penalty_buckets[ep] = [1.0, 2.0]
        model.notify_episode_end(ep, rew)
        assert ep not in model._ep_penalty_buckets    # bucket consumed
        scales.append(model._rampdown_scale())
        lams.append(model._lambda_eff())
        rd_ks.append(model._rd_k)
    # window means: 10, 10, 10, 73.3, 136.7 -> triggers on the 5th episode,
    # then anneals linearly over rampdown_episodes=2 and stays down (one-way)
    assert rd_ks == [None, None, None, None, 1, 2, 3]
    assert scales == [1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0]
    assert lams == [0.5, 0.5, 0.5, 0.5, 0.25, 0.0, 0.0]

    # warm-up episodes never arm the trigger window
    frozen = _rd_model(warmup_grad_steps=50)
    frozen._fhr_grad_steps = 0
    for ep in range(4):
        frozen._ep_penalty_buckets[ep] = [1.0]
        frozen.notify_episode_end(ep, 200.0)
    assert frozen._rd_k is None and frozen._rampdown_scale() == 1.0

    # penalty-gated mode under burst scheduling: episodes with no burst count
    # as no-data (NaN) and are ignored by the gate, not allowed to veto it
    gated = _rd_model(rampdown_penalty_threshold=1.5)
    gated._fhr_grad_steps = 10
    for ep, vals in enumerate([[2.0], [], [2.0]]):    # middle episode: no burst
        if vals:
            gated._ep_penalty_buckets[ep] = vals
        gated.notify_episode_end(ep, 200.0)
    assert gated._rd_k == 1                           # triggered despite the NaN
    below_bar = _rd_model(rampdown_penalty_threshold=1.5)
    below_bar._fhr_grad_steps = 10
    for ep in range(3):
        below_bar._ep_penalty_buckets[ep] = [1.0]     # finite but under the bar
        below_bar.notify_episode_end(ep, 200.0)
    assert below_bar._rd_k is None                    # gate still vetoes

    # "NN%" relative-bar string parsing, and a bad string raises
    pct = _rd_model(rampdown_penalty_threshold="40%")
    assert pct._rd_pen_frac == pytest.approx(0.4)
    assert pct._rd_pen_abs is None
    with pytest.raises(ValueError):
        _rd_model(rampdown_penalty_threshold="forty percent")


# ---------------------------------------------------- 7. save/load round-trip
def test_save_load_roundtrip(tmp_path):
    _seed_all(0)
    model = _filled_model(fhr_weight=0.3, fhr_order=2, warmup_grad_steps=3,
                          c_learning_rate=0.011)
    model.train(gradient_steps=5, batch_size=32)
    with torch.no_grad():
        model.fhr_head.c += 0.25
    model.nan_skips = 2
    path = tmp_path / "fhr_model.zip"
    model.save(path)

    loaded = FHRDQN.load(path, device="cpu")
    assert torch.equal(loaded.fhr_head.c.detach(), model.fhr_head.c.detach())
    assert loaded.fhr_weight == 0.3
    assert loaded.fhr_order == 2
    assert loaded.reward_lags is False
    assert loaded.warmup_grad_steps == 3
    assert loaded.c_learning_rate == 0.011
    assert loaded._fhr_grad_steps == 5
    assert loaded.nan_skips == 2
    for k, v in model.q_net.state_dict().items():
        assert torch.equal(v, loaded.q_net.state_dict()[k]), k


# ---------------------------------------------- 8. FHRSB3Callback artifact contract
def test_callback_run_logger_contract(tmp_path):
    _seed_all(0)
    mon = Monitor(gym.make("CartPole-v1"))
    logger = RunLogger(tmp_path, config_path=None, run_id="cbrun")
    model = FHRDQN("MlpPolicy", mon,
                   **_dqn_kwargs(fhr_weight=0.05, warmup_grad_steps=0,
                                 learning_starts=200, buffer_size=5000))
    callback = FHRSB3Callback(run_logger=logger,
                              analysis_config={"ep_freq": 20},
                              analysis_env=None)
    model.learn(total_timesteps=1500, callback=callback)

    n_eps = len(mon.get_episode_rewards())
    assert n_eps >= 10                       # enough episodes for a best window
    with open(logger.dir / "rewards.csv") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["episode", "reward", "steps"]
    assert len(rows) - 1 == n_eps
    assert np.allclose([float(r[1]) for r in rows[1:]],
                       mon.get_episode_rewards())

    with open(logger.dir / "train_diagnostics.csv") as f:
        drows = list(csv.reader(f))
    header = drows[0]
    assert header[0] == "episode"
    required = {"td_loss", "penalty_raw", "lambda_eff", "c_1", "c_2", "sum_c",
                "companion_radius", "b_h", "unique_eps", "residual_rms",
                "nan_skips", "rampdown_scale"}
    assert required <= set(header)
    assert len(drows) > 1
    b_h = float(drows[-1][header.index("b_h")])
    assert b_h > 0                           # the penalty rode real batches

    for name in ("best", "final", "latest"):  # episode 0 ticks, so latest too
        assert (logger.checkpoints_dir / f"{name}.pt").exists()
    mon.close()


# --------------------------------------------- 9. SB3QAgentAdapter + analyses
def test_adapter_surfaces_and_analysis_compatibility(tmp_path):
    def mini_trained(cls, policy_kwargs):
        _seed_all(5)
        model = cls("MlpPolicy", gym.make("CartPole-v1"),
                    **_dqn_kwargs(seed=5, policy_kwargs=policy_kwargs,
                                  learning_starts=50))
        model.learn(total_timesteps=200)
        return model

    dqn_model = mini_trained(FHRDQN, dict(net_arch=[32, 32]))
    qr_model = mini_trained(FHRQRDQN, dict(net_arch=[32, 32], n_quantiles=5))

    low = [-2.4, -3.0, -0.21, -3.0]
    high = [2.4, 3.0, 0.21, 3.0]
    for model, load_cls, tag in ((dqn_model, FHRDQN, "dqn"),
                                 (qr_model, FHRQRDQN, "qr")):
        adapter = SB3QAgentAdapter(model, epsilon=0.0)
        assert adapter.device == model.device
        out = adapter.policy_net(torch.randn(7, 4))
        assert out.shape == (7, 2)
        action = adapter.act_greedy(torch.randn(1, 4))
        assert isinstance(action, int) and action in range(2)
        pick = adapter.pi(np.zeros(4, dtype=np.float32))
        assert isinstance(pick, int) and pick in range(2)

        # save writes a loadable SB3 checkpoint (RunLogger's .pt naming)
        path = tmp_path / f"{tag}_adapter.pt"
        adapter.save(path)
        loaded = load_cls.load(str(path), device="cpu")
        net = model.q_net if hasattr(model, "q_net") else model.quantile_net
        lnet = loaded.q_net if hasattr(loaded, "q_net") else loaded.quantile_net
        for k, v in net.state_dict().items():
            assert torch.equal(v, lnet.state_dict()[k]), k

        # Q-matrix analysis over a bounded discretised state grid
        grid_env = BoundedObservations(gym.make("CartPole-v1"), low, high)
        qm = q_matrix_dqn(adapter, [4, 4, 4, 4], env=grid_env, batch_size=16)
        assert qm.shape == (256, 2)
        assert np.isfinite(qm).all()
        grid_env.close()

    # Hankel sequence collection via the adapter's pi/policy_net surface
    rollout_env = gym.make("CartPole-v1")
    seqs = collect_hankel_sequences(SB3QAgentAdapter(dqn_model, epsilon=0.0),
                                    rollout_env, seed=1)
    assert "Hankel Q" in seqs and len(seqs["Hankel Q"]) > 0
    rollout_env.close()


# ---------------------------------------- state-conditioned c predictor
from agents.sb3_fhr import FHRCoefficientPredictor, _bellman_init  # noqa: E402


def test_c_predictor_init_matches_bellman_everywhere():
    """Zero-weight/Bellman-bias output layer: every anchor starts with exactly
    the global head's init, for both modes, AR and ARX."""
    torch.manual_seed(0)
    for mode in ("shared", "separate"):
        for reward_lags in (False, True):
            pred = FHRCoefficientPredictor(3, 0.99, reward_lags, mode,
                                           in_dim=8, n_actions=2)
            c0, d0 = _bellman_init(3, 0.99, reward_lags)
            feats = torch.randn(16, 8)
            acts = torch.randint(0, 2, (16,))
            c, d = pred(feats, acts)
            assert torch.allclose(c, c0.expand(16, 3)), (mode, reward_lags)
            if reward_lags:
                assert torch.allclose(d, d0.expand(16, 3))
            else:
                assert d is None


def test_c_predictor_lambda0_bit_exact_matches_stock_sb3():
    """c_predictor construction must not perturb a fhr_weight=0 run: the
    penalty branch is never entered and the extra torch RNG draws happen
    after the policy is built."""
    def run(cls, **extra):
        _seed_all(3)
        model = cls("MlpPolicy", gym.make("CartPole-v1"),
                    **_dqn_kwargs(), **extra)
        model.set_logger(configure(None, ["stdout"]))
        _fill_buffer(model)
        model.train(gradient_steps=8, batch_size=32)
        return model.q_net.state_dict()

    stock = run(DQN)
    fhr = run(FHRDQN, fhr_weight=0.0, c_predictor="separate")
    assert stock.keys() == fhr.keys()
    for k in stock:
        assert torch.equal(stock[k], fhr[k]), k


@pytest.mark.parametrize("mode", ["shared", "separate"])
def test_c_predictor_trains_and_reports(mode):
    _seed_all(0)
    model = _filled_model(fhr_weight=0.5, warmup_grad_steps=0,
                          c_predictor=mode)
    before = {k: v.clone() for k, v in model.fhr_predictor.state_dict().items()}
    head_c0 = model.fhr_head.c.detach().clone()
    model.train(gradient_steps=6, batch_size=32)
    after = model.fhr_predictor.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before), \
        "predictor never updated"
    # the global head is inert when the predictor is active
    assert torch.equal(model.fhr_head.c.detach(), head_c0)
    # last optimiser group holds exactly the predictor's params at c lr
    group = model.policy.optimizer.param_groups[model._fhr_group_index]
    assert group["lr"] == model.c_learning_rate
    assert {id(p) for p in group["params"]} == \
        {id(p) for p in model.fhr_predictor.parameters()}
    rows = model.drain_diagnostics()
    assert rows and np.isfinite(rows[-1]["c_1"])
    assert "c_spread" in rows[-1] and np.isfinite(rows[-1]["c_spread"])


def test_c_predictor_arx_trains():
    _seed_all(1)
    model = _filled_model(fhr_weight=0.5, warmup_grad_steps=0,
                          c_predictor="separate", reward_lags=True)
    model.train(gradient_steps=6, batch_size=32)
    rows = model.drain_diagnostics()
    assert np.isfinite(rows[-1]["d_1"]) and np.isfinite(rows[-1]["penalty_raw"])
    assert model.nan_skips == 0


def test_c_predictor_save_load_roundtrip(tmp_path):
    _seed_all(2)
    model = _filled_model(fhr_weight=0.5, warmup_grad_steps=0,
                          c_predictor="shared")
    model.train(gradient_steps=4, batch_size=32)
    p = tmp_path / "ccond.zip"
    model.save(p)
    loaded = FHRDQN.load(p, env=gym.make("CartPole-v1"), device="cpu")
    assert loaded.c_predictor == "shared"
    for k, v in model.fhr_predictor.state_dict().items():
        assert torch.equal(v, loaded.fhr_predictor.state_dict()[k]), k


def test_fhr_lag_source_variants_dqn():
    """Bad value rejected at _setup_model; detached/target train cleanly and
    the first-step penalty value matches online (q_net_target == q_net at
    init), while the post-step parameters differ from the online variant."""
    try:
        FHRDQN("MlpPolicy", gym.make("CartPole-v1"),
               **_dqn_kwargs(fhr_weight=0.5, fhr_lag_source="bogus"))
        assert False, "expected ValueError for bad fhr_lag_source"
    except ValueError:
        pass
    rows, params = {}, {}
    for src in ("online", "detached", "target"):
        _seed_all(0)
        m = _filled_model(fhr_weight=0.5, fhr_order=2, warmup_grad_steps=0,
                          fhr_lag_source=src)
        _seed_all(123)
        m.train(gradient_steps=1, batch_size=32)
        rows[src] = m.drain_diagnostics()
        params[src] = {k: v.detach().clone()
                       for k, v in m.q_net.state_dict().items()}
    r0 = rows["online"][0]
    assert r0["b_h"] > 0 and np.isfinite(r0["penalty_raw"])
    assert abs(rows["detached"][0]["penalty_raw"] - r0["penalty_raw"]) < 1e-6
    assert abs(rows["target"][0]["penalty_raw"] - r0["penalty_raw"]) < 1e-6
    assert any(not torch.equal(params["online"][k], params["detached"][k])
               for k in params["online"])
