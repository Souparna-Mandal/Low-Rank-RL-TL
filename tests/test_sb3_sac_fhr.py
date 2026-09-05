"""Tests for agents.sb3_sac_fhr: FHRSAC (continuous), SACD/FHRSACD
(SAC-Discrete), the prioritized episodic replay buffer, the continuous
analysis adapter + hankel_rollout_continuous method, and the lambda=0
bit-exactness contracts. CPU-only, tiny nets and budgets."""
import csv
import pathlib
import random
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from stable_baselines3 import SAC                                  # noqa: E402
from stable_baselines3.common.logger import configure              # noqa: E402

from agents.sb3_fhr import (FHR_PARAMS, FHRDQN,                    # noqa: E402
                            FHRCoefficientPredictor,
                            FHREpisodicReplayBuffer, GreedyEvalCallback,
                            _bellman_init)
from agents.sb3_sac_fhr import (FHRSAC, FHRSACD, SACD,             # noqa: E402
                                FHRPrioritizedEpisodicReplayBuffer,
                                SB3SACAdapter, SB3SACDAdapter)
from analysis.low_rank.continuous_rollout import hankel_rollout_continuous  # noqa: E402
from analysis.low_rank.hankel_policy import collect_hankel_sequences  # noqa: E402
from analysis.low_rank.tabular_q_matrix import q_matrix_dqn        # noqa: E402


# --------------------------------------------------------------- helpers
def _seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sac_kwargs(**over):
    kw = dict(policy_kwargs=dict(net_arch=[32, 32]), buffer_size=1000,
              learning_starts=100, batch_size=32, train_freq=1,
              gradient_steps=1, seed=7, device="cpu", verbose=0)
    kw.update(over)
    return kw


def _fill_buffer(model, env_id, n_steps=220, seed=1):
    """Feed the model's replay buffer from a random rollout (actions stored
    raw — fine for penalty mechanics, which don't care about scaling)."""
    env = gym.make(env_id)
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


def _null_logger(model):
    model.set_logger(configure(folder=None, format_strings=[]))
    return model


def _filled_sac(**over):
    model = FHRSAC("MlpPolicy", gym.make("Pendulum-v1"), **_sac_kwargs(**over))
    _null_logger(model)
    _fill_buffer(model, "Pendulum-v1")
    return model


def _filled_sacd(**over):
    model = FHRSACD("MlpPolicy", gym.make("CartPole-v1"), **_sac_kwargs(**over))
    _null_logger(model)
    _fill_buffer(model, "CartPole-v1")
    return model


def _policy_state(model):
    return {k: v.detach().clone()
            for k, v in model.policy.state_dict().items()}


# ------------------------------------------- 1. lambda=0 bit-exactness
def test_fhrsac_lambda0_bit_exact_matches_stock_sac():
    def final_state(cls, extra):
        _seed_all(0)
        model = cls("MlpPolicy", gym.make("Pendulum-v1"),
                    **_sac_kwargs(**extra))
        model.learn(total_timesteps=400)
        state = _policy_state(model)
        ent = model.log_ent_coef.detach().clone()
        model.env.close()
        return state, ent

    stock, stock_ent = final_state(SAC, {})
    fhr, fhr_ent = final_state(FHRSAC, dict(fhr_weight=0.0))
    assert stock.keys() == fhr.keys()
    for k in stock:
        assert torch.equal(stock[k], fhr[k]), k
    assert torch.equal(stock_ent, fhr_ent)


def test_fhrsac_ccond_lambda0_bit_exact():
    # the ccond predictor's construction must not shift the torch RNG stream
    # the actor samples from (fork_rng guard in _fhr_build_predictor)
    def final_state(cls, extra):
        _seed_all(0)
        model = cls("MlpPolicy", gym.make("Pendulum-v1"),
                    **_sac_kwargs(**extra))
        model.learn(total_timesteps=300)
        state = _policy_state(model)
        model.env.close()
        return state

    stock = final_state(SAC, {})
    for mode in ("separate", "shared"):
        fhr = final_state(FHRSAC, dict(fhr_weight=0.0, c_predictor=mode))
        for k in stock:
            assert torch.equal(stock[k], fhr[k]), (mode, k)


def test_fhrsacd_lambda0_bit_exact_matches_sacd():
    def final_state(cls, extra):
        _seed_all(0)
        model = cls("MlpPolicy", gym.make("CartPole-v1"),
                    **_sac_kwargs(**extra))
        model.learn(total_timesteps=400)
        state = _policy_state(model)
        model.env.close()
        return state

    stock = final_state(SACD, {})
    fhr = final_state(FHRSACD, dict(fhr_weight=0.0))
    assert stock.keys() == fhr.keys()
    for k in stock:
        assert torch.equal(stock[k], fhr[k]), k


# ------------------------------------------------------- 2. SACD mechanics
def test_sacd_target_entropy_and_smoke():
    _seed_all(0)
    model = SACD("MlpPolicy", gym.make("CartPole-v1"), **_sac_kwargs())
    assert model.target_entropy == pytest.approx(0.98 * np.log(2))
    model.env.close()

    model = SACD("MlpPolicy", gym.make("CartPole-v1"),
                 **_sac_kwargs(target_entropy_scale=0.5))
    assert model.target_entropy == pytest.approx(0.5 * np.log(2))
    model.learn(total_timesteps=400)
    for p in model.policy.parameters():
        assert torch.isfinite(p).all()
    assert float(torch.exp(model.log_ent_coef.detach())) > 0
    model.env.close()


def test_sacd_alpha_loss_direction():
    """Peaked policy (H < target) must push log_ent_coef UP; a near-uniform
    policy (H > target) must push it DOWN."""
    def one_step_delta(bias):
        _seed_all(0)
        model = _filled_sacd(fhr_weight=0.0)
        with torch.no_grad():
            final = model.actor.logits_net[-1]
            final.weight.zero_()
            final.bias.copy_(torch.tensor(bias))
        before = float(model.log_ent_coef.detach())
        model.train(gradient_steps=1, batch_size=32)
        delta = float(model.log_ent_coef.detach()) - before
        model.env.close() if model.env else None
        return delta

    assert one_step_delta([10.0, -10.0]) > 0     # peaked: entropy ~0 < target
    assert one_step_delta([0.0, 0.0]) < 0        # uniform: entropy log2 > target


# --------------------------------------------- 3. penalty engagement + c
@pytest.mark.parametrize("family", ["sac", "sacd"])
def test_penalty_engages_after_warmup(family):
    _seed_all(3)
    make = _filled_sac if family == "sac" else _filled_sacd
    model = make(fhr_weight=0.5, fhr_order=2, warmup_grad_steps=3,
                 c_learning_rate=1e-2, learning_starts=0)
    c_before = model.fhr_head.c.detach().clone()
    model.train(gradient_steps=8, batch_size=32)
    rows = model.drain_diagnostics()
    assert rows and rows[0]["b_h"] > 0
    # hard warm-up: per-burst nanmean mixes 3 zero-lambda and 5 full-lambda
    # steps; direct per-step check via _lambda_eff bookkeeping
    assert model._fhr_grad_steps == 8
    assert model._lambda_eff() == pytest.approx(0.5)
    assert not torch.equal(model.fhr_head.c.detach(), c_before)
    # the c group lives on the critic optimizer, last group, correct lr
    group = model.critic.optimizer.param_groups[-1]
    assert group["lr"] == pytest.approx(1e-2)
    assert any(p is model.fhr_head.c for p in group["params"])
    # ... and no fhr parameter is reachable from the actor optimizer
    actor_params = {id(p) for g in model.actor.optimizer.param_groups
                    for p in g["params"]}
    assert id(model.fhr_head.c) not in actor_params
    # the schedule restore keeps the c lr after another burst
    model.train(gradient_steps=1, batch_size=32)
    assert model.critic.optimizer.param_groups[-1]["lr"] == pytest.approx(1e-2)


@pytest.mark.parametrize("family", ["sac", "sacd"])
def test_c_gets_no_gradients_during_warmup_or_lambda0(family):
    _seed_all(4)
    make = _filled_sac if family == "sac" else _filled_sacd
    for kwargs in (dict(fhr_weight=0.0),
                   dict(fhr_weight=1.0, warmup_grad_steps=10 ** 6)):
        model = make(learning_starts=0, **kwargs)
        c_before = model.fhr_head.c.detach().clone()
        model.train(gradient_steps=4, batch_size=32)
        assert model.fhr_head.c.grad is None
        assert torch.equal(model.fhr_head.c.detach(), c_before)


def test_ccond_raw_encoding_bellman_init():
    c0, _ = _bellman_init(2, 0.99, False)
    pred = FHRCoefficientPredictor(2, 0.99, False, "separate", in_dim=3,
                                   n_actions=1, action_encoding="raw")
    feats = torch.randn(16, 3)
    acts = torch.randn(16, 1)
    c, d = pred(feats, acts)
    assert d is None
    assert torch.allclose(c, c0.expand_as(c))

    # a shared-mode FHRSAC builds a raw-encoding predictor sized to the
    # critic's penultimate layer and reports ccond diagnostics
    _seed_all(5)
    model = _filled_sac(fhr_weight=0.5, warmup_grad_steps=0,
                        c_predictor="shared", learning_starts=0)
    assert model.fhr_predictor.action_encoding == "raw"
    model.train(gradient_steps=2, batch_size=32)
    rows = model.drain_diagnostics()
    assert np.isfinite(rows[0]["c_spread"])
    assert rows[0]["b_h"] > 0


# ------------------------------------------------------------- 4. PER
def test_per_buffer_unit():
    _seed_all(6)
    env = gym.make("CartPole-v1")
    buf = FHRPrioritizedEpisodicReplayBuffer(
        64, env.observation_space, env.action_space, device="cpu", n_envs=1,
        per_alpha=0.6)
    env.close()
    for t in range(40):
        obs = np.full((1, 4), float(t), dtype=np.float32)
        buf.add(obs, obs, np.array([0]), np.array([1.0]),
                np.array([(t + 1) % 10 == 0]), [{}])
    buf.sample(16)
    assert buf.last_batch_inds.shape == (16,)
    assert (buf.last_batch_inds < 40).all()
    assert buf.last_batch_weights.shape == (16,)
    assert buf.last_batch_weights.max() == pytest.approx(1.0)
    assert (buf.last_batch_weights <= 1.0 + 1e-9).all()

    # boosting one slot's priority makes it dominate the draws
    buf.update_priorities(np.array([7]), np.array([1000.0]))
    counts = 0
    for _ in range(30):
        buf.sample(16)
        counts += int((buf.last_batch_inds == 7).sum())
    assert counts > 100

    # predecessors still resolve identically to the uniform buffer's logic
    keep, pred = buf.predecessors(buf.last_batch_inds, 2)
    t_ok = buf.t_in_episode[buf.last_batch_inds] >= 2
    assert (keep <= t_ok).all()

    # ring overwrite resets the slot to the (large) max priority
    for t in range(70):
        obs = np.zeros((1, 4), dtype=np.float32)
        buf.add(obs, obs, np.array([0]), np.array([0.0]),
                np.array([False]), [{}])
    leaf = buf.tree[buf.buffer_size - 1:]
    assert (leaf > 0).all()          # every slot written at least once


def test_per_plumbing_and_dqn_rejection():
    with pytest.raises(ValueError, match="prioritized_replay"):
        FHRDQN("MlpPolicy", gym.make("CartPole-v1"), prioritized_replay=True,
               policy_kwargs=dict(net_arch=[16]), buffer_size=200,
               device="cpu", seed=1)

    _seed_all(7)
    model = _filled_sacd(fhr_weight=0.5, warmup_grad_steps=0,
                         prioritized_replay=True, learning_starts=0)
    assert isinstance(model.replay_buffer, FHRPrioritizedEpisodicReplayBuffer)
    total_before = float(model.replay_buffer.tree[0])
    model._current_progress_remaining = 0.5
    model.train(gradient_steps=4, batch_size=32)
    assert model.replay_buffer.per_beta == pytest.approx(0.4 + 0.6 * 0.5)
    assert float(model.replay_buffer.tree[0]) != total_before  # priorities updated
    rows = model.drain_diagnostics()
    assert rows and rows[0]["b_h"] >= 0

    # flag/buffer consistency is enforced
    with pytest.raises(ValueError, match="replay buffer"):
        FHRSACD("MlpPolicy", gym.make("CartPole-v1"),
                prioritized_replay=True,
                replay_buffer_class=FHREpisodicReplayBuffer,
                **_sac_kwargs())


# ------------------------------------------------------- 5. save / load
def test_save_load_roundtrip(tmp_path):
    _seed_all(8)
    model = _filled_sac(fhr_weight=0.5, warmup_grad_steps=0,
                        c_predictor="shared", learning_starts=0)
    model.train(gradient_steps=3, batch_size=32)
    path = tmp_path / "fhrsac.zip"
    model.save(path)
    loaded = FHRSAC.load(path, device="cpu")
    assert loaded.fhr_weight == pytest.approx(0.5)
    assert loaded.c_predictor == "shared"
    for (k, a), (_, b) in zip(model.fhr_predictor.state_dict().items(),
                              loaded.fhr_predictor.state_dict().items()):
        assert torch.equal(a, b), k
    for k in model.policy.state_dict():
        assert torch.equal(model.policy.state_dict()[k],
                           loaded.policy.state_dict()[k]), k

    _seed_all(9)
    sacd = _filled_sacd(fhr_weight=0.2, warmup_grad_steps=0,
                        learning_starts=0)
    sacd.train(gradient_steps=3, batch_size=32)
    p2 = tmp_path / "fhrsacd.zip"
    sacd.save(p2)
    loaded2 = FHRSACD.load(p2, device="cpu")
    assert torch.equal(loaded2.fhr_head.c.detach(),
                       sacd.fhr_head.c.detach())
    assert torch.equal(loaded2.log_ent_coef.detach(),
                       sacd.log_ent_coef.detach())
    obs = torch.zeros(1, 4)
    assert int(loaded2.actor.logits(obs).argmax()) == \
        int(sacd.actor.logits(obs).argmax())


# --------------------------------------- 6. eval callback + adapters
def test_greedy_eval_callback_box_and_discrete(tmp_path):
    _seed_all(10)
    model = FHRSAC("MlpPolicy", gym.make("Pendulum-v1"),
                   **_sac_kwargs(learning_starts=50))
    eval_env = gym.make("Pendulum-v1")
    cb = GreedyEvalCallback(eval_env, tmp_path, freq_steps=100, n_episodes=1)
    model.learn(total_timesteps=150, callback=cb)
    rows = list(csv.DictReader(open(tmp_path / "eval.csv")))
    assert rows and float(rows[0]["mean_reward"]) <= 0.0
    eval_env.close()
    model.env.close()

    d = tmp_path / "disc"
    d.mkdir()
    _seed_all(11)
    model = FHRSACD("MlpPolicy", gym.make("CartPole-v1"),
                    **_sac_kwargs(learning_starts=50))
    eval_env = gym.make("CartPole-v1")
    cb = GreedyEvalCallback(eval_env, d, freq_steps=100, n_episodes=1)
    model.learn(total_timesteps=150, callback=cb)
    rows = list(csv.DictReader(open(d / "eval.csv")))
    assert rows and float(rows[0]["mean_reward"]) >= 1.0
    eval_env.close()
    model.env.close()


def test_sacd_adapter_analysis_surfaces():
    _seed_all(12)
    model = _filled_sacd(fhr_weight=0.0)
    adapter = model.qagent_adapter(epsilon=0.0)
    assert isinstance(adapter, SB3SACDAdapter)
    state_t = torch.zeros(1, 4)
    rows = adapter.policy_net(state_t)
    assert rows.shape == (1, 2)
    assert isinstance(adapter.act_greedy(state_t), int)
    assert adapter.pi(np.zeros(4)) in (0, 1)

    env = gym.make("CartPole-v1")
    env = type("B", (gym.Wrapper,), {})(env)
    env.observation_space = gym.spaces.Box(
        low=np.array([-2, -2, -0.2, -2], dtype=np.float32),
        high=np.array([2, 2, 0.2, 2], dtype=np.float32))
    mat = q_matrix_dqn(agent=adapter, env=env,
                       state_discretisation=[3, 3, 3, 3], batch_size=16)
    assert mat.shape == (81, 2)
    seqs = collect_hankel_sequences(adapter, env, seed=5)
    assert len(seqs["Hankel Q"]) > 0
    env.close()


def test_hankel_rollout_continuous_method():
    _seed_all(13)
    model = FHRSAC("MlpPolicy", gym.make("Pendulum-v1"),
                   **_sac_kwargs(learning_starts=50))
    model.learn(total_timesteps=120)
    adapter = model.qagent_adapter(epsilon=0.0)
    assert isinstance(adapter, SB3SACAdapter)

    env = gym.make("Pendulum-v1")
    mats = hankel_rollout_continuous(adapter, env, n_rollouts=2, base_seed=3)
    assert isinstance(mats, tuple) and len(mats) == 2   # Q trace + 1 act dim
    h_q, h_a = mats
    # Pendulum truncates at 200: per-rollout Hankel is (101, 100), stacked x2
    assert h_q.shape == (202, 100) and h_a.shape == (202, 100)
    assert np.isfinite(h_q).all() and np.isfinite(h_a).all()
    env.close()

    disc = gym.make("CartPole-v1")
    with pytest.raises(ValueError, match="continuous-action-only"):
        hankel_rollout_continuous(adapter, disc)
    disc.close()
    model.env.close()


def test_fhr_params_include_per_keys():
    for key in ("prioritized_replay", "per_alpha", "per_beta0"):
        assert key in FHR_PARAMS


# ------------------------------------------- 8. penalised-window rank probe
def test_window_rank_probe_lambda0_bit_exact_and_rows():
    # the probe consumes no RNG and adds no gradients: a lambda=0 run with it
    # enabled must still match stock SAC bit-for-bit
    _seed_all(0)
    stock = SAC("MlpPolicy", gym.make("Pendulum-v1"), **_sac_kwargs())
    stock.learn(total_timesteps=400)
    stock_state = _policy_state(stock)
    stock.env.close()

    _seed_all(0)
    probed = FHRSAC("MlpPolicy", gym.make("Pendulum-v1"),
                    **_sac_kwargs(fhr_weight=0.0, window_rank_every=10,
                                  window_rank_lags=6))
    probed.learn(total_timesteps=400)
    probed_state = _policy_state(probed)
    probed.env.close()
    assert stock_state.keys() == probed_state.keys()
    for k in stock_state:
        assert torch.equal(stock_state[k], probed_state[k]), k

    rows, arrays = probed.drain_window_rank()
    assert rows and arrays
    # padded, stable schema across all rows (nan-filled where short)
    keys = list(rows[0])
    assert all(list(r) == keys for r in rows)
    assert {"sv_01", "sv_07", "pen_sv_01", "pen_sv_03"} <= set(keys)
    assert "sv_08" not in keys and "pen_sv_04" not in keys   # L=6, r=2
    # both critics probed at each tick
    assert {r["critic"] for r in rows} == {0, 1}
    populated = [r for r in rows if r["n_windows"] > 0]
    assert populated
    r0 = populated[-1]
    assert r0["sv_01"] >= r0["sv_02"] >= 0.0                 # sorted spectrum
    assert r0["pen_sv_01"] >= r0["pen_sv_02"]
    W = next(iter(arrays.values()))
    assert W.shape[1] == rows[0]["window_len"] == 7
    # a second drain is empty (state was handed over, not copied)
    assert probed.drain_window_rank() == ([], {})


def test_window_rank_probe_sacd_and_dqn():
    _seed_all(5)
    model = _filled_sacd(fhr_weight=0.0, learning_starts=0,
                         window_rank_every=3, window_rank_lags=4)
    model.train(gradient_steps=6, batch_size=32)
    rows, _ = model.drain_window_rank()
    assert rows and any(r["n_windows"] > 0 for r in rows)
    assert {r["critic"] for r in rows} == {0, 1}

    _seed_all(6)
    dqn = FHRDQN("MlpPolicy", gym.make("CartPole-v1"),
                 policy_kwargs=dict(net_arch=[32, 32]), buffer_size=1000,
                 learning_starts=0, batch_size=32, seed=7, device="cpu",
                 verbose=0, window_rank_every=2, window_rank_lags=4)
    from stable_baselines3.common.logger import configure as _cfg
    dqn.set_logger(_cfg(folder=None, format_strings=[]))
    _fill_buffer(dqn, "CartPole-v1")
    dqn.train(gradient_steps=4, batch_size=32)
    rows, arrays = dqn.drain_window_rank()
    assert rows and any(r["n_windows"] > 0 for r in rows)
    assert {r["critic"] for r in rows} == {0}                # single critic
    assert all(k.endswith("_c0") for k in arrays)


# ------------------------------------------------- fhr_lag_source variants
@pytest.mark.parametrize("family", ["sac", "sacd"])
def test_fhr_lag_source_variants(family):
    """detached/target train cleanly on both critics; at init critic_target
    == critic, so the first-step penalty value matches online. A perturbed
    critic_target then shifts the target-source penalty but not detached's."""
    make = _filled_sac if family == "sac" else _filled_sacd
    with pytest.raises(ValueError, match="fhr_lag_source"):
        make(fhr_weight=0.5, fhr_lag_source="bogus")
    rows = {}
    for src in ("online", "detached", "target"):
        _seed_all(5)
        m = make(fhr_weight=0.5, fhr_order=2, warmup_grad_steps=0,
                 learning_starts=0, fhr_lag_source=src)
        _seed_all(123)
        m.train(gradient_steps=1, batch_size=32)
        rows[src] = m.drain_diagnostics()
        assert m.nan_skips == 0
    r0 = rows["online"][0]
    assert r0["b_h"] > 0 and np.isfinite(r0["penalty_raw"])
    assert abs(rows["detached"][0]["penalty_raw"] - r0["penalty_raw"]) < 1e-6
    assert abs(rows["target"][0]["penalty_raw"] - r0["penalty_raw"]) < 1e-6
    # target source actually reads critic_target
    _seed_all(5)
    mt = make(fhr_weight=0.5, fhr_order=2, warmup_grad_steps=0,
              learning_starts=0, fhr_lag_source="target")
    with torch.no_grad():
        for p in mt.critic_target.parameters():
            p.add_(0.5)
    _seed_all(123)
    mt.train(gradient_steps=1, batch_size=32)
    shifted = mt.drain_diagnostics()[0]["penalty_raw"]
    assert abs(shifted - rows["detached"][0]["penalty_raw"]) > 1e-6


def test_fhr_lag_source_target_twins_shared_by_continuous_hosts():
    """FHRSAC's target twin lives on _FHRContinuousCriticMixin (so FHRTD3
    inherits it) and reproduces critic_target's own forward per critic."""
    from agents.sb3_sac_fhr import _FHRContinuousCriticMixin
    from agents.sb3_td3_fhr import FHRTD3
    assert (FHRSAC._fhr_target_lag_q_fns
            is _FHRContinuousCriticMixin._fhr_target_lag_q_fns)
    assert (FHRTD3._fhr_target_lag_q_fns
            is _FHRContinuousCriticMixin._fhr_target_lag_q_fns)
    _seed_all(2)
    m = _filled_sac(fhr_weight=0.5, warmup_grad_steps=0, learning_starts=0,
                    fhr_lag_source="target")
    m.train(gradient_steps=2, batch_size=32)
    with torch.no_grad():
        for p in m.critic_target.parameters():
            p.add_(0.5)
    replay = m.replay_buffer.sample(16)
    obs, acts = replay.observations, replay.actions
    fns = m._fhr_target_lag_q_fns()
    assert len(fns) == 2
    with torch.no_grad():
        ref = m.critic_target(obs, acts)
        onl = m.critic(obs, acts)
        for i, fn in enumerate(fns):
            assert torch.equal(fn(obs, acts), ref[i].squeeze(1))
            assert not torch.allclose(fn(obs, acts), onl[i].squeeze(1))
    # discrete twin: FHRSACD gathers per head from critic_target(obs)
    _seed_all(2)
    d = _filled_sacd(fhr_weight=0.5, warmup_grad_steps=0, learning_starts=0,
                     fhr_lag_source="target")
    with torch.no_grad():
        for p in d.critic_target.parameters():
            p.add_(0.5)
    replay = d.replay_buffer.sample(16)
    obs, acts = replay.observations, replay.actions.long()
    with torch.no_grad():
        ref = d.critic_target(obs)
        for i, fn in enumerate(d._fhr_target_lag_q_fns()):
            assert torch.equal(fn(obs, acts), ref[i].gather(1, acts).squeeze(1))
