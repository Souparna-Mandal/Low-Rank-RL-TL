"""Tests for agents.sb3_td3_fhr: FHRTD3 (continuous, deterministic actor),
the TD3-native penalty scale (sum over critics against TD3's sum-over-critics
TD term), the gradient-stream probe, the lambda=0 bit-exactness contracts
(with every probe on), policy_delay parity across NaN skips, save/load, PER
plumbing, the analysis adapter and the launcher wiring. CPU-only, tiny nets
and budgets."""
import pathlib
import random
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "stable_baselines_3" / "src"))

from stable_baselines3 import SAC, TD3                             # noqa: E402
from stable_baselines3.common.logger import configure              # noqa: E402
from stable_baselines3.common.noise import (NormalActionNoise,     # noqa: E402
                                            OrnsteinUhlenbeckActionNoise)

from agents.sb3_fhr import FHR_PARAMS, FHREpisodicReplayBuffer     # noqa: E402
from agents.sb3_sac_fhr import (FHRSAC,                            # noqa: E402
                                FHRPrioritizedEpisodicReplayBuffer,
                                SB3SACAdapter)
from agents.sb3_td3_fhr import FHRTD3, SB3TD3Adapter               # noqa: E402
from analysis.low_rank.continuous_rollout import hankel_rollout_continuous  # noqa: E402


# --------------------------------------------------------------- helpers
def _seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _noise(n=1, std=0.1):
    """A FRESH noise object per model: OU noise is stateful, and sharing one
    instance between the stock and FHR runs would desync them."""
    return NormalActionNoise(np.zeros(n), std * np.ones(n))


def _td3_kwargs(**over):
    kw = dict(policy_kwargs=dict(net_arch=[32, 32]), buffer_size=1000,
              learning_starts=100, batch_size=32, train_freq=1,
              gradient_steps=1, seed=7, device="cpu", verbose=0)
    kw.update(over)
    return kw


def _fill_buffer(model, env_id, n_steps=220, seed=1):
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


def _filled_td3(**over):
    model = FHRTD3("MlpPolicy", gym.make("Pendulum-v1"),
                   action_noise=_noise(), **_td3_kwargs(**over))
    _null_logger(model)
    _fill_buffer(model, "Pendulum-v1")
    return model


def _policy_state(model):
    return {k: v.detach().clone()
            for k, v in model.policy.state_dict().items()}


def _final_state(cls, extra, steps=400, seed=0):
    _seed_all(seed)
    model = cls("MlpPolicy", gym.make("Pendulum-v1"), action_noise=_noise(),
                **_td3_kwargs(**extra))
    model.learn(total_timesteps=steps)
    state = _policy_state(model)
    model.env.close()
    return state, model


# ------------------------------------------- 1. lambda=0 bit-exactness
def test_fhrtd3_lambda0_bit_exact_matches_stock_td3():
    stock, _ = _final_state(TD3, {})
    fhr, _ = _final_state(FHRTD3, dict(fhr_weight=0.0))
    assert stock.keys() == fhr.keys()
    # the state dict carries the targets, so the delayed polyak cadence is
    # covered by the same comparison
    assert any("actor_target" in k for k in stock)
    assert any("critic_target" in k for k in stock)
    for k in stock:
        assert torch.equal(stock[k], fhr[k]), k


def test_fhrtd3_lambda0_bit_exact_with_every_probe_on():
    # both probes are measurements: no RNG, no .grad, no parameter change
    stock, _ = _final_state(TD3, {})
    probed, model = _final_state(FHRTD3, dict(fhr_weight=0.0, grad_probe_every=1,
                                              window_rank_every=10,
                                              window_rank_lags=6))
    for k in stock:
        assert torch.equal(stock[k], probed[k]), k
    rows = model.drain_diagnostics()
    assert rows and any(np.isfinite(r["grad_ratio"]) for r in rows)
    wrows, _ = model.drain_window_rank()
    assert wrows and {r["critic"] for r in wrows} == {0, 1}


def test_fhrtd3_ccond_lambda0_bit_exact():
    # the ccond predictor's construction must not shift the torch RNG stream
    # the target-policy smoothing noise draws from (fork_rng guard)
    stock, _ = _final_state(TD3, {}, steps=300)
    for mode in ("separate", "shared"):
        fhr, _ = _final_state(FHRTD3, dict(fhr_weight=0.0, c_predictor=mode),
                              steps=300)
        for k in stock:
            assert torch.equal(stock[k], fhr[k]), (mode, k)


# ----------------------------------------- 2. the TD3-native penalty scale
def test_penalty_is_sum_over_critics_on_td3_scale():
    assert FHRSAC._fhr_critic_reduction == "mean"
    assert FHRTD3._fhr_critic_reduction == "sum"
    _seed_all(3)
    model = _filled_td3(fhr_weight=0.5, fhr_order=2, warmup_grad_steps=0,
                        learning_starts=0)
    buf = model.replay_buffer
    replay = buf.sample(32)
    qs = [q.squeeze(1) for q in model.critic(replay.observations, replay.actions)]
    assert len(qs) == 2
    d_sum = model._fhr_base_diag(td_loss=1.0, lam=0.5)
    pen_sum = model._fhr_penalty_multi(qs, model._lag_q_fns(), 0.5, d_sum)
    model._fhr_critic_reduction = "mean"
    d_mean = model._fhr_base_diag(td_loss=1.0, lam=0.5)
    pen_mean = model._fhr_penalty_multi(qs, model._lag_q_fns(), 0.5, d_mean)
    model._fhr_critic_reduction = "sum"
    assert pen_sum is not None and d_sum["b_h"] > 0
    # sum_i Huber_i, not the mean: exactly N times the SAC-convention value
    assert float(pen_sum.detach()) == pytest.approx(2 * float(pen_mean.detach()), rel=1e-6)
    assert d_sum["penalty_raw"] == pytest.approx(2 * d_mean["penalty_raw"], rel=1e-6)
    assert d_sum["penalty_weighted"] == pytest.approx(0.5 * d_sum["penalty_raw"])
    assert d_sum["loss_ratio"] == pytest.approx(d_sum["penalty_raw"] / 1.0)
    assert d_sum["rho_loss"] == pytest.approx(0.5 * d_sum["loss_ratio"])

    # td_loss diag is the stock sum-over-critics critic loss SB3 itself logs
    model.train(gradient_steps=1, batch_size=32)
    rows = model.drain_diagnostics()
    assert rows[-1]["td_loss"] == pytest.approx(
        model.logger.name_to_value["train/critic_loss"], rel=1e-6)
    assert rows[-1]["rho_loss"] == pytest.approx(
        rows[-1]["penalty_weighted"] / rows[-1]["td_loss"], rel=1e-6)


# ------------------------------------------ 3. penalty engagement/wiring
def test_penalty_engages_after_warmup():
    _seed_all(3)
    model = _filled_td3(fhr_weight=0.5, fhr_order=2, warmup_grad_steps=3,
                        c_learning_rate=1e-2, learning_starts=0)
    c_before = model.fhr_head.c.detach().clone()
    model.train(gradient_steps=8, batch_size=32)
    rows = model.drain_diagnostics()
    assert rows and rows[0]["b_h"] > 0
    assert model._fhr_grad_steps == 8
    assert model._lambda_eff() == pytest.approx(0.5)
    assert not torch.equal(model.fhr_head.c.detach(), c_before)
    group = model.critic.optimizer.param_groups[-1]
    assert group["lr"] == pytest.approx(1e-2)
    assert any(p is model.fhr_head.c for p in group["params"])
    actor_params = {id(p) for g in model.actor.optimizer.param_groups
                    for p in g["params"]}
    assert id(model.fhr_head.c) not in actor_params
    model.train(gradient_steps=1, batch_size=32)
    assert model.critic.optimizer.param_groups[-1]["lr"] == pytest.approx(1e-2)


def test_c_gets_no_gradients_during_warmup_or_lambda0():
    _seed_all(4)
    for kwargs in (dict(fhr_weight=0.0),
                   dict(fhr_weight=1.0, warmup_grad_steps=10 ** 6)):
        model = _filled_td3(learning_starts=0, grad_probe_every=1, **kwargs)
        c_before = model.fhr_head.c.detach().clone()
        model.train(gradient_steps=4, batch_size=32)
        assert model.fhr_head.c.grad is None       # the probe never touches .grad
        assert torch.equal(model.fhr_head.c.detach(), c_before)


# ------------------------------------------------- 4. gradient-stream probe
def test_grad_probe_columns_and_baseline_measurement():
    _seed_all(4)
    model = _filled_td3(fhr_weight=0.5, warmup_grad_steps=0, learning_starts=0,
                        grad_probe_every=1)
    model.train(gradient_steps=1, batch_size=32)     # one step: no burst nanmean
    r = model.drain_diagnostics()[-1]
    for k in ("grad_norm_td", "grad_norm_pen", "grad_ratio", "grad_rho",
              "grad_cos", "loss_ratio", "rho_loss"):
        assert np.isfinite(r[k]), k
    assert r["grad_norm_td"] > 0 and r["grad_norm_pen"] > 0
    assert r["grad_ratio"] == pytest.approx(r["grad_norm_pen"] / r["grad_norm_td"])
    assert -1.0 <= r["grad_cos"] <= 1.0
    assert r["grad_rho"] == pytest.approx(0.5 * r["grad_ratio"])

    # off by default: columns exist (stable schema) but stay nan
    _seed_all(4)
    off = _filled_td3(fhr_weight=0.5, warmup_grad_steps=0, learning_starts=0)
    off.train(gradient_steps=1, batch_size=32)
    r_off = off.drain_diagnostics()[-1]
    assert "grad_ratio" in r_off and np.isnan(r_off["grad_ratio"])

    # cadence: every k-th gradient step only
    _seed_all(4)
    every3 = _filled_td3(fhr_weight=0.5, warmup_grad_steps=0, learning_starts=0,
                         grad_probe_every=3)
    for _ in range(6):
        every3.train(gradient_steps=1, batch_size=32)
    flags = [np.isfinite(r["grad_ratio"]) for r in every3.drain_diagnostics()]
    assert flags == [True, False, False, True, False, False]

    # the lambda=0 baseline measures the same streams: unweighted ratio > 0,
    # weighted ratios exactly 0, penalty never in the loss
    _seed_all(5)
    base = _filled_td3(fhr_weight=0.0, learning_starts=0, grad_probe_every=1)
    base.train(gradient_steps=1, batch_size=32)
    rb = base.drain_diagnostics()[-1]
    assert np.isfinite(rb["grad_ratio"]) and rb["grad_ratio"] > 0
    assert rb["grad_rho"] == 0.0 and rb["rho_loss"] == 0.0
    assert rb["penalty_weighted"] == 0.0 and rb["lambda_eff"] == 0.0


def test_grad_probe_on_sac_lambda0_bit_exact_and_logged():
    def state(cls, extra):
        _seed_all(0)
        m = cls("MlpPolicy", gym.make("Pendulum-v1"), **_td3_kwargs(**extra))
        m.learn(total_timesteps=300)
        st = _policy_state(m)
        m.env.close()
        return st, m
    stock, _ = state(SAC, {})
    probed, m = state(FHRSAC, dict(fhr_weight=0.0, grad_probe_every=1))
    for k in stock:
        assert torch.equal(stock[k], probed[k]), k
    rows = m.drain_diagnostics()
    assert rows and any(np.isfinite(r["grad_ratio"]) for r in rows)


# ------------------------------------------- 5. NaN skip vs policy_delay
def test_nan_skip_keeps_policy_delay_parity(monkeypatch):
    _seed_all(6)
    model = _filled_td3(fhr_weight=0.5, warmup_grad_steps=0, learning_starts=0)
    assert model.policy_delay == 2
    orig = model._fhr_penalty_multi
    calls = {"n": 0}

    def nan_on_second_call(*a, **kw):
        calls["n"] += 1
        out = orig(*a, **kw)
        return out * float("nan") if calls["n"] == 2 else out
    monkeypatch.setattr(model, "_fhr_penalty_multi", nan_on_second_call)

    for step in range(4):
        prev = {k: v.clone() for k, v in model.actor.state_dict().items()}
        model.train(gradient_steps=1, batch_size=32)
        assert model._n_updates == step + 1          # stock: counted at loop top
        moved = any(not torch.equal(prev[k], v)
                    for k, v in model.actor.state_dict().items())
        delayed = model._n_updates % model.policy_delay == 0
        skipped = step == 1                            # the NaN step (n_updates 2)
        assert moved == (delayed and not skipped), (step, moved, delayed)
    assert model.nan_skips == 1
    rows = model.drain_diagnostics()
    assert [np.isfinite(r["actor_loss"]) for r in rows] == [False, False, False, True]
    assert rows[1]["nan_skips"] == 1


# ------------------------------------------------------- 6. save / load
def test_save_load_roundtrip(tmp_path):
    _seed_all(8)
    model = _filled_td3(fhr_weight=0.5, warmup_grad_steps=0, c_predictor="shared",
                        learning_starts=0, grad_probe_every=5)
    model.train(gradient_steps=3, batch_size=32)
    path = tmp_path / "fhrtd3.zip"
    model.save(path)
    loaded = FHRTD3.load(path, device="cpu")
    assert loaded.fhr_weight == pytest.approx(0.5)
    assert loaded.c_predictor == "shared" and loaded.grad_probe_every == 5
    assert loaded._fhr_critic_reduction == "sum"
    assert isinstance(loaded.action_noise, NormalActionNoise)
    for (k, a), (_, b) in zip(model.fhr_predictor.state_dict().items(),
                              loaded.fhr_predictor.state_dict().items()):
        assert torch.equal(a, b), k
    for k in model.policy.state_dict():
        assert torch.equal(model.policy.state_dict()[k],
                           loaded.policy.state_dict()[k]), k


# ------------------------------------------------ 7. adapter + analyses
def test_hankel_rollout_continuous_method():
    _seed_all(13)
    model = FHRTD3("MlpPolicy", gym.make("Pendulum-v1"), action_noise=_noise(),
                   **_td3_kwargs(learning_starts=50))
    model.learn(total_timesteps=120)
    adapter = model.qagent_adapter(epsilon=0.0)
    assert isinstance(adapter, SB3TD3Adapter) and isinstance(adapter, SB3SACAdapter)
    assert adapter.policy_net is None
    env = gym.make("Pendulum-v1")
    mats = hankel_rollout_continuous(adapter, env, n_rollouts=2, base_seed=3)
    assert isinstance(mats, tuple) and len(mats) == 2
    h_q, h_a = mats
    assert h_q.shape == (202, 100) and h_a.shape == (202, 100)
    assert np.isfinite(h_q).all() and np.isfinite(h_a).all()
    # deterministic: the same rollout twice is identical (no exploration noise)
    mats2 = hankel_rollout_continuous(adapter, env, n_rollouts=2, base_seed=3)
    assert np.array_equal(mats2[0], h_q)
    env.close()
    model.env.close()


# ---------------------------------------------------------------- 8. PER
def test_per_plumbing():
    _seed_all(7)
    model = _filled_td3(fhr_weight=0.5, warmup_grad_steps=0,
                        prioritized_replay=True, learning_starts=0)
    assert isinstance(model.replay_buffer, FHRPrioritizedEpisodicReplayBuffer)
    total_before = float(model.replay_buffer.tree[0])
    model._current_progress_remaining = 0.5
    model.train(gradient_steps=4, batch_size=32)
    assert model.replay_buffer.per_beta == pytest.approx(0.4 + 0.6 * 0.5)
    assert float(model.replay_buffer.tree[0]) != total_before
    rows = model.drain_diagnostics()
    assert rows and rows[0]["b_h"] >= 0
    with pytest.raises(ValueError, match="replay buffer"):
        FHRTD3("MlpPolicy", gym.make("Pendulum-v1"), prioritized_replay=True,
               replay_buffer_class=FHREpisodicReplayBuffer, **_td3_kwargs())


# ------------------------------------------------- 9. contracts + launcher
def test_fhr_params_contract_and_launcher_mirror():
    import run_sb3_seeds as R
    assert "grad_probe_every" in FHR_PARAMS
    assert set(R.FHR_PARAMS) == set(FHR_PARAMS)
    assert R._algo_class("td3") is FHRTD3


def test_launcher_builds_td3_with_action_noise():
    import run_sb3_seeds as R
    env = gym.make("Pendulum-v1")
    cfg = {"algo": {"type": "td3", "n_timesteps": 100, "learning_rate": 1e-3,
                    "buffer_size": 500, "learning_starts": 10, "batch_size": 16,
                    "gamma": 0.99, "tau": 0.005, "train_freq": 1,
                    "gradient_steps": 1, "policy_delay": 2,
                    "target_policy_noise": 0.2, "target_noise_clip": 0.5,
                    "noise_type": "normal", "noise_std": 0.1,
                    "net_arch": [8, 8], "n_critics": 2},
           "agent": {"fhr_weight": 0.1, "fhr_order": 2, "grad_probe_every": 5},
           "experiment": {"_device": "cpu"}}
    model = R._build_model(cfg, env, seed=1)
    assert isinstance(model, FHRTD3)
    assert isinstance(model.action_noise, NormalActionNoise)
    assert model.action_noise._sigma.shape == (1,)
    assert model.action_noise._sigma[0] == pytest.approx(0.1)
    assert model.grad_probe_every == 5 and model.fhr_weight == pytest.approx(0.1)
    assert len(model.critic.q_networks) == 2
    cfg["algo"]["noise_type"] = "ornstein-uhlenbeck"
    ou = R._build_model(cfg, env, seed=1)
    assert isinstance(ou.action_noise, OrnsteinUhlenbeckActionNoise)
    env.close()
