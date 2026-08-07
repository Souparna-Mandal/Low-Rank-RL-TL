"""Smoke tests for PPOAgent + ppo_training_loop and the config_ppo.yaml flow.
Run as `python tests/test_ppo.py` or via pytest from repo root."""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import gymnasium as gym

from agents.ppo_agent import PPOAgent, compute_gae
from ppo_training import (ppo_training_loop, _collect_rollout,
                          greedy_episode_return, frozen_running_stats)
from environments.base_env import make_environment
from experiment import load_config, build_env, build_ppo_agent, train_ppo

REPO = pathlib.Path(__file__).resolve().parents[1]

NO_TRANSFORMS = {"discrete_config": None,
                 "normalise": {"action": {}, "state": {}},
                 "clip": {"action": False, "state": []}}


def _pendulum(**over):
    """Pendulum-v1 through make_environment with the transform keys merged."""
    kw = {k: dict(v) if isinstance(v, dict) else v
          for k, v in NO_TRANSFORMS.items()}
    for k, v in over.items():
        kw[k] = {**kw[k], **v} if isinstance(kw.get(k), dict) else v
    return make_environment("Pendulum-v1", **kw)


def _running(**over):
    return {"obs": True, "reward": True, "gamma": 0.99,
            "clip_obs": 10.0, "clip_reward": 10.0, **over}


def _seed(s=0):
    torch.manual_seed(s)
    np.random.seed(s)


def _agent(**overrides):
    kwargs = dict(env=gym.make("CartPole-v1"), device="cpu",
                  rollout_steps=96, minibatch_size=32, update_epochs=2)
    kwargs.update(overrides)
    return PPOAgent(**kwargs)


class _DiscreteObsEnv(gym.Env):
    """Discrete observations, which PPOAgent still refuses (use one_hot_obs)."""
    observation_space = gym.spaces.Discrete(5)
    action_space = gym.spaces.Discrete(2)


class _ContinuousEnv(gym.Env):
    """3-D Box actions, to exercise the sum-over-action-dims paths that a 1-D
    env like Pendulum cannot distinguish from a scalar."""
    observation_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
    action_space = gym.spaces.Box(-2.0, 2.0, (3,), dtype=np.float32)

    def __init__(self, ep_len=10):
        self.ep_len = ep_len
        self.t = 0

    def reset(self, seed=None, options=None):
        self.t = 0
        return np.zeros(4, np.float32), {}

    def step(self, action):
        self.t += 1
        return np.zeros(4, np.float32), 1.0, self.t >= self.ep_len, False, {}


def test_flat_obs_assert_still_enforced():
    try:
        PPOAgent(env=_DiscreteObsEnv())
        raise RuntimeError("Discrete observation space should have been rejected")
    except AssertionError as e:
        assert "flat Box observation space" in str(e)


def test_box_action_space_accepted():
    agent = PPOAgent(env=gym.make("Pendulum-v1"))
    assert agent.continuous and agent.act_dim == 1
    assert agent.log_std is not None and agent.log_std.shape == (1,)
    # Discrete stays discrete, with no log_std to optimise.
    disc = _agent()
    assert not disc.continuous and disc.log_std is None


def test_continuous_rollout_shapes_and_update():
    _seed(6)
    env = _ContinuousEnv()
    agent = PPOAgent(env=env, rollout_steps=64, minibatch_size=16,
                     update_epochs=2)
    state, _ = env.reset(seed=0)
    buf, _, _, finished = _collect_rollout(agent, env, state, 0.0)
    assert buf["acts"].shape == (64, 3) and buf["acts"].dtype == np.float32
    assert np.isfinite(buf["logps"]).all() and len(finished) > 0
    before = [p.detach().clone() for p in agent.actor.parameters()]
    agent.update(buf)
    assert any(not torch.equal(b, p.detach())
               for b, p in zip(before, agent.actor.parameters()))
    for p in agent._all_params():
        assert torch.isfinite(p).all()


def test_log_std_is_optimised():
    """Guards the silent failure: a subclass that rebuilds the optimiser over
    actor+critic only would leave log_std frozen and never error."""
    _seed(7)
    env = _ContinuousEnv()
    agent = PPOAgent(env=env, rollout_steps=64, minibatch_size=16,
                     update_epochs=2, ent_coef=0.01)
    assert any(p is agent.log_std for p in agent._all_params())
    state, _ = env.reset(seed=0)
    buf, _, _, _ = _collect_rollout(agent, env, state, 0.0)
    before = agent.log_std.detach().clone()
    agent.update(buf)
    assert agent.log_std.grad is not None, "log_std received no gradient"
    assert not torch.equal(before, agent.log_std.detach()), \
        "log_std was not stepped by the optimiser"


def test_act_greedy_continuous_returns_mean():
    _seed(8)
    env = _ContinuousEnv()
    agent = PPOAgent(env=env, rollout_steps=16)
    obs = np.zeros(4, np.float32)
    greedy = agent.act_greedy(obs)
    assert greedy.shape == (3,)
    with torch.no_grad():
        mean = agent.actor(torch.as_tensor(obs).unsqueeze(0)).squeeze(0).numpy()
    assert np.allclose(greedy, mean)


def test_continuous_logp_sums_over_action_dims():
    """log_prob must be one scalar per timestep, not one per action dim."""
    _seed(9)
    agent = PPOAgent(env=_ContinuousEnv(), rollout_steps=16)
    obs = torch.zeros(5, 4)
    dist = agent._dist(obs)
    acts = dist.sample()
    assert acts.shape == (5, 3)
    assert agent._logp(dist, acts).shape == (5,)
    assert agent._entropy(dist).shape == (5,)


def test_collect_rollout_bookkeeping():
    _seed(2)
    agent = _agent()
    env = gym.make("CartPole-v1")
    state, _ = env.reset(seed=7)
    buf, _, _, finished = _collect_rollout(agent, env, state, 0.0)
    n = agent.rollout_steps
    # Segments partition [0, n) in order.
    assert buf["seg_bounds"][0][0] == 0 and buf["seg_bounds"][-1][1] == n
    for (a0, b0), (a1, _) in zip(buf["seg_bounds"], buf["seg_bounds"][1:]):
        assert b0 == a1 and a0 < b0
    # Every segment is a finished episode except a possible rollout-cut tail
    # (absent when the last episode ends exactly at the rollout boundary).
    n_seg = len(buf["seg_terminal"])
    assert len(finished) in (n_seg, n_seg - 1)
    if len(finished) == n_seg - 1:  # cut mid-episode: non-terminal, bootstraps
        assert buf["seg_terminal"][-1] is False
        assert np.isfinite(buf["seg_boot_value"][-1])


def test_update_moves_actor():
    _seed(3)
    agent = _agent()
    env = gym.make("CartPole-v1")
    state, _ = env.reset(seed=3)
    buf, _, _, _ = _collect_rollout(agent, env, state, 0.0)
    before = [p.detach().clone() for p in agent.actor.parameters()]
    agent.update(buf)
    changed = any(not torch.equal(b, p.detach())
                  for b, p in zip(before, agent.actor.parameters()))
    assert changed, "update() did not move the actor"
    for p in list(agent.actor.parameters()) + list(agent.critic.parameters()):
        assert torch.isfinite(p).all()


def test_config_flow_end_to_end():
    cfg = load_config(REPO / "experiments" / "dqn_cartpole" / "config_ppo.yaml")
    cfg["agent"].update(rollout_steps=128, minibatch_size=64, update_epochs=2)
    cfg["training"].update(no_episodes=3, no_eps_to_avg=2, solved_reward=1e9)
    env = build_env(cfg)
    agent = build_ppo_agent(cfg, env)
    rewards = train_ppo(cfg, agent, env, progress=False)
    assert len(rewards) == 3 and all(np.isfinite(r) for r in rewards)


class _NeverEndingEnv(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        return np.zeros(2, np.float32), {}

    def step(self, action):
        return np.zeros(2, np.float32), 0.0, False, False, {}


def test_training_loop_raises_on_never_terminating_env():
    _seed(5)
    env = _NeverEndingEnv()
    agent = _agent(env=env, rollout_steps=16)
    try:
        ppo_training_loop(agent, env, no_episodes=5, solved_reward=1e9,
                          np_seed=1, progress=False)
        raise AssertionError("expected RuntimeError for never-terminating env")
    except RuntimeError as e:
        assert "time_limit" in str(e)


def test_training_loop_solved_early_stop():
    _seed(4)
    agent = _agent(rollout_steps=64)
    env = gym.make("CartPole-v1")
    # solved_reward=0 with tiny patience: stops at the patience count, not the
    # full budget.
    rewards = ppo_training_loop(agent, env, no_episodes=10_000, solved_reward=0,
                                early_stopping_patience_eps=2, np_seed=11,
                                no_eps_to_avg=2, progress=False)
    assert 2 <= len(rewards) < 100


def _gae_buf(rews, values, seg_bounds, seg_terminal, seg_boot):
    return {"rews": np.asarray(rews, np.float64),
            "values": np.asarray(values, np.float64),
            "seg_bounds": seg_bounds, "seg_terminal": seg_terminal,
            "seg_boot_value": seg_boot}


def test_compute_gae_terminal_vs_truncated():
    """The distinction the shared helper exists to protect: a terminal segment
    bootstraps 0, a truncated one bootstraps seg_boot_value. At gamma=lam=1
    both have a closed form, so this pins the semantics exactly."""
    rews, values = [1.0, 2.0, 3.0], [0.5, 0.25, 0.125]
    tail = np.cumsum(rews[::-1])[::-1]  # sum(rews[t:])

    adv, ret = compute_gae(
        _gae_buf(rews, values, [(0, 3)], [True], [99.0]), gamma=1.0, lam=1.0)
    assert np.allclose(adv, tail - np.asarray(values))
    assert np.allclose(ret, tail), "terminal segment must ignore seg_boot_value"

    boot = 7.0
    adv_t, ret_t = compute_gae(
        _gae_buf(rews, values, [(0, 3)], [False], [boot]), gamma=1.0, lam=1.0)
    assert np.allclose(ret_t, tail + boot), "truncated segment must bootstrap"
    assert not np.allclose(ret, ret_t)


def test_compute_gae_segments_are_independent():
    """A segment must not leak advantage across its boundary."""
    rews, values = [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]
    split, _ = compute_gae(
        _gae_buf(rews, values, [(0, 2), (2, 4)], [True, True], [0.0, 0.0]),
        gamma=1.0, lam=1.0)
    whole, _ = compute_gae(
        _gae_buf(rews[:2], values[:2], [(0, 2)], [True], [0.0]),
        gamma=1.0, lam=1.0)
    assert np.allclose(split[:2], whole) and np.allclose(split[2:], whole)


def test_every_variant_uses_the_shared_gae():
    """Guards the drift the refactor removed: no variant may carry its own
    copy of the GAE recursion."""
    import agents.ppo_agent as ppo_mod
    root = pathlib.Path(ppo_mod.__file__).parent
    offenders = [p.name for p in sorted(root.rglob("*.py"))
                 if "adv[t] = gae" in p.read_text() and p.name != "ppo_agent.py"]
    assert not offenders, f"re-derived GAE loop in: {offenders}"
    # and the one surviving copy is the shared helper, reachable from an agent
    assert PPOAgent._gae(_agent(), _gae_buf(
        [1.0], [0.0], [(0, 1)], [True], [0.0]))[0].shape == (1,)


def test_rescale_then_clip_action_compose():
    """Regression: ClipAction re-advertises Box(-inf, inf), so RescaleAction
    applied AFTER it cannot build a rescale. base_env must rescale first."""
    env = _pendulum(normalise={"action": {"min": -1.0, "max": 1.0}},
                    clip={"action": True})
    env.reset(seed=0)
    env.step(np.array([9.0], np.float32))  # far out of bounds: must not raise
    agent = PPOAgent(env=env, rollout_steps=16)
    assert agent.continuous and agent.act_dim == 1


def test_running_normalisation_reports_raw_and_normalised():
    _seed(11)
    env = _pendulum(normalise={"running": _running()})
    agent = PPOAgent(env=env, rollout_steps=400, minibatch_size=64,
                     update_epochs=1)
    state, _ = env.reset(seed=0)
    buf, _, _, finished = _collect_rollout(agent, env, state, 0.0)
    raw = buf["raw_returns"]
    assert len(raw) == len(finished) and len(raw) > 0
    # Pendulum's true return is very negative; the normalised one is rescaled
    # by the running return std, so the two streams must not coincide.
    assert all(r < -100 for r in raw), raw
    assert not np.allclose(raw, finished)
    # reward clipping is active: no single step exceeds the +-10 bound
    assert np.abs(buf["rews"]).max() <= 10.0 + 1e-6
    # observation normalisation is active: obs are standardised, not raw
    assert np.abs(buf["obs"]).max() <= 10.0 + 1e-6


def test_raw_returns_equal_returns_without_normalisation():
    """The two streams must coincide exactly when normalisation is off, which
    is what keeps every existing discrete result unchanged."""
    _seed(12)
    agent = _agent()
    env = gym.make("CartPole-v1")
    state, _ = env.reset(seed=3)
    buf, _, _, finished = _collect_rollout(agent, env, state, 0.0)
    assert buf["raw_returns"] == finished


def test_eval_does_not_move_running_stats():
    _seed(13)
    env = _pendulum(normalise={"running": _running()})
    agent = PPOAgent(env=env, rollout_steps=200, minibatch_size=64,
                     update_epochs=1)
    state, _ = env.reset(seed=0)
    _collect_rollout(agent, env, state, 0.0)  # prime the running stats

    def snapshot():
        out, e = [], env
        while e is not None:
            for attr in ("obs_rms", "return_rms"):
                rms = getattr(e, attr, None)
                if rms is not None:
                    out.append((np.copy(rms.mean), np.copy(rms.var)))
            e = getattr(e, "env", None)
        return out

    before = snapshot()
    assert before, "expected running statistics to exist"
    greedy_episode_return(agent, env, seed=1, max_steps=200)
    for (m0, v0), (m1, v1) in zip(before, snapshot()):
        assert np.array_equal(m0, m1) and np.array_equal(v0, v1), \
            "evaluation perturbed the running statistics"


def test_frozen_running_stats_restores_flag():
    env = _pendulum(normalise={"running": _running()})
    with frozen_running_stats(env):
        pass
    e, seen = env, False
    while e is not None:
        if hasattr(e, "update_running_mean"):
            assert e.update_running_mean is True
            seen = True
        e = getattr(e, "env", None)
    assert seen


def test_solved_threshold_uses_raw_returns():
    """With reward normalisation on, a normalised average has no fixed scale;
    the solved check must read the raw stream or it fires meaninglessly."""
    _seed(14)
    env = _pendulum(normalise={"running": _running()})
    agent = PPOAgent(env=env, rollout_steps=400, minibatch_size=64,
                     update_epochs=1)
    rewards, raw = ppo_training_loop(
        agent, env, no_episodes=4, solved_reward=1e9, np_seed=2,
        no_eps_to_avg=2, progress=False, return_raw=True)
    assert len(rewards) == len(raw) == 4
    assert all(r < -100 for r in raw)          # raw Pendulum returns
    assert not np.allclose(rewards, raw)       # normalised stream differs


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("all ppo tests passed")
