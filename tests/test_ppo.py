"""Smoke tests for PPOAgent + ppo_training_loop and the config_ppo.yaml flow.
Run as `python tests/test_ppo.py` or via pytest from repo root."""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import gymnasium as gym

from agents.ppo_agent import PPOAgent
from ppo_training import ppo_training_loop, _collect_rollout
from experiment import load_config, build_env, build_ppo_agent, train_ppo

REPO = pathlib.Path(__file__).resolve().parents[1]


def _seed(s=0):
    torch.manual_seed(s)
    np.random.seed(s)


def _agent(**overrides):
    kwargs = dict(env=gym.make("CartPole-v1"), device="cpu",
                  rollout_steps=96, minibatch_size=32, update_epochs=2)
    kwargs.update(overrides)
    return PPOAgent(**kwargs)


def test_space_asserts_fail_fast():
    try:
        PPOAgent(env=gym.make("Pendulum-v1"))
        raise RuntimeError("Box action space should have been rejected")
    except AssertionError as e:
        assert "Discrete action space" in str(e)


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
    rewards = train_ppo(cfg, agent, env)
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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("all ppo tests passed")
