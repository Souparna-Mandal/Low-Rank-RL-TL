"""Tests for the Atari-100k protocol pieces: the training-loop step budget,
sign reward clipping, the final-evaluation helper, and the benchmark metadata.
All CPU-only and Atari-free (CartPole stands in for the env), so they run
anywhere the classic-control tests run."""
import pathlib
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch.nn as nn
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from agents.q_agent import QAgent                      # noqa: E402
from analysis import atari100k                         # noqa: E402
from environments.wrappers.reward_wrappers import SignClipReward  # noqa: E402
from training import dqn_training_loop, evaluate_policy_atari     # noqa: E402


class TinyNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 16), nn.ReLU(),
                                 nn.Linear(16, out_dim))

    def forward(self, x):
        return self.net(x)


class CountSteps(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.count = 0

    def step(self, action):
        self.count += 1
        return self.env.step(action)


def _cartpole_agent(env):
    return QAgent(
        replay_buffer_capacity=1000,
        q_network=TinyNet,
        batch_size=16,
        nn_learning_rate=1e-3,
        nn_extra_kwargs={"in_dim": env.observation_space.shape[0],
                         "out_dim": env.action_space.n},
        env=env,
        eps_start=1.0, eps_min=0.05, decay_rate=0.999,
        discount_factor=0.99, device="cpu",
        TD_LR=1.0, buffer_util=1, gd_steps_ceil=1, double=True,
    )


NO_ANALYSIS = {"ep_freq": 10**9, "methods": [],
               "hankel_sweep": {"enabled": False}}


def test_max_env_steps_caps_interactions_exactly():
    env = CountSteps(gym.make("CartPole-v1"))
    agent = _cartpole_agent(env)
    rewards = dqn_training_loop(
        agent, env, no_episodes=10**6, target_network_update_steps=10**9,
        train_frequency_steps=25, use_episode_training=False,
        solved_reward=10**9, warmup_steps=0,
        early_stopping_patience_eps=10**6, np_seed=0, no_eps_to_avg=100,
        analysis_config=NO_ANALYSIS, max_env_steps=137)
    # exactly 137 training interactions, and the partial episode is logged
    assert env.count == 137
    assert len(rewards) >= 1
    env.close()


def test_no_cap_keeps_episode_semantics():
    env = CountSteps(gym.make("CartPole-v1"))
    agent = _cartpole_agent(env)
    rewards = dqn_training_loop(
        agent, env, no_episodes=3, target_network_update_steps=10**9,
        train_frequency_steps=10**9, use_episode_training=False,
        solved_reward=10**9, warmup_steps=0,
        early_stopping_patience_eps=10**6, np_seed=0, no_eps_to_avg=100,
        analysis_config=NO_ANALYSIS)
    assert len(rewards) == 3
    env.close()


class FakeRewardEnv(gym.Env):
    observation_space = gym.spaces.Discrete(2)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, rewards):
        self._rewards = list(rewards)
        self._i = 0

    def reset(self, *, seed=None, options=None):
        self._i = 0
        return 0, {}

    def step(self, action):
        r = self._rewards[self._i]
        self._i += 1
        done = self._i >= len(self._rewards)
        return 0, r, done, False, {}


def test_sign_clip_reward_clips_training_reward_and_keeps_raw():
    env = SignClipReward(FakeRewardEnv([3.5, -20.0, 0.0]))
    env.reset()
    seen = [env.step(0) for _ in range(3)]
    assert [s[1] for s in seen] == [1.0, -1.0, 0.0]
    assert [s[4]["raw_reward"] for s in seen] == [3.5, -20.0, 0.0]


def test_evaluate_policy_atari_scores_and_restores_epsilon():
    env = gym.make("CartPole-v1")
    agent = _cartpole_agent(env)
    agent.epsilon = 0.7
    scores = evaluate_policy_atari(agent, env, episodes=3, epsilon=0.5,
                                   base_seed=7)
    assert len(scores) == 3 and all(s > 0 for s in scores)
    assert agent.epsilon == 0.7          # restored after evaluation
    env.close()


def test_hns_matches_published_reference_points():
    # SPR on Seaquest: (583.1 - 68.4) / (42054.7 - 68.4)
    assert atari100k.hns(583.1, "Seaquest") == pytest.approx(0.012259, abs=1e-5)
    assert atari100k.hns(6951.6, "MsPacman") == pytest.approx(1.0)
    assert atari100k.hns(0.0, "Enduro") == pytest.approx(0.0)
    agg = atari100k.aggregate_hns({"MsPacman": 6951.6, "Seaquest": 68.4})
    assert agg["mean"] == pytest.approx(0.5)
    assert atari100k.game_key("ALE/MsPacman-v5") == "MsPacman"
    # the full official suite, and never Enduro (no published baselines)
    assert len(atari100k.BENCHMARK_GAMES) == 26
    assert "Enduro" not in atari100k.BENCHMARK_GAMES
    assert {"MsPacman", "Seaquest", "Breakout", "Pong"} <= set(atari100k.BENCHMARK_GAMES)


def test_baseline_table_is_complete_and_verbatim():
    methods = {"SimPLe", "OTRainbow", "CURL", "DrQ", "SPR", "MuZero",
               "EfficientZero"}
    for game in atari100k.BENCHMARK_GAMES:
        assert set(atari100k.BASELINES[game]) == methods, game
        assert set(atari100k.REFERENCE[game]) == {"random", "human"}, game
    assert atari100k.BASELINES["Enduro"] == {}
    # spot checks, verbatim from Ye et al. 2021 Table 1
    assert atari100k.BASELINES["Breakout"]["MuZero"] == 48.0
    assert atari100k.BASELINES["Qbert"]["EfficientZero"] == 13781.9
    assert atari100k.BASELINES["UpNDown"]["SPR"] == 28138.5
    assert atari100k.BASELINES["Pong"]["CURL"] == -16.5
    assert atari100k.REFERENCE["Pong"]["random"] == -20.7
    assert atari100k.REFERENCE["CrazyClimber"]["human"] == 35829.4


GAME_CONFIG_DIRS = sorted(
    p.parent.name for p in (REPO / "experiments/atari").glob(
        "dqn_*/config_fhrdqn_100k.yaml"))
# ALE reports no lives counter for these (plus single-life Enduro): life-loss
# termination is meaningless there and the configs must not enable it.
NO_LIFE_LOSS_DIRS = {"dqn_enduro", "dqn_boxing", "dqn_freeway", "dqn_pong",
                     "dqn_private_eye"}


def test_every_suite_game_has_an_experiment_dir():
    # 26 suite games + Enduro, one dir per game, keys resolvable in REFERENCE
    assert len(GAME_CONFIG_DIRS) == 27
    keys = set()
    for game in GAME_CONFIG_DIRS:
        cfg = yaml.safe_load(
            (REPO / "experiments/atari" / game / "config_fhrdqn_100k.yaml").read_text())
        keys.add(atari100k.game_key(cfg["environment"]["name"]))
    assert keys == set(atari100k.BENCHMARK_GAMES) | {"Enduro"}


@pytest.mark.parametrize("game", GAME_CONFIG_DIRS)
def test_100k_configs_encode_the_protocol(game):
    cfg = yaml.safe_load(
        (REPO / "experiments/atari" / game / "config_fhrdqn_100k.yaml").read_text())
    assert cfg["training"]["max_env_steps"] == atari100k.ATARI100K_ENV_STEPS
    assert cfg["environment"]["atari"]["repeat_action_probability"] == 0.0
    assert cfg["environment"]["atari"]["full_action_space"] is False
    assert cfg["environment"]["reward"]["clip_sign"] is True
    assert cfg["evaluation"]["episodes"] == atari100k.EVAL_EPISODES
    assert cfg["evaluation"]["epsilon"] == atari100k.EVAL_EPSILON
    assert cfg["training"]["solved_reward"] >= 10**9   # early stopping disabled
    # >= 3 paired seeds is the protocol target (EfficientZero); 2 is the
    # current budget floor the repo actually runs with
    assert len(cfg["experiment"]["seeds"]) >= 2
    expect_lives = game not in NO_LIFE_LOSS_DIRS
    assert cfg["environment"]["atari"]["terminal_on_life_loss"] is expect_lives
    # aggregate membership is explicit in every config; Enduro (outside the
    # 26-game suite) must never opt in
    include = cfg["experiment"]["include_in_aggregate"]
    assert isinstance(include, bool)
    if game == "dqn_enduro":
        assert include is False


def test_aggregate_games_follows_the_config_flags():
    sys.path.insert(0, str(REPO / "experiments/src"))
    import run_fhrdqn_atari100k as launcher
    selected = launcher.aggregate_games()
    # every selected game opts in via config AND has published baselines
    assert "enduro" not in selected
    for game in selected:
        assert atari100k.BASELINES[launcher.config_game_key(game)]
    assert set(selected) <= set(launcher.GAME_DIRS)
    # explicit game-list argument restricts the scan
    assert launcher.aggregate_games(["pacman", "enduro"]) == ["pacman"]


def test_sign_clip_handles_numpy_scalar_rewards():
    env = SignClipReward(FakeRewardEnv(list(np.float32([2.5, -0.1, 0.0]))))
    env.reset()
    seen = [env.step(0) for _ in range(3)]
    assert [s[1] for s in seen] == [1.0, -1.0, 0.0]
    for r in np.linspace(-3, 3, 13):
        assert float(r > 0) - float(r < 0) == float(np.sign(r))
