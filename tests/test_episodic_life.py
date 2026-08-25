"""EpisodicLifeWrapper: life loss ends the agent episode while the game
continues; only real game-over (or a seeded reset) restarts the ALE. Uses a
fake lives-bearing env so no ROM stepping is needed, plus one real-ALE
construction check through build_env."""
import pathlib
import sys

import gymnasium as gym
import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from environments.wrappers.atari_wrappers import EpisodicLifeWrapper  # noqa: E402


class _FakeALE:
    def __init__(self, env):
        self._env = env

    def lives(self):
        return self._env.lives_left


class FakeLivesEnv(gym.Env):
    """3 lives; a life is lost every `life_len` steps; game over after the
    third. reset() restarts everything (that is exactly what the wrapper must
    AVOID calling mid-game)."""
    observation_space = gym.spaces.Box(0, 255, shape=(4,), dtype=np.uint8)
    action_space = gym.spaces.Discrete(3)

    def __init__(self, life_len=4):
        self.life_len = life_len
        self.lives_left = 3
        self.t = 0
        self.full_resets = 0
        self.ale = _FakeALE(self)

    @property
    def unwrapped(self):
        return self

    def _obs(self):
        return np.full(4, self.t % 256, dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        self.lives_left = 3
        self.t = 0
        self.full_resets += 1
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        terminated = False
        if self.t % self.life_len == 0:
            self.lives_left -= 1
            terminated = self.lives_left == 0    # real game over on last life
        return self._obs(), 1.0, terminated, False, {}


def test_life_loss_terminates_agent_episode_but_game_continues():
    env = EpisodicLifeWrapper(FakeLivesEnv(life_len=4))
    env.reset(seed=0)
    assert env.unwrapped.full_resets == 1
    steps_to_terminal = 0
    terminated = False
    while not terminated:
        _, _, terminated, _, _ = env.step(0)
        steps_to_terminal += 1
    assert steps_to_terminal == 4                # first life lost
    assert env.unwrapped.lives_left == 2
    # unseeded reset after a life loss: NO underlying restart, game continues
    env.reset()
    assert env.unwrapped.full_resets == 1
    assert env.unwrapped.t == 5                  # the continuation no-op step
    assert env.unwrapped.lives_left == 2


def test_real_game_over_and_seeded_reset_restart():
    env = EpisodicLifeWrapper(FakeLivesEnv(life_len=2))
    env.reset(seed=0)
    for _ in range(2):                           # play through lives 1 and 2
        terminated = False
        while not terminated:
            _, _, terminated, _, _ = env.step(0)
        env.reset()
    # third life: terminal is REAL game over
    terminated = False
    while not terminated:
        _, _, terminated, _, _ = env.step(0)
    assert env.unwrapped.lives_left == 0 and env.was_real_done
    env.reset()
    assert env.unwrapped.full_resets == 2        # true restart
    assert env.unwrapped.lives_left == 3
    # a SEEDED reset mid-game must also truly restart (reproducibility)
    _, _, terminated, _, _ = env.step(0)
    while not terminated:
        _, _, terminated, _, _ = env.step(0)
    env.reset(seed=123)
    assert env.unwrapped.full_resets == 3


def test_build_env_wires_episodic_life_and_eval_override():
    pytest.importorskip("ale_py")
    from experiment import build_env
    base = {
        "environment": {
            "name": "ALE/Krull-v5", "render_mode": None,
            "discrete_config": None,
            "normalise": {"action": {}, "state": {}},
            "clip": {"action": False, "state": {}},
            "reward": {"clip_sign": True},
            "atari": {
                "noop_max": 30, "frame_skip": 4, "screen_size": 84,
                "terminal_on_life_loss": False, "episodic_life": True,
                "grayscale_obs": True, "grayscale_newaxis": False,
                "scale_obs": False, "frame_stack": 4,
                "repeat_action_probability": 0.0, "full_action_space": False,
            },
        }
    }
    env = build_env(base)
    layers = []
    e = env
    while hasattr(e, "env"):
        layers.append(type(e).__name__)
        e = e.env
    assert "EpisodicLifeWrapper" in layers
    env.close()
    # both flags on is a config error, not a silent game-restart protocol
    bad = {"environment": {**base["environment"],
                           "atari": {**base["environment"]["atari"],
                                     "terminal_on_life_loss": True}}}
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_env(bad)


# --------------------------------------------------------------------------
# force_full_reset: the escape hatch the training loop needs after an analysis
# rollout hijacks the training env (see training.reset_after_analysis).
# --------------------------------------------------------------------------

def test_force_full_reset_restarts_after_a_life_loss():
    """Without it, a reset() following a life-loss termination continues the
    same game — which would hand the next training episode the game state the
    ANALYSIS policy reached."""
    env = EpisodicLifeWrapper(FakeLivesEnv(life_len=4))
    env.reset()
    for _ in range(4):                      # burn one life -> terminated
        _, _, terminated, _, _ = env.step(0)
    assert terminated and env.unwrapped.lives_left == 2

    before = env.unwrapped.full_resets
    env.force_full_reset()
    env.reset()
    assert env.unwrapped.full_resets == before + 1, "should truly restart"
    assert env.unwrapped.lives_left == 3
    assert env.unwrapped.t == 0


def test_plain_reset_after_life_loss_still_continues_the_game():
    """The default protocol is unchanged — force_full_reset is opt-in."""
    env = EpisodicLifeWrapper(FakeLivesEnv(life_len=4))
    env.reset()
    for _ in range(4):
        env.step(0)
    before = env.unwrapped.full_resets
    env.reset()
    assert env.unwrapped.full_resets == before, "must NOT restart the game"
    assert env.unwrapped.lives_left == 2


def test_force_full_reset_survives_nested_wrappers_via_get_wrapper_attr():
    """The wrapper sits BELOW frame-stack/reward wrappers in build_env, and
    Gymnasium 1.x does not forward attributes through them — plain getattr
    silently misses it, which is why training.reset_after_analysis uses
    get_wrapper_attr."""
    inner = EpisodicLifeWrapper(FakeLivesEnv(life_len=4))
    env = gym.Wrapper(gym.Wrapper(inner))
    env.reset()
    for _ in range(4):
        env.step(0)
    assert getattr(env, "force_full_reset", None) is None, "getattr must not find it"

    before = inner.unwrapped.full_resets
    env.get_wrapper_attr("force_full_reset")()
    env.reset()
    assert inner.unwrapped.full_resets == before + 1


def test_reset_after_analysis_restarts_only_when_something_rolled_out():
    from training import reset_after_analysis

    env = gym.Wrapper(EpisodicLifeWrapper(FakeLivesEnv(life_len=4)))
    env.reset()
    for _ in range(4):
        env.step(0)
    before = env.unwrapped.full_resets
    reset_after_analysis(env, rolled_out=True)
    assert env.unwrapped.full_resets == before + 1


def test_reset_after_analysis_without_a_rollout_keeps_the_life_protocol():
    """A tick that never touched `env` leaves it where the ordinary
    end-of-episode reset put it — forcing a restart there would throw away the
    remaining lives and change the training protocol."""
    from training import reset_after_analysis

    env = gym.Wrapper(EpisodicLifeWrapper(FakeLivesEnv(life_len=4)))
    env.reset()
    for _ in range(4):
        env.step(0)
    before = env.unwrapped.full_resets
    reset_after_analysis(env, rolled_out=False)
    assert env.unwrapped.full_resets == before, "must NOT restart the game"
    assert env.unwrapped.lives_left == 2


def test_reset_after_analysis_is_a_noop_for_envs_without_the_wrapper():
    from training import reset_after_analysis

    plain = gym.make("CartPole-v1")
    plain.reset()
    obs, _ = reset_after_analysis(plain, rolled_out=True)
    assert obs is not None
    plain.close()


def test_run_analysis_tick_reports_whether_it_touched_the_env():
    """The flag reset_after_analysis keys off: no rollout blocks configured =>
    False, an enabled hankel_sweep => True."""
    from training import run_analysis_tick

    env = gym.Wrapper(EpisodicLifeWrapper(FakeLivesEnv(life_len=4)))
    env.reset()
    assert run_analysis_tick(agent=None, env=env, analysis_config={}) is False
    assert run_analysis_tick(agent=None, env=env,
                             analysis_config={"methods": [],
                                              "hankel_sweep": {"enabled": False}}) is False
