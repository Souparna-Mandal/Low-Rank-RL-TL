"""Atari protocol wrappers.

EpisodicLifeWrapper reproduces the Dopamine/DeepMind-baselines episodic-life
protocol that the published Atari-100k agents (DER/DrQ/SPR/BBF via Dopamine's
``terminal_on_life_loss`` + runner, EfficientZero via ``EpisodicLifeEnv``)
train under: losing a life ends the AGENT's episode — the value bootstrap is
cut and the replay buffer starts a new episode — but the GAME continues from
the current state on the next life; only real game-over (or truncation)
restarts the ALE. Gymnasium's ``AtariPreprocessing(terminal_on_life_loss=
True)`` is NOT that protocol: its reset fully restarts the game, so an agent
trains exclusively on first-life opening play.

Sits between AtariPreprocessing (built with ``terminal_on_life_loss=False``)
and the frame-stack wrapper.
"""
import gymnasium as gym


class EpisodicLifeWrapper(gym.Wrapper):
    """Life loss => terminated=True for the caller; the underlying game keeps
    running. ``reset()`` after a life-loss termination advances one no-op step
    instead of resetting; after real game-over/truncation — or whenever a seed
    is passed (seeded resets must be reproducible full restarts) — it truly
    resets. The reward of the continuation no-op step is discarded, matching
    the reference implementations."""

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        # 0 < lives: at game over the counter hits 0, which was_real_done
        # already covers — don't double-report it as a life-loss episode end.
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        if self.was_real_done or seed is not None:
            obs, info = self.env.reset(seed=seed, options=options)
        else:
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:  # the no-op itself ended the game
                obs, info = self.env.reset(options=options)
        self.was_real_done = False
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info
