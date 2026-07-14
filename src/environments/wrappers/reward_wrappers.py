import gymnasium as gym


class ScaleReward(gym.Wrapper):
    """Multiply every reward by a constant scale factor.

    Reward scaling is environment preprocessing (Mnih et al. 2015 clip Atari
    rewards at this same layer): unclipped Atari scores (e.g. Seaquest's
    20..1000+) produce heavy-tailed TD errors that skew PER priorities and
    saturate Huber losses tuned for unit-scale rewards. A scale like 0.01
    restores the reward range those hyperparameters assume, for any agent.

    The raw (unscaled) reward is stashed in ``info["raw_reward"]`` so the
    training loop can keep logging true game scores — logged reward curves and
    ``solved_reward`` thresholds stay comparable across runs with and without
    scaling; only what the agent trains on changes.
    """

    def __init__(self, env: gym.Env, scale: float):
        super().__init__(env)
        self.scale = float(scale)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["raw_reward"] = reward
        return obs, reward * self.scale, terminated, truncated, info
