import gymnasium as gym
from gymnasium.wrappers import (RescaleAction, RescaleObservation, ClipAction,
                                TransformObservation, AtariPreprocessing,
                                FrameStackObservation, NormalizeObservation,
                                NormalizeReward, ClipReward,
                                RecordEpisodeStatistics)
import yaml
import pathlib
import numpy as np

from .wrappers.action_wrappers import DiscretiseActionWrapper
from .wrappers.discrete_wrappers import DiscreteActionWrapper, DiscreteStateWrapper
from .wrappers.observation_wrappers import OneHotObservationWrapper

environments = {}

## Initialise the Environments ## 

def make_environment(env_name: str, render_mode = None, 
                    discrete_config: dict = {'no_action_bins': 0,
                                            'no_state_bins': 0 },
                    **env_kwargs) -> gym.Env:
    """Create a new environment and return the gym environment type, Pass in the config for 
    state/action space discretisation via the argument and any extra args like normalising actions by the agents
    or normalising state observations is done via additional keyword arguments. 

    Args:
        env_name (str): name of an existing gym environment like acrobot-v2

    Returns:
        gym.env: the created environment object.
    """
    # Atari envs processing
    atari_cfg = env_kwargs.get('atari', None)
    if atari_cfg:
        env = gym.make(env_name, render_mode=render_mode, frameskip=1)
        env = AtariPreprocessing(
            env,
            noop_max=atari_cfg['noop_max'],
            frame_skip=atari_cfg['frame_skip'],
            screen_size=atari_cfg['screen_size'],
            terminal_on_life_loss=atari_cfg['terminal_on_life_loss'],
            grayscale_obs=atari_cfg['grayscale_obs'],
            grayscale_newaxis=atari_cfg['grayscale_newaxis'],
            scale_obs=atari_cfg['scale_obs'],
        )
        if atari_cfg['frame_stack'] > 1:
            env = FrameStackObservation(env, stack_size=atari_cfg['frame_stack'])
        return env

    env = gym.make(env_name, render_mode=render_mode)

    # Episode cap for envs registered without one (CliffWalking wanders
    # unboundedly under a random policy). config: time_limit: <steps>.
    if env_kwargs.get('time_limit'):
        env = gym.wrappers.TimeLimit(env, max_episode_steps=env_kwargs['time_limit'])

    # Discrete(n) observations -> one-hot vectors for MLP agents.
    # config: one_hot_obs: true.
    if env_kwargs.get('one_hot_obs'):
        env = OneHotObservationWrapper(env)

    # 1-D Box action space -> Discrete(n) evenly spaced actions (Pendulum).
    # config: discrete_action_bins: <n>. The result is a Discrete space, so
    # the Box-only action transforms downstream cannot follow it.
    if env_kwargs.get('discrete_action_bins'):
        if env_kwargs.get('clip', {}).get('action') or \
                env_kwargs.get('normalise', {}).get('action'):
            raise ValueError(
                "discrete_action_bins produces a Discrete action space and "
                "cannot be combined with clip.action or normalise.action")
        env = DiscretiseActionWrapper(env, env_kwargs['discrete_action_bins'])

    # Discretise the State Action Space
    if discrete_config is not None:
        if discrete_config['no_action_bins'] > 0:
            env = DiscreteActionWrapper(env, n_actions=discrete_config['no_action_bins'])
        if discrete_config['no_state_bins'] > 0:
            env = DiscreteStateWrapper(env, n_states=discrete_config['no_state_bins'])  

    # Normalise the Actions. MUST come before ClipAction: gymnasium's
    # ClipAction re-advertises the action space as Box(-inf, inf), and
    # RescaleAction cannot build a rescale from an infinite range. Applied
    # first, the agent sees [min, max] -> clipped there -> rescaled to the
    # env's real bounds, which is the standard continuous-control setup.
    if len(env_kwargs['normalise']['action']) > 0:
        env = RescaleAction(env,
                            min_action = env_kwargs['normalise']['action']['min'],
                            max_action = env_kwargs['normalise']['action']['max'])

    # Clip Actions
    if env_kwargs['clip']['action'] : # Boolean Flag
        env = ClipAction(env)

    # Clip State Space
    if len(env_kwargs['clip']['state']) > 0:
        low  = np.array(env_kwargs['clip']['state']['min'], dtype=np.float32)
        high = np.array(env_kwargs['clip']['state']['max'], dtype=np.float32)
        orig = env.observation_space
        low  = np.where(np.isnan(low),  orig.low,  low)
        high = np.where(np.isnan(high), orig.high, high)
        new_space = gym.spaces.Box(low=low, high=high, dtype=orig.dtype)
        env =  TransformObservation( env,
                    lambda obs, low=low, high=high: np.clip(obs, low, high),
                    observation_space=new_space)
        
    # Normalise the State Observations (fixed linear rescale to a known range;
    # see normalise.running.obs below for the running-statistics version)
    if len(env_kwargs['normalise']['state']) > 0:
        env = RescaleObservation(env,
                            np.array(env_kwargs['normalise']['state']['min'], dtype=np.float32),
                            np.array(env_kwargs['normalise']['state']['max'], dtype=np.float32))

    # Running-statistics normalisation, the CleanRL continuous-control stack.
    # config: normalise: {running: {obs: true, reward: true, gamma: 0.99,
    #                               clip_obs: 10.0, clip_reward: 10.0}}
    # Order matters. RecordEpisodeStatistics goes on FIRST so it sees the raw
    # reward: info["episode"]["r"] then carries the true return while step()
    # hands the agent the normalised one, which is what lets the training loop
    # report both.
    running = env_kwargs['normalise'].get('running') or {}
    if running.get('obs') or running.get('reward'):
        env = RecordEpisodeStatistics(env)
    if running.get('obs'):
        env = NormalizeObservation(env)
        clip_obs = running.get('clip_obs', 10.0)
        if clip_obs:
            env = TransformObservation(
                env, lambda obs, c=clip_obs: np.clip(obs, -c, c),
                observation_space=gym.spaces.Box(
                    -clip_obs, clip_obs, env.observation_space.shape,
                    dtype=np.float32))
    if running.get('reward'):
        env = NormalizeReward(env, gamma=running.get('gamma', 0.99))
        clip_reward = running.get('clip_reward', 10.0)
        if clip_reward:
            env = ClipReward(env, -clip_reward, clip_reward)
    return env