import gymnasium as gym
from gymnasium.wrappers import (RescaleAction, RescaleObservation, ClipAction,
                                TransformObservation, AtariPreprocessing,
                                FrameStackObservation)
import yaml
import pathlib
import numpy as np

from .wrappers.discrete_wrappers import DiscreteStateWrapper
from .wrappers.generative_wrappers import GenerativeStateWrapper
from .wrappers.observation_wrappers import UnderlyingStateWrapper
from .wrappers.reward_wrappers import ScaleReward, SignClipReward

environments = {}

## Initialise the Environments ## 

def make_environment(env_name: str, render_mode = None,
                    discrete_config: dict = None,
                    **env_kwargs) -> gym.Env:
    """Create a new environment and return the gym environment type.

    discrete_config: {'state_bins': [bins per obs dim]} wraps the env in a
    DiscreteStateWrapper (applied last, so it bins over the clipped/normalised
    bounds). `generative: true` adds GenerativeStateWrapper (teleport() for
    generative-model rollouts). Normalising/clipping of actions and state
    observations is done via additional keyword arguments.

    Args:
        env_name (str): name of an existing gym environment like acrobot-v2

    Returns:
        gym.env: the created environment object.
    """
    # Reward preprocessing (applies to both the Atari and classic branches).
    # config: reward: {scale: 0.01} — raw reward stays in info["raw_reward"].
    reward_cfg = env_kwargs.get('reward', None) or {}

    # Atari envs processing
    atari_cfg = env_kwargs.get('atari', None)
    if atari_cfg:
        # ALE/... ids only exist after ale_py registers them — notebooks do
        # this import themselves, but subprocess runners come through here.
        # register_envs is idempotent; without ale_py, gym.make raises its own
        # clear NamespaceNotFound.
        try:
            import ale_py
            gym.register_envs(ale_py)
        except ImportError:
            pass
        # Optional ALE-level protocol settings. The Atari-100k benchmark
        # (SimPLe/CURL/SPR/EfficientZero lineage) uses a deterministic ALE —
        # repeat_action_probability 0 (no sticky actions) and the per-game
        # minimal action set — while the v5 defaults are sticky 0.25. Omitted
        # keys keep the v5 defaults, so existing configs are unaffected.
        ale_kwargs = {}
        if atari_cfg.get('repeat_action_probability') is not None:
            ale_kwargs['repeat_action_probability'] = atari_cfg['repeat_action_probability']
        if atari_cfg.get('full_action_space') is not None:
            ale_kwargs['full_action_space'] = atari_cfg['full_action_space']
        env = gym.make(env_name, render_mode=render_mode, frameskip=1, **ale_kwargs)
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
        # One reward wrapper at a time (both stash raw_reward in info; stacking
        # would clobber it): sign clipping (DQN-family Atari convention) wins.
        if reward_cfg.get('clip_sign'):
            env = SignClipReward(env)
        elif reward_cfg.get('scale') is not None:
            env = ScaleReward(env, reward_cfg['scale'])
        return env

    env = gym.make(env_name, render_mode=render_mode)

    # Generative-model access (teleport to arbitrary states). Innermost so
    # teleport's reset clears the TimeLimit/termination bookkeeping.
    if env_kwargs.get('generative'):
        env = GenerativeStateWrapper(env)

    # Native-state observation for envs whose obs is a lossy encoding of the
    # true state (e.g. Acrobot's cos/sin). Applied before clip/discretise so
    # those bin the underlying state that teleport writes.
    state_obs_cfg = env_kwargs.get('state_observation')
    if state_obs_cfg:
        env = UnderlyingStateWrapper(env, low=state_obs_cfg['low'],
                                     high=state_obs_cfg['high'])

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
        
    # Normalise the Actions  
    if len(env_kwargs['normalise']['action']) > 0:
        env = RescaleAction(env, 
                            min_action = env_kwargs['normalise']['action']['min'],
                            max_action = env_kwargs['normalise']['action']['max'])
        
    # Normalise the State Observations
    if len(env_kwargs['normalise']['state']) > 0:
        env = RescaleObservation(env,
                            np.array(env_kwargs['normalise']['state']['min'], dtype=np.float32),
                            np.array(env_kwargs['normalise']['state']['max'], dtype=np.float32))

    # Scale Rewards
    if reward_cfg.get('scale') is not None:
        env = ScaleReward(env, reward_cfg['scale'])

    # State discretisation service — applied last so its bin bounds are the
    # clipped/normalised (finite) ones. Observations pass through unchanged.
    if discrete_config is not None and discrete_config.get('state_bins'):
        env = DiscreteStateWrapper(env, state_bins=discrete_config['state_bins'])
    return env