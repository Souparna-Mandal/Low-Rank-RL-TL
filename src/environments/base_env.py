import gymnasium as gym 
from gymnasium.wrappers import RescaleAction, RescaleObservation, ClipAction, TransformObservation
import yaml 
import pathlib
import numpy as np

from .wrappers.discrete_wrappers import DiscreteActionWrapper, DiscreteStateWrapper

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
    env = gym.make(env_name, render_mode=render_mode)
    
    # Discretise the State Action Space
    if discrete_config is not None:
        if discrete_config['no_action_bins'] > 0:
            env = DiscreteActionWrapper(env, n_actions=discrete_config['no_action_bins'])
        if discrete_config['no_state_bins'] > 0:
            env = DiscreteStateWrapper(env, n_states=discrete_config['no_state_bins'])  

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
    return env