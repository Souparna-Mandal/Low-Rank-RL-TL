import numpy as np
import gymnasium as gym 
from gymnasium import ActionWrapper, ObservationWrapper, Wrapper
from gymnasium.spaces import Dict, Box, MultiDiscrete

class DiscreteActionWrapper(ActionWrapper):
    def __init__(self, env, n_actions: list):
        super().__init__(env)


class DiscreteStateWrapper(ObservationWrapper):
    def __init__(self, env: gym.Env, n_states: list):
        super().__init__(env)
        assert isinstance(env.observation_space, Box), ("DiscreteStateWrapper requires a Box observation space")  # Error check
        self.n_states = np.array(n_states) #list of discrete values in every dimension 
        self.update_obs_space()
        self.new_space = self.new_space_bins # list of ndarrays
        
    def update_obs_space(self,):
        self._no_dims = self.env.observation_space.shape
        assert self._no_dims[0] == len(self.n_states), ("Discretisation config dimensions are not matching")  # Error check
        
        self._env_d_type = self.env.observation_space.dtype
        self.low = self.env.observation_space.low   # np.ndarray of the Lower bounds 
        self.high = self.env.observation_space.high # np.ndarray of the Upper bounds 
        
        self.observation_space = Dict({
            "index": MultiDiscrete(self.n_states),
            "value": Box(low=self.low, high=self.high,
                        shape=self._no_dims, dtype=self._env_d_type)
        })
        
    @property
    def new_space_bins(self,):
        new_space = [np.linspace(low,high,count, endpoint=False) + (high-low)/(2*count) 
                        for low,high,count in zip(self.low, self.high, self.n_states)]
        return new_space

    def observation(self, obs):
        new_space_idxs = ((obs -self.low) / (self.high - self.low) * self.n_states).astype(int)
        new_space_idxs = np.clip(new_space_idxs, 0, self.n_states - 1)
        
        new_obs = np.array([self.new_space[d][new_space_idxs[d]] for d in range(self._no_dims[0])],
                            dtype=self._env_d_type)
        return {"index": new_space_idxs, "value": new_obs}
