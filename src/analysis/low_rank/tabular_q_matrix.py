from agents.q_agent import QAgent
from itertools import product, batched
import gymnasium as gym 
import numpy as np
import torch
import math

def _discrete_state_gen(env, state_discretisation, batch_size):
    """Returns a generator through the discretised state space for 
    batched forward passed

    Args:
        env (gym.Env): Gymnassium Envrionment to gather state space size & other metrics
        
        state_discretisation (list/tuple): list of tuple of number of discrete states per dimension, 
        e.g. [4,10,4] to discretise dim1 into 4 states, 
        dim2 into 10 states and dim3 into 4 states
        
        batch_size (int): how many indices to return at once for the table
    """
    obs_space = env.observation_space
    state_space = [[] for _ in range(obs_space.shape[0])] # first index selects the dim
    for i ,(lo,high) in enumerate(zip(obs_space.low, obs_space.high)):
        step_size = (high - lo) / state_discretisation[i]
        val  = lo + step_size / 2 
        while val < high:
            state_space[i].append(val)
            val = val + step_size
    # We need to return all possible cartesian product in batches
    return batched(product(*state_space), n=batch_size)

def q_matrix_dqn (agent: QAgent, state_discretisation: list , env: gym.Env, batch_size: int = 64):
    """
    """
    # initialising an empty Q matrix
    q_matrix = np.zeros((math.prod(state_discretisation),env.action_space.n))
    # getting a generator to give batches of states for forward pass
    state_space_gen = _discrete_state_gen(env,state_discretisation, batch_size)
    curr = 0
    for state_batch in state_space_gen:
        state_batch_t = torch.tensor(state_batch, dtype=torch.float32, device=agent.device)
        # Q values for batch_size of states
        Q_state = agent.policy_net(state_batch_t)
        next_curr = curr + Q_state.shape[0]
        q_matrix[curr: next_curr,:] = Q_state.detach().cpu().numpy() #SLOW unless mps shared mem
        curr =  next_curr
    return q_matrix