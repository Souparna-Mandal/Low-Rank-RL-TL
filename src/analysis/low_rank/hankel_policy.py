from agents.q_agent import QAgent
import gymnasium as gym 
import numpy as np
import torch
import scipy

def hankel_rollout(agent:QAgent, env: gym.Env, seed: int = 52):
    """Given an agent with a policy pi_theeta, we rollout the policy 
    in the given environment and look at at the rank of the Hankel 
    matrix of the value or Q function of the trajector.

    Args:
        agent (QAgent): Deep Q Learning agent.
    """
    state, _ = env.reset(seed=seed)
    terminated = truncated = False
    # trajectory_tau = [] # list of tuples (state,action)
    tau_q = []
    while not (terminated or truncated):
        # getting the action from current policy
        action = agent.pi(state)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        with torch.no_grad():
            q_val = agent.policy_net(state_t)[0, action].item() 
            
        # add to the trajectories
        # trajectory_tau.append((state,action, q_val))
        tau_q.append(q_val)
        state, _, terminated, truncated, _ = env.step(action) # get next state
        
    # Building the hankel matrix 
    tau_len = len(tau_q)
    mid_index = int(tau_len/2) # We can exploit this further and see which hankel matrix is the 
    # most low rank and appropriately speed up value iteration.
    hankel = scipy.linalg.hankel(tau_q[:mid_index+1],  tau_q[mid_index:])
    return hankel