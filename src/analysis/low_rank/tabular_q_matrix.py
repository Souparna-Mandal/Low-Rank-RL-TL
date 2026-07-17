from agents.q_agent import QAgent
from itertools import product, batched
import gymnasium as gym
import numpy as np
import torch
import math
from scipy.cluster.vq import kmeans2

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

def q_matrix_tabular(agent, env: gym.Env = None):
    """The tabular agent's own (n_states, n_actions) Q table — no probing needed.
    `env` is unused; accepted because the analysis dispatch injects agent/env."""
    return np.array(agent.Q, dtype=float)

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

def q_matrix_rollout(agent: QAgent, env: gym.Env, no_episodes: int = 3,
                     max_states: int = 4000, feature_clusters: int | None = None,
                     seed: int = 52, batch_size: int = 256):
    """Pixel-friendly analogue of `q_matrix_dqn`. A per-dimension grid over a
    (4, 84, 84) observation space is meaningless (and astronomically large), so
    instead the rows are states actually visited by the current (eps-greedy)
    policy — the trajectory-sampled Q-matrix construction used by Yang et al.
    (ICLR 2020, "Harnessing Structures for Value-Based Planning and RL") to show
    Atari Q matrices are approximately low rank.

    If `feature_clusters` is set and the policy network exposes `phi(x)` (the
    conv-trunk embedding), visited states are additionally k-means-binned in that
    latent space and Q rows are averaged per bin — the latent-space analogue of
    the per-dimension bins used for classic-control envs. Empty bins are dropped.

    Args:
        no_episodes (int): policy rollouts to harvest states from.
        max_states (int): cap on total states collected across rollouts.
        feature_clusters (int | None): number of latent k-means bins; None skips
            the aggregated matrix.
        batch_size (int): states per forward pass.

    Returns:
        (N, n_actions) visited-state Q-matrix, and additionally the
        (K, n_actions) latent-binned Q-matrix when clustering is active.
    """
    states = []
    for ep in range(no_episodes):
        state, _ = env.reset(seed=seed + ep)
        terminated = truncated = False
        while not (terminated or truncated) and len(states) < max_states:
            states.append(np.asarray(state))
            action = agent.pi(state)
            state, _, terminated, truncated, _ = env.step(action)
        if len(states) >= max_states:
            break

    feature_fn = getattr(agent.policy_net, "phi", None)
    q_rows, feature_rows = [], []
    with torch.no_grad():
        for i in range(0, len(states), batch_size):
            state_batch_t = torch.as_tensor(np.stack(states[i: i + batch_size]),
                                            dtype=torch.float32, device=agent.device)
            q_rows.append(agent.policy_net(state_batch_t).cpu().numpy())
            if feature_clusters is not None and feature_fn is not None:
                feature_rows.append(feature_fn(state_batch_t).cpu().numpy())
    q_matrix = np.concatenate(q_rows, axis=0)

    if feature_clusters is None or feature_fn is None:
        return q_matrix

    features = np.concatenate(feature_rows, axis=0)
    _, labels = kmeans2(features.astype(np.float64), feature_clusters,
                        minit="++", seed=seed)
    q_binned = np.stack([q_matrix[labels == k].mean(axis=0)
                         for k in range(feature_clusters) if (labels == k).any()])
    return q_matrix, q_binned