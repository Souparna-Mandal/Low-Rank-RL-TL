"""Continuous-action-only Hankel-rank check on policy rollouts.

Registered as the analysis method `hankel_rollout_continuous` (usable from an
`analysis.methods` config entry). It rolls the agent's DETERMINISTIC policy
`n_rollouts` times and builds one stacked Hankel matrix per signal:

  * the critic trace  q_t = Q(s_t, pi(s_t))  (min over twin critics), and
  * each action dimension's trajectory  a_t[j] = pi(s_t)[j]  — the "does the
    policy itself have a low Hankel rank" check.

Rollouts are trimmed to the shortest length and their Hankel matrices are
stacked vertically: if every rollout is a response of the SAME order-r linear
recurrence (shared poles, rollout-specific amplitudes), every row of every
per-rollout Hankel lies in the span of the same r mode vectors, so the
stacked matrix still has rank <= r — stacking strengthens the check instead
of inflating the rank.

This check only applies to Box action spaces (a discrete action index is not
a continuous signal; the discrete stack has the hankel_sweep for its value
traces), so any other space raises.

Agent surface (SB3SACAdapter provides it): pi(state) -> env-scale np action,
device, and optionally q_value(obs_t, act_t) -> (B,) tensor. Without q_value
only the action Hankels are returned — the config's `outputs` list must
match: [Hankel Q(s,pi(s)), Hankel pi(s)[0], ..., Hankel pi(s)[d-1]] with the
Q entry dropped when include_q is false or unsupported.
"""
import gymnasium as gym
import numpy as np
import torch

from analysis.low_rank.hankel_policy import _hankel_from_sequence


def hankel_rollout_continuous(agent, env: gym.Env, n_rollouts: int = 3,
                              base_seed: int = 52, include_q: bool = True,
                              max_steps: int = 10000):
    """Stacked rollout Hankel matrices for a continuous-action policy.

    Returns a tuple of matrices: (Hankel of Q(s_t, pi(s_t)) if include_q and
    the agent exposes q_value) followed by one Hankel per action dimension.
    """
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError(
            "hankel_rollout_continuous is a continuous-action-only check "
            f"(got {type(env.action_space).__name__}); the discrete stack "
            "uses analysis.hankel_sweep instead")
    act_dim = int(np.prod(env.action_space.shape))
    use_q = include_q and hasattr(agent, "q_value")

    q_seqs: list[np.ndarray] = []
    act_seqs: list[np.ndarray] = []
    for k in range(n_rollouts):
        state, _ = env.reset(seed=base_seed + k)
        done = False
        steps = 0
        qs: list[float] = []
        acts: list[np.ndarray] = []
        while not done and steps < max_steps:
            action = agent.pi(state)
            flat = np.asarray(action, dtype=float).reshape(-1)
            acts.append(flat)
            if use_q:
                state_t = torch.as_tensor(
                    np.asarray(state), dtype=torch.float32,
                    device=agent.device).unsqueeze(0)
                act_t = torch.as_tensor(
                    flat, dtype=torch.float32,
                    device=agent.device).unsqueeze(0)
                qs.append(float(agent.q_value(state_t, act_t)[0].item()))
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        act_seqs.append(np.asarray(acts, dtype=float))
        if use_q:
            q_seqs.append(np.asarray(qs, dtype=float))

    h_min = min(len(a) for a in act_seqs)
    if h_min < 2:
        raise ValueError(f"rollouts too short for a Hankel matrix (H={h_min})")

    def stacked_hankel(seqs):
        return np.vstack([_hankel_from_sequence(np.asarray(s[:h_min],
                                                           dtype=float))
                          for s in seqs])

    matrices = []
    if use_q:
        matrices.append(stacked_hankel(q_seqs))
    for j in range(act_dim):
        matrices.append(stacked_hankel([a[:, j] for a in act_seqs]))
    return tuple(matrices) if len(matrices) > 1 else matrices[0]
