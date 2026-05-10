"""Hankel-matrix rank analysis of value/Q/policy trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import gymnasium as gym

from low_rank_rl.agents.base import BaseAgent


@dataclass
class HankelMetrics:
    sequence_type:   str
    sequence_length: int
    hankel_shape:    tuple[int, int]
    singular_values: np.ndarray
    numerical_rank:  int

    def summary(self) -> str:
        m, n = self.hankel_shape
        return (
            f"Hankel({self.sequence_type}) {m}x{n}  |  "
            f"seq_len={self.sequence_length}  |  "
            f"num_rank={self.numerical_rank}/{min(m, n)}"
        )


def build_hankel_matrix(sequence: np.ndarray, n_rows: int) -> np.ndarray:
    T      = len(sequence)
    n_cols = T - n_rows + 1
    if n_cols < 1:
        raise ValueError(f"Sequence length {T} too short for n_rows={n_rows}.")
    H = np.zeros((n_rows, n_cols), dtype=np.float64)
    for i in range(n_rows):
        H[i] = sequence[i: i + n_cols]
    return H


def _env_horizon(env: gym.Env, fallback: int = 1000) -> int:
    """Upper bound on episode length; prefer the env's built-in cap."""
    spec = getattr(env, "spec", None)
    if spec is not None and getattr(spec, "max_episode_steps", None):
        return int(spec.max_episode_steps)
    return fallback


def collect_trajectory(
    agent: BaseAgent,
    env: gym.Env,
    n_steps: int | None = None,
    use_greedy: bool = True,
) -> dict[str, np.ndarray]:
    """Roll out one episode under the (greedy) policy.

    When ``n_steps`` is ``None`` we run up to the env's
    ``spec.max_episode_steps`` and stop on natural termination/truncation —
    i.e. the full trajectory produced by the current optimal policy π.
    """
    max_steps = n_steps if n_steps is not None else _env_horizon(env)

    states, actions, values, q_taken, policy = [], [], [], [], []
    obs, _ = env.reset()

    for _ in range(max_steps):
        action = agent.act(obs, training=not use_greedy)
        q_row  = agent.q_matrix(obs[np.newaxis])[0]

        states.append(obs.copy())
        actions.append(action)
        values.append(float(q_row.max()))
        q_taken.append(float(q_row[action]))
        policy.append(float(q_row.argmax()))

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    return {
        "states":  np.array(states),
        "actions": np.array(actions, dtype=np.float64),
        "value":   np.array(values),
        "q_taken": np.array(q_taken),
        "policy":  np.array(policy, dtype=np.float64),
    }


def hankel_rank_metrics(
    agent: BaseAgent,
    env: gym.Env,
    sequence_type: str = "value",
    n_steps: int | None = None,
    n_rows: int | None = None,
    tol: float = 1e-5,
) -> HankelMetrics:
    """Numerical rank of the Hankel matrix of a greedy-policy trajectory.

    ``n_steps=None``  → roll out the full episode under the greedy policy.
    ``n_rows=None``   → square Hankel (``T // 2``), which maximises
                        ``min(n_rows, n_cols)`` and therefore the largest
                        rank we can possibly detect.
    """
    traj = collect_trajectory(agent, env, n_steps=n_steps, use_greedy=True)
    seq  = traj[sequence_type]
    T    = len(seq)
    if n_rows is None:
        n_rows = max(2, T // 2)

    H     = build_hankel_matrix(seq, n_rows)
    sigma = np.linalg.svd(H, compute_uv=False)

    threshold = tol * sigma[0] if sigma[0] > 0 else tol
    num_rank  = int(np.sum(sigma > threshold))

    return HankelMetrics(
        sequence_type=sequence_type,
        sequence_length=T,
        hankel_shape=H.shape,
        singular_values=sigma,
        numerical_rank=num_rank,
    )


def dmd_from_hankel(H: np.ndarray, rank: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Dynamic Mode Decomposition: fit A with H[:,1:] ≈ A · H[:,:-1]."""
    X  = H[:, :-1]
    Xp = H[:, 1:]

    U, sigma, Vt = np.linalg.svd(X, full_matrices=False)
    if rank is not None:
        U, sigma, Vt = U[:, :rank], sigma[:rank], Vt[:rank, :]

    A_tilde                 = U.T @ Xp @ Vt.T @ np.diag(1.0 / (sigma + 1e-12))
    eigenvalues, eigvectors = np.linalg.eig(A_tilde)
    modes                   = Xp @ Vt.T @ np.diag(1.0 / (sigma + 1e-12)) @ eigvectors
    return modes, eigenvalues



# Also look at the Log of Q function. 