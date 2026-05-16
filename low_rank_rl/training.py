"""Shared training loops used by experiments/run.py and experiments/notebook.ipynb.

Three drivers cover all agents in this package:

- ``train_step_based``     — DQN / Q-learning / SARSA (TD updates per step).
- ``train_episode_based``  — Monte Carlo (``agent.end_episode()`` at episode end).
- ``train_ppo``            — PPO (GAE + clipped surrogate; bootstraps ``last_value``
                              on truncation).

Every driver:

- returns ``TrainingLog`` with ``durations``, ``rewards``, ``rank_history``;
- displays a ``tqdm`` progress bar with a rolling-mean reward in the postfix;
- invokes an optional ``on_episode(ep, duration, reward)`` callback after each
  episode so callers (e.g. a notebook) can update a live plot.

The DQN loop follows the PyTorch official RL tutorial
(https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html);
the PPO loop follows OpenAI Spinning Up's pseudocode
(https://spinningup.openai.com/en/latest/algorithms/ppo.html).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Optional

from tqdm.auto import tqdm

from low_rank_rl.analysis.rank   import compute_rank_metrics, sample_states
from low_rank_rl.analysis.hankel import hankel_rank_metrics


EpisodeCallback = Callable[[int, int, float], None]


@dataclass
class TrainingLog:
    durations:    list[int]    = field(default_factory=list)
    rewards:      list[float]  = field(default_factory=list)
    rank_history: list[dict]   = field(default_factory=list)
    eval_history: list[dict]   = field(default_factory=list)
    best_eval:    float        = float("-inf")
    best_episode: int          = 0


def _evaluate(agent, env, n_episodes: int) -> float:
    """Mean greedy-policy return over n_episodes (no exploration, no learning)."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            action = agent.act(obs, training=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += float(reward)
        rewards.append(total)
    return sum(rewards) / max(len(rewards), 1)


def _checkpoint(agent, env, cfg: dict) -> tuple[dict, object]:
    """Q-matrix rank + optional Hankel spectra for the current greedy policy.

    Hankel capture is controlled by ``cfg['analysis']['hankel_sequence_types']``
    (list of {"value", "q_taken", "policy", "actions"}). Empty/missing ->
    skipped, matching the previous snapshot behaviour.
    """
    n_rank_samples = cfg["training"]["n_rank_samples"]
    analysis_cfg   = cfg.get("analysis", {}) or {}
    hankel_seqs    = analysis_cfg.get("hankel_sequence_types") or []

    states = sample_states(env, n_rank_samples)
    m      = compute_rank_metrics(agent, states)
    snap: dict = {"numerical_rank": m.numerical_rank}

    if hankel_seqs:
        hsnap = {}
        for s in hankel_seqs:
            hsnap[s] = hankel_rank_metrics(
                agent, env,
                sequence_type=s,
                n_steps=analysis_cfg.get("hankel_steps"),
                n_rows=analysis_cfg.get("hankel_n_rows"),
            )
        snap["hankel"] = hsnap

    return snap, m


def _progress(n_ep: int, desc: str, enable: bool):
    return tqdm(range(1, n_ep + 1), desc=desc, disable=not enable, dynamic_ncols=True)


def _postfix(rewards: list[float], window: int = 20, agent=None) -> dict[str, str]:
    recent = rewards[-window:] if rewards else [0.0]
    mean_r = sum(recent) / len(recent)
    out = {"ep_reward": f"{rewards[-1]:+.1f}" if rewards else "—",
           f"mean{len(recent)}": f"{mean_r:+.1f}"}
    eps = getattr(agent, "epsilon", None)
    if eps is not None:
        out["eps"] = f"{float(eps):.3f}"
    return out


def train_step_based(
    agent,
    env,
    cfg: dict,
    on_episode: Optional[EpisodeCallback] = None,
    progress: bool = True,
) -> TrainingLog:
    n_ep             = cfg["training"]["n_episodes"]
    checkpoint_every = cfg["training"]["rank_checkpoint_every"]
    eval_every       = cfg["training"].get("eval_every")
    eval_episodes    = cfg["training"].get("eval_episodes", 5)

    log        = TrainingLog()
    bar        = _progress(n_ep, f"{type(agent).__name__} train", progress)
    best_state = None  # snapshot of policy_net.state_dict() at best eval

    for ep in bar:
        obs, _         = env.reset()
        done, t, total = False, 0, 0.0
        while not done:
            action = agent.act(obs, training=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs if not terminated else None, terminated)
            obs    = next_obs
            t     += 1
            total += float(reward)

        log.durations.append(t)
        log.rewards.append(total)

        if ep % checkpoint_every == 0:
            snap, m = _checkpoint(agent, env, cfg)
            log.rank_history.append({"episode": ep, **snap})
            eps_attr = getattr(agent, "epsilon", None)
            eps_str  = f"  eps={float(eps_attr):.3f}" if eps_attr is not None else ""
            bar.write(f"  ep {ep:4d}/{n_ep}  dur={t:4d}  R={total:+.1f}{eps_str}  {m.summary()}")

        if eval_every and ep % eval_every == 0 and hasattr(agent, "policy_net"):
            mean_r = _evaluate(agent, env, eval_episodes)
            log.eval_history.append({"episode": ep, "mean_reward": mean_r})
            if mean_r > log.best_eval:
                log.best_eval    = mean_r
                log.best_episode = ep
                best_state       = copy.deepcopy(agent.policy_net.state_dict())
                bar.write(f"  ep {ep:4d}/{n_ep}  eval={mean_r:+.1f}  ★ new best")

        bar.set_postfix(_postfix(log.rewards, agent=agent))
        if on_episode is not None:
            on_episode(ep, t, total)

    if best_state is not None:
        agent.policy_net.load_state_dict(best_state)
        agent.target_net.load_state_dict(best_state)

    return log


def train_episode_based(
    agent,
    env,
    cfg: dict,
    on_episode: Optional[EpisodeCallback] = None,
    progress: bool = True,
) -> TrainingLog:
    n_ep             = cfg["training"]["n_episodes"]
    checkpoint_every = cfg["training"]["rank_checkpoint_every"]
    n_rank_samples   = cfg["training"]["n_rank_samples"]

    log = TrainingLog()
    bar = _progress(n_ep, f"{type(agent).__name__} train", progress)

    for ep in bar:
        obs, _         = env.reset()
        done, t, total = False, 0, 0.0
        while not done:
            action = agent.act(obs, training=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            obs    = next_obs
            t     += 1
            total += float(reward)
        agent.end_episode()

        log.durations.append(t)
        log.rewards.append(total)

        if ep % checkpoint_every == 0:
            snap, m = _checkpoint(agent, env, cfg)
            log.rank_history.append({"episode": ep, **snap})
            bar.write(f"  ep {ep:4d}/{n_ep}  dur={t:4d}  R={total:+.1f}  {m.summary()}")

        bar.set_postfix(_postfix(log.rewards, agent=agent))
        if on_episode is not None:
            on_episode(ep, t, total)

    return log


def train_ppo(
    agent,
    env,
    cfg: dict,
    on_episode: Optional[EpisodeCallback] = None,
    progress: bool = True,
) -> TrainingLog:
    import torch

    n_ep             = cfg["training"]["n_episodes"]
    checkpoint_every = cfg["training"]["rank_checkpoint_every"]
    n_rank_samples   = cfg["training"]["n_rank_samples"]

    log = TrainingLog()
    bar = _progress(n_ep, f"{type(agent).__name__} train", progress)

    for ep in bar:
        obs, _                    = env.reset()
        done, t, total, last_val  = False, 0, 0.0, 0.0
        while not done:
            action = agent.act(obs, training=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            obs    = next_obs
            t     += 1
            total += float(reward)
            if done and not terminated:
                s = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                with torch.no_grad():
                    last_val = float(agent.critic(s).item())
        agent.end_episode(last_value=last_val)

        log.durations.append(t)
        log.rewards.append(total)

        if ep % checkpoint_every == 0:
            snap, m = _checkpoint(agent, env, cfg)
            log.rank_history.append({"episode": ep, **snap})
            bar.write(f"  ep {ep:4d}/{n_ep}  dur={t:4d}  R={total:+.1f}  {m.summary()}")

        bar.set_postfix(_postfix(log.rewards, agent=agent))
        if on_episode is not None:
            on_episode(ep, t, total)

    return log


def train(
    agent,
    env,
    cfg: dict,
    on_episode: Optional[EpisodeCallback] = None,
    progress: bool = True,
) -> TrainingLog:
    """Pick the right driver based on ``cfg['agent']``."""
    name = cfg["agent"]
    if name == "ppo":
        return train_ppo(agent, env, cfg, on_episode=on_episode, progress=progress)
    if name == "monte_carlo":
        return train_episode_based(agent, env, cfg, on_episode=on_episode, progress=progress)
    return train_step_based(agent, env, cfg, on_episode=on_episode, progress=progress)
