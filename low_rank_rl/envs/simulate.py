"""Roll out a trained policy on an env and render the result as an animation.

Training envs are built without ``render_mode`` for speed, so ``rollout_policy``
spins up a fresh env with ``render_mode="rgb_array"`` using the same
``make_env`` defaults (action discretisation / obs normalisation) that were
used at train time, so the agent sees identically-shaped observations.

Typical notebook usage:

    frames, info = rollout_policy(agent, cfg["env_id"], env_kwargs=cfg.get("env_kwargs"))
    HTML(animate_rollout(frames).to_jshtml())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.figure import Figure

from low_rank_rl.envs.registry import make_env


@dataclass
class RolloutResult:
    frames:  list[np.ndarray] = field(default_factory=list)
    rewards: list[float]      = field(default_factory=list)
    actions: list[int]        = field(default_factory=list)
    states:  list[np.ndarray] = field(default_factory=list)
    terminated: bool          = False
    truncated:  bool          = False

    @property
    def total_reward(self) -> float:
        return float(sum(self.rewards))

    @property
    def n_steps(self) -> int:
        return len(self.actions)


def rollout_policy(
    agent,
    env_id: str,
    env_kwargs: Optional[dict[str, Any]] = None,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    greedy: bool = True,
) -> RolloutResult:
    """Run one episode greedily and collect RGB frames.

    ``greedy=True`` passes ``training=False`` to ``agent.act``, which
    switches every agent in this package to its deterministic/argmax policy.
    """
    env = make_env(env_id, render_mode="rgb_array", **(env_kwargs or {}))
    try:
        reset_kwargs = {"seed": seed} if seed is not None else {}
        obs, _ = env.reset(**reset_kwargs)

        result = RolloutResult()
        frame  = env.render()
        if frame is not None:
            result.frames.append(np.asarray(frame))

        step_cap = max_steps if max_steps is not None else 10_000
        for _ in range(step_cap):
            action = int(agent.act(obs, training=not greedy))
            obs, reward, terminated, truncated, _ = env.step(action)

            result.states.append(np.asarray(obs).copy())
            result.actions.append(action)
            result.rewards.append(float(reward))
            frame = env.render()
            if frame is not None:
                result.frames.append(np.asarray(frame))

            if terminated or truncated:
                result.terminated = bool(terminated)
                result.truncated  = bool(truncated)
                break

        return result
    finally:
        env.close()


def animate_rollout(
    frames: list[np.ndarray],
    interval: int = 50,
    figsize: tuple[float, float] = (6, 4),
    title: str | None = None,
) -> animation.FuncAnimation:
    """Return a ``FuncAnimation`` — in a notebook, display with
    ``HTML(animate_rollout(frames).to_jshtml())``."""
    assert len(frames) > 0, "Cannot animate an empty frame list."

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    if title:
        ax.set_title(title)
    im = ax.imshow(frames[0])

    def _update(i: int):
        im.set_data(frames[i])
        return (im,)

    ani = animation.FuncAnimation(
        fig, _update, frames=len(frames),
        interval=interval, blit=True, repeat=False,
    )
    plt.close(fig)
    return ani


def save_rollout_gif(
    frames: list[np.ndarray],
    path: str,
    fps: int = 30,
    figsize: tuple[float, float] = (6, 4),
) -> None:
    ani = animate_rollout(frames, interval=int(1000 / fps), figsize=figsize)
    ani.save(path, writer="pillow", fps=fps)


def plot_rollout_strip(
    frames: list[np.ndarray],
    n_cols: int = 8,
    title: str | None = None,
) -> Figure:
    """Evenly sample ``n_cols`` frames and show them as a single row — a quick
    static summary for when a full animation is overkill."""
    assert len(frames) > 0, "Cannot plot an empty frame list."
    n = min(n_cols, len(frames))
    idx = np.linspace(0, len(frames) - 1, n, dtype=int)

    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.5))
    if n == 1:
        axes = [axes]
    for ax, i in zip(axes, idx):
        ax.imshow(frames[i])
        ax.set_title(f"t={i}", fontsize=9)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
