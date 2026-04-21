"""Train an agent from a YAML config and run the full low-rank analysis suite."""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from low_rank_rl.envs import make_env
from low_rank_rl.agents import DQNAgent, QLearningAgent, SarsaAgent, MonteCarloAgent, PPOAgent
from low_rank_rl.analysis.rank import compute_rank_metrics, sample_states
from low_rank_rl.analysis.tensor import build_value_tensor, hosvd_spectra
from low_rank_rl.analysis.hankel import hankel_rank_metrics
from low_rank_rl.analysis.successor import compare_shift_rank
from low_rank_rl.visualization.training import plot_episode_durations
from low_rank_rl.visualization.rank_analysis import (
    plot_singular_value_spectrum,
    plot_hosvd_spectra,
    plot_rank_vs_episode,
    plot_hankel_spectrum,
    plot_shift_comparison,
)
from low_rank_rl.visualization.value_fn import plot_value_heatmap


def build_agent(agent_name: str, env, kwargs: dict):
    obs_space = env.observation_space
    n_obs     = obs_space.shape[0]
    n_actions = env.action_space.n

    if agent_name == "dqn":
        return DQNAgent(n_obs=n_obs, n_actions=n_actions, **kwargs)
    if agent_name in ("qlearning", "q_learning"):
        return QLearningAgent(n_actions=n_actions, obs_low=obs_space.low, obs_high=obs_space.high, **kwargs)
    if agent_name == "sarsa":
        return SarsaAgent(n_actions=n_actions, obs_low=obs_space.low, obs_high=obs_space.high, **kwargs)
    if agent_name == "monte_carlo":
        return MonteCarloAgent(n_actions=n_actions, obs_low=obs_space.low, obs_high=obs_space.high, **kwargs)
    if agent_name == "ppo":
        return PPOAgent(n_obs=n_obs, n_actions=n_actions, **kwargs)
    raise ValueError(f"Unknown agent: {agent_name!r}")


def _rank_snapshot(agent, env, n_rank_samples):
    states = sample_states(env, n_rank_samples)
    m      = compute_rank_metrics(agent, states)
    return {
        "numerical_rank":  m.numerical_rank,
        "stable_rank":     m.stable_rank,
        "effective_rank":  m.effective_rank,
        "normalised_rank": m.normalised_numerical_rank,
    }, m


def train_step_based(agent, env, cfg):
    n_ep             = cfg["training"]["n_episodes"]
    checkpoint_every = cfg["training"]["rank_checkpoint_every"]
    n_rank_samples   = cfg["training"]["n_rank_samples"]

    durations, rank_history = [], []
    for ep in range(1, n_ep + 1):
        obs, _  = env.reset()
        done, t = False, 0
        while not done:
            action = agent.act(obs, training=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs if not terminated else None, terminated)
            obs = next_obs
            t  += 1
        durations.append(t)

        if ep % checkpoint_every == 0:
            snap, m = _rank_snapshot(agent, env, n_rank_samples)
            rank_history.append({"episode": ep, **snap})
            print(f"  ep {ep:4d}/{n_ep}  dur={t:4d}  {m.summary()}")

    return durations, rank_history


def train_episode_based(agent, env, cfg):
    n_ep             = cfg["training"]["n_episodes"]
    checkpoint_every = cfg["training"]["rank_checkpoint_every"]
    n_rank_samples   = cfg["training"]["n_rank_samples"]

    durations, rank_history = [], []
    for ep in range(1, n_ep + 1):
        obs, _  = env.reset()
        done, t = False, 0
        while not done:
            action = agent.act(obs, training=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            t  += 1
        agent.end_episode()
        durations.append(t)

        if ep % checkpoint_every == 0:
            snap, m = _rank_snapshot(agent, env, n_rank_samples)
            rank_history.append({"episode": ep, **snap})
            print(f"  ep {ep:4d}/{n_ep}  dur={t:4d}  {m.summary()}")

    return durations, rank_history


def train_ppo(agent: PPOAgent, env, cfg):
    import torch

    n_ep             = cfg["training"]["n_episodes"]
    checkpoint_every = cfg["training"]["rank_checkpoint_every"]
    n_rank_samples   = cfg["training"]["n_rank_samples"]

    durations, rank_history = [], []
    for ep in range(1, n_ep + 1):
        obs, _             = env.reset()
        done, t, last_val  = False, 0, 0.0
        while not done:
            action = agent.act(obs, training=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            t  += 1
            if done and not terminated:
                s = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                with torch.no_grad():
                    last_val = float(agent.critic(s).item())
        agent.end_episode(last_value=last_val)
        durations.append(t)

        if ep % checkpoint_every == 0:
            snap, m = _rank_snapshot(agent, env, n_rank_samples)
            rank_history.append({"episode": ep, **snap})
            print(f"  ep {ep:4d}/{n_ep}  dur={t:4d}  {m.summary()}")

    return durations, rank_history


def run_analysis(agent, env, cfg: dict, out_dir: pathlib.Path) -> None:
    n_rank_samples = cfg["training"]["n_rank_samples"]
    analysis_cfg   = cfg.get("analysis", {})

    print("\n-- Q-matrix rank analysis --")
    states  = sample_states(env, n_rank_samples)
    metrics = compute_rank_metrics(agent, states)
    print(metrics.summary())
    plot_singular_value_spectrum(metrics).savefig(out_dir / "q_spectrum.png", dpi=150)

    print("\n-- Value tensor (HOSVD) --")
    V_tensor = build_value_tensor(
        agent, env,
        dims=analysis_cfg.get("value_tensor_dims"),
        n_bins=analysis_cfg.get("value_tensor_bins", 20),
    )
    plot_hosvd_spectra(hosvd_spectra(V_tensor)).savefig(out_dir / "hosvd_spectra.png", dpi=150)
    print(f"  Tensor shape: {V_tensor.shape}")

    print("\n-- Hankel analysis --")
    for seq_type in ("value", "q_taken"):
        hm = hankel_rank_metrics(
            agent, env,
            sequence_type=seq_type,
            n_steps=analysis_cfg.get("hankel_steps", 500),
            n_rows=analysis_cfg.get("hankel_n_rows", None),
        )
        print(f"  {hm.summary()}")
        plot_hankel_spectrum(hm).savefig(out_dir / f"hankel_{seq_type}.png", dpi=150)

    print("\n-- Successor measure shift comparison --")
    probe_states = sample_states(env, min(64, n_rank_samples))
    comparison   = compare_shift_rank(agent, env, probe_states)
    print(comparison.summary())
    plot_shift_comparison(comparison).savefig(out_dir / "successor_shift.png", dpi=150)

    print("\n-- Value function heatmap --")
    plot_value_heatmap(agent, env).savefig(out_dir / "value_heatmap.png", dpi=150)
    print(f"\nAll figures saved to {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an RL agent and run low-rank analysis.")
    parser.add_argument("--config",   required=True, help="Path to YAML config file")
    parser.add_argument("--episodes", type=int, default=None, help="Override n_episodes")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.episodes is not None:
        cfg["training"]["n_episodes"] = args.episodes

    env     = make_env(cfg["env_id"], **cfg.get("env_kwargs", {}))
    agent   = build_agent(cfg["agent"], env, cfg.get("agent_kwargs", {}))
    out_dir = pathlib.Path(cfg["output"]["save_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Env     : {cfg['env_id']}")
    print(f"Agent   : {agent}")
    print(f"Output  : {out_dir}")
    print(f"Episodes: {cfg['training']['n_episodes']}\n")

    agent_name = cfg["agent"]
    if agent_name == "ppo":
        durations, rank_history = train_ppo(agent, env, cfg)
    elif agent_name == "monte_carlo":
        durations, rank_history = train_episode_based(agent, env, cfg)
    else:
        durations, rank_history = train_step_based(agent, env, cfg)

    plot_episode_durations(durations).savefig(out_dir / "training_durations.png", dpi=150)
    if rank_history:
        plot_rank_vs_episode(rank_history).savefig(out_dir / "rank_vs_episode.png", dpi=150)

    if cfg["output"].get("save_agent", True):
        agent.save(out_dir / "agent.pt")

    run_analysis(agent, env, cfg, out_dir)
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
