"""Train an agent from a YAML config and run the full low-rank analysis suite.

Training loops live in ``low_rank_rl.training`` and are shared with
``experiments/notebook.ipynb``. The DQN loop follows the PyTorch tutorial
(https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) and
the PPO loop follows OpenAI Spinning Up
(https://spinningup.openai.com/en/latest/algorithms/ppo.html).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from low_rank_rl.envs               import make_env
from low_rank_rl.agents             import DQNAgent, QLearningAgent, SarsaAgent, MonteCarloAgent, PPOAgent
from low_rank_rl.training           import train
from low_rank_rl.analysis.rank      import compute_rank_metrics, sample_states, canonical_subsample
from low_rank_rl.analysis.tensor    import build_value_tensor, hosvd_spectra
from low_rank_rl.analysis.hankel    import hankel_rank_metrics
from low_rank_rl.analysis.successor import compare_shift_rank
from low_rank_rl.visualization.training      import plot_episode_durations, plot_episode_rewards
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
        return QLearningAgent(env=env, n_actions=n_actions, **kwargs)
    if agent_name == "sarsa":
        return SarsaAgent(n_actions=n_actions, obs_low=obs_space.low, obs_high=obs_space.high, **kwargs)
    if agent_name == "monte_carlo":
        return MonteCarloAgent(n_actions=n_actions, obs_low=obs_space.low, obs_high=obs_space.high, **kwargs)
    if agent_name == "ppo":
        return PPOAgent(n_obs=n_obs, n_actions=n_actions, **kwargs)
    raise ValueError(f"Unknown agent: {agent_name!r}")


def run_analysis(agent, env, cfg: dict, out_dir: pathlib.Path) -> None:
    n_rank_samples = cfg["training"]["n_rank_samples"]
    analysis_cfg   = cfg.get("analysis", {})

    print("\n-- Q-matrix rank analysis --")
    states  = sample_states(env, n_rank_samples)
    metrics = compute_rank_metrics(agent, states)
    print(f"  probe states: {states.shape[0]}")
    print(metrics.summary())
    plot_singular_value_spectrum(metrics).savefig(out_dir / "q_spectrum.png", dpi=150)

    print("\n-- Value tensor (HOSVD) --")
    V_tensor = build_value_tensor(
        agent, env,
        dims=analysis_cfg.get("value_tensor_dims"),
        n_bins=analysis_cfg.get("value_tensor_bins"),
    )
    plot_hosvd_spectra(hosvd_spectra(V_tensor)).savefig(out_dir / "hosvd_spectra.png", dpi=150)
    print(f"  Tensor shape: {V_tensor.shape}")

    print("\n-- Hankel analysis --")
    for seq_type in ("value", "q_taken"):
        hm = hankel_rank_metrics(
            agent, env,
            sequence_type=seq_type,
            n_steps=analysis_cfg.get("hankel_steps"),
            n_rows=analysis_cfg.get("hankel_n_rows"),
        )
        print(f"  {hm.summary()}")
        plot_hankel_spectrum(hm).savefig(out_dir / f"hankel_{seq_type}.png", dpi=150)

    print("\n-- Successor measure shift comparison --")
    probe_states = canonical_subsample(env, min(64, n_rank_samples))
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

    log = train(agent, env, cfg, progress=True)

    plot_episode_durations(log.durations).savefig(out_dir / "training_durations.png", dpi=150)
    plot_episode_rewards(log.rewards).savefig(out_dir / "training_rewards.png", dpi=150)
    if log.rank_history:
        plot_rank_vs_episode(log.rank_history).savefig(out_dir / "rank_vs_episode.png", dpi=150)

    if cfg["output"].get("save_agent", True):
        agent.save(out_dir / "agent.pt")

    run_analysis(agent, env, cfg, out_dir)
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
