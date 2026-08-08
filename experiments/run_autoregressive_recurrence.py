"""Run one baseline-DQN training run with the autoregressive value-recurrence probe.

Shared by the CartPole and Acrobot notebooks so both experiments are driven by
exactly the same code path and differ only in their config file.

From the command line, with the experiment directory as the working directory:

    cd experiments/dqn_cartpole
    python ../run_autoregressive_recurrence.py config_autoregressive_recurrence.yaml

From a notebook:

    from run_autoregressive_recurrence import run_experiment
    run_directory, episode_rewards = run_experiment("config_autoregressive_recurrence.yaml")

The run writes everything under <experiment dir>/runs/<timestamp>/, including
autoregressive_value_metrics.csv, autoregressive_value_coefficients.csv and
autoregressive_rollouts/epNNNNNN.npz, which are what the plotting helpers and
the result viewer app read.
"""
import argparse
import pathlib
import sys

import torch
import torch.nn as nn

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from experiment import (build_agent, build_env, load_config,  # noqa: E402
                        make_run_logger, train)


class QNetwork(nn.Module):
    """Maps a state of shape (observation_dim,) to one Q-value per action.

    Plain multi-layer perceptron with ReLU activations, matching the network
    the other DQN experiments in this repository use, so the value signal this
    probe analyses is the ordinary baseline one and not a special architecture.
    """

    def __init__(self, in_dim, out_dim, hidden_sizes=(128, 128)):
        super().__init__()
        layers, last_width = [], in_dim
        for hidden_width in hidden_sizes:
            layers += [nn.Linear(last_width, hidden_width), nn.ReLU()]
            last_width = hidden_width
        layers.append(nn.Linear(last_width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def run_experiment(config_path="config_autoregressive_recurrence.yaml",
                   device=None, no_episodes=None, seed=None):
    """Train one baseline DQN agent with the probe attached at every checkpoint.

    Args:
        config_path: path to the experiment's config, relative to the working
            directory (which must be the experiment directory).
        device: optional override of the config's device, e.g. "cpu".
        no_episodes: optional override of the training budget, for quick checks.
        seed: optional override of the experiment seed, to run several seeds.

    Returns:
        (run_directory, episode_rewards). run_directory is where every artifact
        for this run was written.
    """
    config = load_config(config_path)
    if device is not None:
        config["experiment"]["device"] = device
        config["experiment"]["_device"] = device
    if no_episodes is not None:
        config["training"]["no_episodes"] = no_episodes
    if seed is not None:
        config["experiment"]["seed"] = seed
        torch.manual_seed(seed)

    environment = build_env(config)
    network_kwargs = {
        "in_dim": environment.observation_space.shape[0],
        "out_dim": environment.action_space.n,
        "hidden_sizes": config["network"]["hidden_sizes"],
    }
    agent = build_agent(config, environment, q_network=QNetwork,
                        nn_extra_kwargs=network_kwargs)
    run_logger = make_run_logger(config, config_path=config_path)
    episode_rewards = train(config, agent, environment, run_logger=run_logger)
    environment.close()
    return (run_logger.dir if run_logger is not None else None), episode_rewards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?",
                        default="config_autoregressive_recurrence.yaml")
    parser.add_argument("--device", default=None,
                        help="override the config device, e.g. cpu")
    parser.add_argument("--episodes", type=int, default=None,
                        help="override the training budget")
    parser.add_argument("--seed", type=int, default=None,
                        help="override the experiment seed")
    arguments = parser.parse_args()
    run_directory, episode_rewards = run_experiment(
        arguments.config, device=arguments.device,
        no_episodes=arguments.episodes, seed=arguments.seed)
    print(f"finished {len(episode_rewards)} episodes; artifacts in {run_directory}")


if __name__ == "__main__":
    main()
