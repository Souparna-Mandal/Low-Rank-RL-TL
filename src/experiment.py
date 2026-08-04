""" Each builder passes a whole config section straight through with `**cfg[section]` in the jupyter notebooks where experiments are run.
So the config keys are the function parameters — adding a hyperparameter under
`environment`/`agent`/`training` needs no notebook change. The only things the
notebook still supplies are objects like the Q-network class and its
derived nn_extra_kwargs.
"""
import pathlib

import numpy as np
import torch
import yaml

from environments.base_env import make_environment
from agents.q_agent import QAgent
from training import dqn_training_loop
from utils.device import resolve_device


def load_config(path="config.yaml") -> dict:
    """Load config.yaml, resolve the device, and seed torch/numpy. The resolved
    device is stashed at cfg["experiment"]["_device"] for the builders to read."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["experiment"]["_device"] = resolve_device(cfg["experiment"]["device"])
    seed = cfg["experiment"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    return cfg


def build_env(cfg: dict, render_mode=None):
    """
    Pass render_mode to override (e.g. an
    rgb_array eval env for video)."""
    env_cfg = dict(cfg["environment"])
    env_name = env_cfg.pop("name")
    if render_mode is not None:
        env_cfg["render_mode"] = render_mode
    return make_environment(env_name, **env_cfg)


def build_agent(cfg: dict, env, q_network, nn_extra_kwargs: dict):
    return QAgent(
        **cfg["agent"],
        q_network=q_network,
        nn_extra_kwargs=nn_extra_kwargs,
        env=env,
        device=cfg["experiment"]["_device"],
    )


def build_ppo_agent(cfg: dict, env):
    """PPOAgent from cfg["agent"] and cfg["network"]["hidden_sizes"];
    see experiments/*/config_ppo.yaml."""
    from agents.ppo_agent import PPOAgent
    return PPOAgent(
        **cfg["agent"],
        hidden_sizes=tuple(cfg["network"]["hidden_sizes"]),
        env=env,
        device=cfg["experiment"]["_device"],
    )


def train(cfg: dict, agent, env, run_logger=None, DEBUG=False):
    """dqn_training_loop"""
    return dqn_training_loop(
        agent, env,
        **cfg["training"],
        np_seed=cfg["experiment"]["seed"],
        analysis_config=cfg["analysis"],
        atari=bool(cfg["environment"].get("atari")),
        run_logger=run_logger,
        DEBUG=DEBUG,
    )


def train_ppo(cfg: dict, agent, env, DEBUG=False, progress=True):
    """ppo_training_loop driven by cfg["training"] (no analysis/run_logger
    hooks — PPO results land in the notebook or a results_ppo/ cache).
    progress=False silences the tqdm bar."""
    from ppo_training import ppo_training_loop
    return ppo_training_loop(
        agent, env,
        **cfg["training"],
        np_seed=cfg["experiment"]["seed"],
        DEBUG=DEBUG,
        progress=progress,
    )


def make_run_logger(cfg: dict):
    """A RunLogger under runs/<timestamp>/ when experiment.save_artifacts is set,
    else None (analysis renders inline). Imported lazily"""
    from analysis.run_logger import RunLogger
    if cfg["experiment"].get("save_artifacts", True):
        return RunLogger(pathlib.Path.cwd(), config_path="config.yaml")
    return None
