""" Each builder passes a whole config section straight through with `**cfg[section]` in the jupyter notebooks where experiments are run.
So the config keys are the function parameters — adding a hyperparameter under
`environment`/`agent`/`training` needs no notebook change. The only things the
notebook still supplies are objects like the Q-network class and its
derived nn_extra_kwargs.
"""
import pathlib
import random

import numpy as np
import torch
import yaml

from environments.base_env import make_environment
from agents.q_agent import QAgent
from training import dqn_training_loop, policy_iteration_loop
from utils.device import resolve_device


def load_config(path="config.yaml", seed=None) -> dict:
    """Load config.yaml, resolve the device, and seed torch/numpy/random. The
    resolved device is stashed at cfg["experiment"]["_device"] for the builders
    to read. Python's `random` is seeded too because the replay buffers sample
    through it — without this, matched-seed variant comparisons are unmatched
    at the replay-sampling level. Pass seed to override
    cfg["experiment"]["seed"] — multi-seed comparison notebooks call this once
    per entry of cfg["experiment"]["seeds"]."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["experiment"]["_device"] = resolve_device(cfg["experiment"]["device"])
    if seed is not None:
        cfg["experiment"]["seed"] = seed
    seed = cfg["experiment"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
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


def build_agent(cfg: dict, env, q_network=None, nn_extra_kwargs: dict = None,
                agent_cls=QAgent):
    """Build the agent from cfg["agent"]. Pass agent_cls to select the agent
    (RainbowDQNAgent, TabularPolicyIterationAgent, ...; default QAgent). Leave
    q_network=None for network-less agents like tabular PI. The config keys
    under `agent` must match that class's constructor."""
    kwargs = dict(cfg["agent"])
    if q_network is not None:
        kwargs["q_network"] = q_network
        kwargs["nn_extra_kwargs"] = nn_extra_kwargs
    return agent_cls(
        **kwargs,
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


def train_policy_iteration(cfg: dict, agent, env, run_logger=None, DEBUG=False):
    """policy_iteration_loop driven by cfg["training"]."""
    return policy_iteration_loop(
        agent, env,
        **cfg["training"],
        np_seed=cfg["experiment"]["seed"],
        analysis_config=cfg["analysis"],
        run_logger=run_logger,
        DEBUG=DEBUG,
    )


def make_run_logger(cfg: dict, config_path: str = "config.yaml"):
    """A RunLogger under runs/<experiment.name>_<timestamp>/ when
    experiment.save_artifacts is set, else None (analysis renders inline).
    Imported lazily. Pass config_path"""
    from analysis.run_logger import RunLogger
    if cfg["experiment"].get("save_artifacts", True):
        return RunLogger(pathlib.Path.cwd(), config_path=config_path,
                         name=cfg["experiment"].get("name"))
    return None
