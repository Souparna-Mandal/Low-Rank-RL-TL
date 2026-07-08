"""Thin config-driven builders so an experiment notebook is just wiring: change
config.yaml and it flows through here without editing the notebook.

Each builder passes a whole config section straight through with `**cfg[section]`,
so the config keys ARE the function parameters — adding a hyperparameter under
`environment`/`agent`/`training` needs no notebook change. The only things the
notebook still supplies are genuine code objects (the Q-network class and its
derived nn_extra_kwargs), not config keys.
"""
import pathlib

import numpy as np
import torch
import yaml

from environments.base_env import make_environment
from agents.q_agent import QAgent
from training import dqn_training_loop


def _resolve_device(name: str) -> str:
    """Map a config device request to an available torch device."""
    if name == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return name


def load_config(path="config.yaml") -> dict:
    """Load config.yaml, resolve the device, and seed torch/numpy. The resolved
    device is stashed at cfg["experiment"]["_device"] for the builders to read."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["experiment"]["_device"] = _resolve_device(cfg["experiment"]["device"])
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


def make_run_logger(cfg: dict):
    """A RunLogger under runs/<timestamp>/ when experiment.save_artifacts is set,
    else None (analysis renders inline). Imported lazily"""
    from analysis.run_logger import RunLogger
    if cfg["experiment"].get("save_artifacts", True):
        return RunLogger(pathlib.Path.cwd(), config_path="config.yaml")
    return None