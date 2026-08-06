"""Headless twin of exp1_hankel.ipynb's benchmark cell (single-seed pilot).

Writes to the SAME cache the notebook reads (results_hankel/<variant>_s<seed>.npz,
runs/<variant>_s<seed>/), so after this finishes the notebook's benchmark cell
prints `cached:` and its analysis cells load these results directly. Do not run
this and the notebook's benchmark cell at the same time — they would fight over
the same run dirs and npz files.

Run from this directory:  nohup python -u run_hankel_grid.py > hankel_grid_s0.log 2>&1 &
"""
import csv
import json
import pathlib
import random
import shutil
import sys
import time

import numpy as np
import torch
import yaml

import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment import load_config, build_env, build_agent, train
from agents.hankel_dqn_agent import HankelDQNAgent
from analysis.low_rank.hankel_policy import collect_hankel_sequences, _hankel_from_sequence
from analysis.low_rank.rank import compute_rank_metrics
from analysis.run_logger import RunLogger
from training import _greedy_episode_return

HERE = pathlib.Path(__file__).resolve().parent


class NatureCNN(torch.nn.Module):
    """Maps a (C, 84, 84) frame stack to Q-values of shape (n_actions,).
    Built by the agent via q_network(**nn_extra_kwargs); uint8 frames are
    normalised to [0, 1] inside forward (scale_obs=False keeps the buffer uint8)."""
    def __init__(self, in_channels, n_actions, fc_hidden=512):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),          torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),          torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            flat_dim = self.features(dummy).shape[1]
        self.head = torch.nn.Sequential(
            torch.nn.Linear(flat_dim, fc_hidden), torch.nn.ReLU(),
            torch.nn.Linear(fc_hidden, n_actions),
        )

    def forward(self, x):
        x = x.float() / 255.0
        return self.head(self.features(x))


# Must stay in sync with the notebook's VARIANTS so the cache keys line up.
VARIANTS = {
    "baseline": dict(hankel_weight=1e-2, warmup_grad_steps=10**9),  # diagnostics-only classical DQN
    "config":   dict(),                                             # config_hankel.yaml as-is
}
SEEDS = [0]
RESULTS = HERE / "results_hankel"
RESULTS.mkdir(exist_ok=True)


def resolve_cfg(overrides, seed):
    cfg = load_config(HERE / "config_hankel.yaml")
    cfg["experiment"]["seed"] = seed
    cfg["agent"].update(overrides)
    cfg["analysis"] = {"ep_freq": 25, "methods": []}
    return cfg


def cache_key(cfg):
    parts = {k: cfg[k] for k in ("environment", "network", "agent", "training")}
    parts["seed"] = cfg["experiment"]["seed"]
    return json.dumps(parts, sort_keys=True, default=str)


def is_cached(out_path, key):
    if not out_path.exists():
        return False
    with np.load(out_path) as d:
        return "cfg_json" in d.files and str(d["cfg_json"]) == key


def run_one(cfg, out_path, run_id, key):
    seed = cfg["experiment"]["seed"]
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    env = build_env(cfg)
    nn_extra = {"in_channels": env.observation_space.shape[0],
                "n_actions": env.action_space.n,
                "fc_hidden": cfg["network"]["fc_hidden"]}
    agent = build_agent(cfg, env, q_network=NatureCNN, nn_extra_kwargs=nn_extra,
                        agent_cls=HankelDQNAgent)
    diags = []
    def train_hook(_orig=agent.train):
        d = _orig()
        if d is not None:
            diags.append(d)
        return d
    agent.train = train_hook

    run_dir = HERE / "runs" / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    logger = RunLogger(HERE, config_path=str(HERE / "config_hankel.yaml"), run_id=run_id)
    with open(logger.dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    rewards = train(cfg, agent, env, run_logger=logger)

    eps = agent.epsilon
    agent.epsilon = 0.0
    evals = [_greedy_episode_return(agent, env, seed=30_000 + i) for i in range(20)]
    seqs = collect_hankel_sequences(agent, env, seed=777)
    eff_rank_q = compute_rank_metrics(_hankel_from_sequence(np.asarray(seqs["Hankel Q"])))[0]
    agent.epsilon = eps

    with open(logger.dir / "eval.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward"])
        w.writerows(enumerate(evals))

    diag_arrays = ({f"diag_{k}": np.array([d[k] for d in diags], float) for k in diags[0]}
                   if diags else {})
    np.savez(out_path, rewards=np.array(rewards, float), evals=np.array(evals, float),
             eff_rank_q=eff_rank_q, nan_skips=agent.nan_skips, cfg_json=key,
             **diag_arrays)
    env.close()


if __name__ == "__main__":
    for variant, ov in VARIANTS.items():
        for seed in SEEDS:
            out = RESULTS / f"{variant}_s{seed}.npz"
            cfg = resolve_cfg(ov, seed)
            key = cache_key(cfg)
            if is_cached(out, key):
                print("cached:", out.name, flush=True)
                continue
            if out.exists():
                print("config changed, re-running:", out.name, flush=True)
            print(f"starting {variant}_s{seed} "
                  f"(agent cfg: { {k: cfg['agent'][k] for k in ('hankel_weight', 'hankel_order', 'window_len', 'n_windows', 'gate_threshold')} })",
                  flush=True)
            t0 = time.time()
            run_one(cfg, out, run_id=f"{variant}_s{seed}", key=key)
            print(f"{out.name}: {time.time() - t0:.0f}s", flush=True)
    print("grid complete", flush=True)
