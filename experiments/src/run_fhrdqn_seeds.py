"""Parallel multi-seed launcher for the FHR-DQN comparison experiments.

One (arm, seed) training per subprocess — the nets are tiny, so a single run
only uses ~10-20% of the GPU and the 2 arms x N seeds fan-out runs comfortably
in parallel. Like run_autoregressive_recurrence.py, run it with the experiment
directory as cwd (config paths and cached/ are resolved against cwd):

    cd experiments/classical_control/dqn_mountaincar
    python ../../src/run_fhrdqn_seeds.py                 # all arms x config seeds, 6-wide
    python ../../src/run_fhrdqn_seeds.py --max-workers 3 # gentler fan-out
    python ../../src/run_fhrdqn_seeds.py --force         # rerun pairs already in the manifest

Child mode (one training in this process; used internally by the launcher):

    python ../../src/run_fhrdqn_seeds.py --arm fhr --seed 21

Completed runs are recorded in cached/fhrdqn_runs_manifest.json (run dirs
relative to the experiment dir), so a relaunch only runs missing/failed pairs. The
comparison notebooks import launch_all()/load_runs() from this file and read
rewards from each run dir's rewards.csv — the training loop rewrites that file
at every analysis tick, so partially finished runs are readable too. Child
stdout/stderr goes to cached/logs/fhrdqn_<arm>_seed<N>.log. Children are pinned to one
BLAS/OMP thread each (the fable launcher precedent) so the fan-out doesn't
oversubscribe the CPU.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch.nn as nn
import yaml

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent   # experiments/src
SRC = SCRIPTS_DIR.parents[1] / "src"                     # repo library code
CONFIG = "configs/config_fhrdqn.yaml"
MANIFEST = "cached/fhrdqn_runs_manifest.json"
LOGS = "cached/logs"
ARMS = ("baseline", "fhr")


class QNetwork(nn.Module):
    """Maps a state (obs_dim,) -> Q-values (n_actions,). Same MLP as all three
    comparison notebooks; built by the agent via q_network(**nn_extra_kwargs)."""
    def __init__(self, in_dim, out_dim, hidden_sizes=(64, 64)):
        super().__init__()
        layers, last = [], in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def run_one(arm, seed, episodes=None):
    """Child mode: one full training run in this process (cwd = experiment dir).
    The baseline arm strips the FHR term (fhr_weight = 0.0) — the single
    difference from the FHR arm. Returns the run directory."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from experiment import load_config, build_env, build_agent, train, make_run_logger
    from agents.fhrdqn_agent import FHRDQNAgent

    cfg = load_config(CONFIG, seed=seed)          # seeds torch / numpy / random
    if arm == "baseline":
        cfg["experiment"]["name"] += "_baseline"
        cfg["agent"]["fhr_weight"] = 0.0
    cfg["experiment"]["name"] += f"_seed{seed}"
    if episodes is not None:                      # smoke-test override
        cfg["training"]["no_episodes"] = episodes
    env = build_env(cfg)
    nn_extra_kwargs = {"in_dim": env.observation_space.shape[0],
                       "out_dim": env.action_space.n,
                       "hidden_sizes": cfg["network"]["hidden_sizes"]}
    agent = build_agent(cfg, env, QNetwork, nn_extra_kwargs, agent_cls=FHRDQNAgent)
    logger = make_run_logger(cfg, config_path=CONFIG, base_dir="cached")
    train(cfg, agent, env, run_logger=logger)
    return logger.dir


def _config_seeds(exp_dir):
    with open(exp_dir / CONFIG) as f:
        cfg = yaml.safe_load(f)
    return list(cfg["experiment"].get("seeds") or [cfg["experiment"]["seed"]])


def _load_manifest(exp_dir):
    path = exp_dir / MANIFEST
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"seeds": [], "runs": {arm: {} for arm in ARMS}}


def _run_recorded(exp_dir, manifest, arm, seed):
    rel = manifest["runs"].get(arm, {}).get(str(seed))
    return rel is not None and (exp_dir / rel / "rewards.csv").exists()


def launch_all(max_workers=6, force=False, exp_dir=None, episodes=None):
    """Fan out one subprocess per (arm, seed) from the config's seed list.
    Pairs already recorded in the manifest are skipped unless force=True.
    Returns the manifest dict; raises if any run fails (successes are kept,
    so a relaunch resumes from the failures)."""
    exp_dir = pathlib.Path(exp_dir or pathlib.Path.cwd()).resolve()
    seeds = _config_seeds(exp_dir)
    manifest = _load_manifest(exp_dir)
    manifest["seeds"] = seeds
    jobs = [(arm, s) for arm in ARMS for s in seeds
            if force or not _run_recorded(exp_dir, manifest, arm, s)]
    skipped = 2 * len(seeds) - len(jobs)
    if skipped:
        print(f"skipping {skipped} already-completed run(s) recorded in {MANIFEST}")
    if not jobs:
        return manifest

    (exp_dir / LOGS).mkdir(parents=True, exist_ok=True)
    child_env = dict(os.environ, MPLBACKEND="Agg", TQDM_DISABLE="1",
                     OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                     OPENBLAS_NUM_THREADS="1")
    lock = threading.Lock()
    failures = []

    def work(job):
        arm, seed = job
        log_path = exp_dir / LOGS / f"fhrdqn_{arm}_seed{seed}.log"
        cmd = [sys.executable, str(pathlib.Path(__file__).resolve()),
               "--arm", arm, "--seed", str(seed)]
        if episodes is not None:
            cmd += ["--episodes", str(episodes)]
        t0 = time.time()
        with open(log_path, "w") as lf:
            proc = subprocess.run(cmd, cwd=exp_dir, env=child_env,
                                  stdout=lf, stderr=subprocess.STDOUT)
        mins = (time.time() - t0) / 60
        run_dir = None
        for line in reversed(log_path.read_text().splitlines()):
            if line.startswith("RUN_DIR="):
                run_dir = line.removeprefix("RUN_DIR=")
                break
        if proc.returncode != 0 or run_dir is None:
            print(f"[{arm} seed {seed}] FAILED after {mins:.1f} min — see {log_path}")
            with lock:
                failures.append((arm, seed, str(log_path)))
            return
        rel = os.path.relpath(run_dir, exp_dir)
        with lock:
            manifest["runs"].setdefault(arm, {})[str(seed)] = rel
            with open(exp_dir / MANIFEST, "w") as f:
                json.dump(manifest, f, indent=2)
        print(f"[{arm} seed {seed}] done in {mins:.1f} min -> {rel}")

    print(f"launching {len(jobs)} run(s), {min(max_workers, len(jobs))} at a time: {jobs}")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(work, jobs))
    if failures:
        raise RuntimeError(f"{len(failures)} run(s) failed (completed runs are "
                           f"kept in {MANIFEST}; relaunch to retry): {failures}")
    return manifest


def load_runs(arm, exp_dir=None):
    """Manifest -> [{seed, cfg, rewards, run_dir}] in config-seed order, the
    structure the comparison notebooks' aggregation cells expect. cfg is the
    config.yaml copied into the run dir (note: the baseline copy does not
    reflect the in-memory fhr_weight=0 override)."""
    exp_dir = pathlib.Path(exp_dir or pathlib.Path.cwd()).resolve()
    manifest = _load_manifest(exp_dir)
    runs = []
    for seed in manifest["seeds"]:
        rel = manifest["runs"].get(arm, {}).get(str(seed))
        if rel is None or not (exp_dir / rel / "rewards.csv").exists():
            raise RuntimeError(f"no completed {arm} run for seed {seed} — "
                               f"launch_all() it first ({LOGS}/ has any failures)")
        run_dir = exp_dir / rel
        rewards = np.loadtxt(run_dir / "rewards.csv", delimiter=",", skiprows=1,
                             ndmin=2)[:, 1]
        with open(run_dir / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        runs.append({"seed": seed, "cfg": cfg, "rewards": rewards,
                     "run_dir": run_dir})
    return runs


def _load_run_agent(run_dir, device="cpu"):
    """Rebuild the trained agent from a run dir (config copy + final
    checkpoint) with an rgb_array eval env wrapped exactly like training —
    the normalise/clip wrappers are load-bearing (e.g. MountainCar)."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from experiment import build_env, build_agent
    from agents.fhrdqn_agent import FHRDQNAgent
    run_dir = pathlib.Path(run_dir)
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["experiment"]["_device"] = device
    env = build_env(cfg, render_mode="rgb_array")
    nn_extra_kwargs = {"in_dim": env.observation_space.shape[0],
                       "out_dim": env.action_space.n,
                       "hidden_sizes": cfg["network"]["hidden_sizes"]}
    agent = build_agent(cfg, env, QNetwork, nn_extra_kwargs, agent_cls=FHRDQNAgent)
    agent.load(run_dir / "checkpoints" / "final.pt")
    return agent, env


def record_final_videos(arm, exp_dir=None):
    """One greedy-policy episode per manifest run of `arm`, saved as
    <run_dir>/videos/epfinal-episode-0.mp4 -> [(seed, mp4 path)]. The rollout
    is reset with the run's own seed."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from analysis.visualisations.rollout_video import record_greedy_episode
    out = []
    for r in load_runs(arm, exp_dir):
        agent, env = _load_run_agent(r["run_dir"])
        prefix = record_greedy_episode(agent, env, str(r["run_dir"] / "videos"),
                                       episode="final", seed=r["seed"])
        env.close()
        out.append((r["seed"], r["run_dir"] / "videos" / f"{prefix}-episode-0.mp4"))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=ARMS,
                        help="child mode: run this single arm (requires --seed)")
    parser.add_argument("--seed", type=int, help="child mode: seed for --arm")
    parser.add_argument("--episodes", type=int, default=None,
                        help="override training.no_episodes (smoke tests)")
    parser.add_argument("--max-workers", type=int, default=6,
                        help="parallel trainings in launcher mode (default 6)")
    parser.add_argument("--force", action="store_true",
                        help="rerun (arm, seed) pairs already in the manifest")
    args = parser.parse_args()
    if args.arm is not None:
        if args.seed is None:
            parser.error("--arm requires --seed")
        run_dir = run_one(args.arm, args.seed, episodes=args.episodes)
        print(f"RUN_DIR={run_dir}")
    else:
        launch_all(max_workers=args.max_workers, force=args.force,
                   episodes=args.episodes)


if __name__ == "__main__":
    main()
