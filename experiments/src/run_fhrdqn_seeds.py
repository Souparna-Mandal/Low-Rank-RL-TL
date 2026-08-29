"""Parallel multi-seed launcher for the FHR-DQN comparison experiments.

One (arm, seed) training per subprocess — the nets are tiny, so a single run
only uses ~10-20% of the GPU and the 2 arms x N seeds fan-out runs comfortably
in parallel. Like run_autoregressive_recurrence.py, run it with the experiment
directory as cwd (config paths and cached/ are resolved against cwd):

    cd experiments/classical_control/dqn_mountaincar
    python ../../src/run_fhrdqn_seeds.py                 # all arms x config seeds, 6-wide
    python ../../src/run_fhrdqn_seeds.py --max-workers 3 # gentler fan-out
    python ../../src/run_fhrdqn_seeds.py --force         # rerun pairs already in the manifest

Numbered FHR experiments: the config's experiment.fhr_experiments block maps an
experiment number to ONE full FHR parameter set — any subset of fhr_weight,
fhr_order, reward_lags, warmup_grad_steps, c_learning_rate; unspecified params
inherit the agent block. --experiment N trains that arm (run names tagged
_exp<N>) plus the shared baseline (fhr_weight 0, plain _baseline names — reused
from the manifest if already trained), over all config seeds. Several numbers
train several experiments in one launch. Everything records into the single
cached/fhrdqn_runs_manifest.json under arm keys "baseline"/"fhr"/"exp<N>", so
the result viewer's compare mode shows baseline vs each exp<N> as variants:

    python ../../src/run_fhrdqn_seeds.py --experiment 2
    python ../../src/run_fhrdqn_seeds.py --experiment 1 2 3

Run at most one launcher at a time per experiment dir — concurrent launchers
would race on the manifest file and the shared baseline arm.

Child mode (one training in this process; used internally by the launcher):

    python ../../src/run_fhrdqn_seeds.py --arm fhr --seed 21
    python ../../src/run_fhrdqn_seeds.py --arm fhr --seed 21 \
        --agent-overrides '{"fhr_weight": 0.1, "fhr_order": 2}' --name-tag exp2

Completed runs are recorded in cached/fhrdqn_runs_manifest.json (run dirs
relative to the experiment dir), so a relaunch only runs missing/failed pairs. The
comparison notebooks import launch_all()/load_runs() from this file and read
rewards from each run dir's rewards.csv — the training loop rewrites that file
at every analysis tick, so partially finished runs are readable too. Child
stdout/stderr goes to cached/logs/fhrdqn_<key>_seed<N>.log, where <key> is
baseline, fhr, or exp<N>. Children are pinned to one BLAS/OMP thread each  so the fan-out doesn't
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


def run_one(arm, seed, episodes=None, agent_overrides=None, name_tag=None):
    """Child mode: one full training run in this process (cwd = experiment dir).
    The baseline arm strips the FHR term (fhr_weight = 0.0) — the single
    difference from the FHR arm; a numbered-experiment arm applies its FHR
    parameter set on top of the config's agent block via `agent_overrides`.
    name_tag (e.g. exp2) lands in the run-dir name before the seed token so the
    result viewer groups each experiment as its own variant. Returns the run
    directory."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from experiment import load_config, build_env, build_agent, train, make_run_logger
    from agents.fhrdqn_agent import FHRDQNAgent

    cfg = load_config(CONFIG, seed=seed)          # seeds torch / numpy / random
    if arm == "baseline":
        cfg["agent"]["fhr_weight"] = 0.0
    elif agent_overrides:
        cfg["agent"].update(agent_overrides)
    if name_tag:
        cfg["experiment"]["name"] += f"_{name_tag}"
    elif arm == "baseline":
        cfg["experiment"]["name"] += "_baseline"  # shared-baseline naming
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
    return {"seeds": [], "runs": {}}


def _run_recorded(exp_dir, manifest, arm, seed):
    rel = manifest["runs"].get(arm, {}).get(str(seed))
    return rel is not None and (exp_dir / rel / "rewards.csv").exists()


# The FHR parameter set a numbered experiment may override — everything else
# (net size, replay, epsilon schedule, ...) stays identical to the baseline so
# an exp<N>-vs-baseline gap is attributable to the FHR block alone.
FHR_PARAMS = ("fhr_weight", "fhr_order", "reward_lags",
              "warmup_grad_steps", "c_learning_rate", "fhr_lag_source",
              "c_predictor", "prioritized_replay",
              "rampdown_reward_threshold", "rampdown_penalty_threshold",
              "rampdown_penalty_topk", "rampdown_patience_eps",
              "rampdown_episodes")


def _experiment_overrides(exp_dir, experiment):
    """experiment.fhr_experiments[<experiment>] from the config: ONE full FHR
    parameter set (any subset of FHR_PARAMS; the rest inherit the agent block)
    that the numbered experiment trains against the shared baseline."""
    with open(exp_dir / CONFIG) as f:
        cfg = yaml.safe_load(f)
    sweeps = cfg["experiment"].get("fhr_experiments") or {}
    overrides = sweeps.get(experiment, sweeps.get(str(experiment)))
    if overrides is None:
        raise ValueError(
            f"experiment {experiment} is not defined under "
            f"experiment.fhr_experiments in {CONFIG} "
            f"(defined: {sorted(map(str, sweeps)) or 'none'})")
    if not isinstance(overrides, dict) or not overrides:
        raise ValueError(
            f"experiment {experiment} must be a non-empty mapping of FHR "
            f"params, e.g. {{fhr_weight: 0.1, fhr_order: 2}} — got "
            f"{overrides!r} (the old list-of-weights form is not supported)")
    unknown = sorted(set(overrides) - set(FHR_PARAMS))
    if unknown:
        raise ValueError(f"experiment {experiment} overrides unknown FHR "
                         f"params {unknown} — allowed: {list(FHR_PARAMS)}")
    if float(overrides.get("fhr_weight", 1.0)) == 0.0:
        raise ValueError("fhr_weight 0 is the baseline arm, which every "
                         "launch already includes — drop it from the sweep")
    return overrides


def _arm_specs(exp_dir, experiments):
    """[(manifest/log key, extra child argv, agent overrides)] for one launch:
    the legacy two arms, or the shared baseline + one arm per requested
    experiment number. Overrides travel to the child as JSON (exact float
    round-trip) and are also recorded in the manifest, because the run dir's
    config.yaml copy does not reflect them — rebuilding a trained agent
    (record_final_videos) must re-apply them or coefficient shapes mismatch."""
    baseline = ("baseline", ["--arm", "baseline"], {"fhr_weight": 0.0})
    if not experiments:
        return [baseline, ("fhr", ["--arm", "fhr"], None)]
    if len(set(experiments)) != len(experiments):
        raise ValueError(f"duplicate experiment numbers: {experiments}")
    specs = [baseline]                              # shared with legacy runs
    for n in experiments:
        overrides = _experiment_overrides(exp_dir, n)
        specs.append((f"exp{n}", ["--arm", "fhr",
                                  "--agent-overrides", json.dumps(overrides),
                                  "--name-tag", f"exp{n}"], overrides))
    return specs


def launch_all(max_workers=6, force=False, exp_dir=None, episodes=None,
               experiments=None):
    """Fan out one subprocess per (arm, seed) from the config's seed list.
    Pairs already recorded in the manifest are skipped unless force=True —
    including the shared baseline, so several experiments reuse one baseline
    training. experiments=[N, ...] (or a single int) swaps the legacy fhr arm
    for one exp<N>-tagged arm per requested entry of the config's
    experiment.fhr_experiments. Returns the manifest dict; raises if any run
    fails (successes are kept, so a relaunch resumes from the failures)."""
    exp_dir = pathlib.Path(exp_dir or pathlib.Path.cwd()).resolve()
    if isinstance(experiments, int):
        experiments = [experiments]
    seeds = _config_seeds(exp_dir)
    manifest = _load_manifest(exp_dir)
    manifest["seeds"] = seeds
    specs = _arm_specs(exp_dir, experiments)
    # each arm's effective agent overrides, keyed like runs — the rebuild path
    # (load_runs -> record_final_videos) re-applies them on the config copy
    manifest["overrides"] = {**manifest.get("overrides", {}),
                             **{k: ov for k, _, ov in specs if ov is not None}}
    mpath = MANIFEST
    # force never touches the shared baseline in an --experiment launch: the
    # point of forcing exp<N> is a changed FHR set, and silently repointing
    # the baseline would swap the reference under every OTHER experiment.
    # Retrain the baseline deliberately with --force and no --experiment.
    jobs = [(key, extra, s) for key, extra, _ in specs for s in seeds
            if (force and (key != "baseline" or not experiments))
            or not _run_recorded(exp_dir, manifest, key, s)]
    skipped = len(specs) * len(seeds) - len(jobs)
    if skipped:
        print(f"skipping {skipped} already-completed run(s) recorded in {mpath}")
    if not jobs:
        return manifest

    (exp_dir / LOGS).mkdir(parents=True, exist_ok=True)
    child_env = dict(os.environ, MPLBACKEND="Agg", TQDM_DISABLE="1",
                     OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                     OPENBLAS_NUM_THREADS="1")
    lock = threading.Lock()
    failures = []

    def work(job):
        key, extra, seed = job
        log_path = exp_dir / LOGS / f"fhrdqn_{key}_seed{seed}.log"
        cmd = [sys.executable, str(pathlib.Path(__file__).resolve()),
               *extra, "--seed", str(seed)]
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
            print(f"[{key} seed {seed}] FAILED after {mins:.1f} min — see {log_path}")
            with lock:
                failures.append((key, seed, str(log_path)))
            return
        # a relative RUN_DIR is relative to the child's cwd (exp_dir), not the
        # launcher's — resolve it there before recording it in the manifest
        rel = os.path.relpath(os.path.join(exp_dir, run_dir), exp_dir)
        with lock:
            manifest["runs"].setdefault(key, {})[str(seed)] = rel
            with open(exp_dir / mpath, "w") as f:
                json.dump(manifest, f, indent=2)
        print(f"[{key} seed {seed}] done in {mins:.1f} min -> {rel}")

    print(f"launching {len(jobs)} run(s), {min(max_workers, len(jobs))} at a "
          f"time: {[(k, s) for k, _, s in jobs]}")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(work, jobs))
    if failures:
        raise RuntimeError(f"{len(failures)} run(s) failed (completed runs are "
                           f"kept in {mpath}; relaunch to retry): {failures}")
    return manifest


def load_runs(arm, exp_dir=None):
    """Manifest -> [{seed, cfg, rewards, steps, run_dir}] in config-seed order,
    the structure the comparison notebooks' aggregation cells expect. arm is
    "baseline", the legacy "fhr", or a numbered experiment's "exp<N>". steps is
    the per-episode env-step count column of rewards.csv, or None on runs that
    predate it (the notebooks fall back to |reward| there — exact for the
    ±1-reward-per-step classic-control envs). cfg is the config.yaml copied
    into the run dir (note: the copy reflects neither the baseline's
    fhr_weight=0 override nor an experiment's FHR overrides)."""
    exp_dir = pathlib.Path(exp_dir or pathlib.Path.cwd()).resolve()
    manifest = _load_manifest(exp_dir)
    if not manifest["seeds"]:
        raise RuntimeError(f"no runs recorded in {MANIFEST} under {exp_dir} — "
                           f"launch_all() first")
    runs = []
    for seed in manifest["seeds"]:
        rel = manifest["runs"].get(arm, {}).get(str(seed))
        if rel is None or not (exp_dir / rel / "rewards.csv").exists():
            raise RuntimeError(f"no completed {arm} run for seed {seed} in "
                               f"{MANIFEST} — launch_all() it first "
                               f"({LOGS}/ has any failures)")
        run_dir = exp_dir / rel
        table = np.loadtxt(run_dir / "rewards.csv", delimiter=",", skiprows=1,
                           ndmin=2)
        with open(run_dir / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        runs.append({"seed": seed, "cfg": cfg, "rewards": table[:, 1],
                     "steps": table[:, 2] if table.shape[1] > 2 else None,
                     # the FHR params this arm actually trained with, on top of
                     # cfg["agent"] (None for the legacy fhr arm / old manifests)
                     "agent_overrides": manifest.get("overrides", {}).get(arm),
                     "run_dir": run_dir})
    return runs


def _load_run_agent(run_dir, device="cpu", agent_overrides=None):
    """Rebuild the trained agent from a run dir (config copy + final
    checkpoint) with an rgb_array eval env wrapped exactly like training —
    the normalise/clip wrappers are load-bearing (e.g. MountainCar).
    agent_overrides is the arm's manifest-recorded FHR set: the config copy
    does not carry it, and without it a checkpoint from an arm that changed
    fhr_order / reward_lags cannot load (coefficient shapes mismatch)."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from experiment import build_env, build_agent
    from agents.fhrdqn_agent import FHRDQNAgent
    run_dir = pathlib.Path(run_dir)
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    if agent_overrides:
        cfg["agent"].update(agent_overrides)
    cfg["experiment"]["_device"] = device
    env = build_env(cfg, render_mode="rgb_array")
    nn_extra_kwargs = {"in_dim": env.observation_space.shape[0],
                       "out_dim": env.action_space.n,
                       "hidden_sizes": cfg["network"]["hidden_sizes"]}
    agent = build_agent(cfg, env, QNetwork, nn_extra_kwargs, agent_cls=FHRDQNAgent)
    agent.load(run_dir / "checkpoints" / "final.pt")
    return agent, env


def record_final_videos(arm, exp_dir=None):
    """One greedy-policy episode per manifest run of `arm` ("baseline", "fhr",
    or "exp<N>"), saved as <run_dir>/videos/epfinal-episode-0.mp4 ->
    [(seed, mp4 path)]. The rollout is reset with the run's own seed."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from analysis.visualisations.rollout_video import record_greedy_episode
    out = []
    for r in load_runs(arm, exp_dir):
        agent, env = _load_run_agent(r["run_dir"],
                                     agent_overrides=r["agent_overrides"])
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
                        help="rerun (arm, seed) pairs already in the manifest; "
                             "with --experiment the shared baseline is kept — "
                             "use --force without --experiment to retrain it")
    parser.add_argument("--experiment", type=int, nargs="+", default=None,
                        help="launcher mode: numbered experiment(s) from the "
                             "config's experiment.fhr_experiments — trains the "
                             "shared baseline plus one exp<N> arm per number")
    parser.add_argument("--agent-overrides", default=None,
                        help="child mode: JSON dict of FHR params applied on "
                             "top of the config's agent block")
    parser.add_argument("--name-tag", default=None,
                        help="child mode: run-name tag placed before the seed "
                             "token, e.g. exp2")
    args = parser.parse_args()
    if args.arm is not None:
        if args.seed is None:
            parser.error("--arm requires --seed")
        overrides = json.loads(args.agent_overrides) if args.agent_overrides else None
        run_dir = run_one(args.arm, args.seed, episodes=args.episodes,
                          agent_overrides=overrides, name_tag=args.name_tag)
        print(f"RUN_DIR={run_dir}")
    else:
        launch_all(max_workers=args.max_workers, force=args.force,
                   episodes=args.episodes, experiments=args.experiment)


if __name__ == "__main__":
    main()
