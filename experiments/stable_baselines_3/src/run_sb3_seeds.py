"""Parallel multi-seed launcher for the SB3 (Stable-Baselines3) FHR comparison.

The SB3 twin of run_fhrdqn_seeds.py: one (arm, seed) training per subprocess,
run with the experiment directory as cwd (config paths and cached/ resolved
against cwd):

    cd experiments/stable_baselines_3/cartpole
    python ../src/run_sb3_seeds.py                    # baseline + fhr arms
    python ../src/run_sb3_seeds.py --experiment 1 2   # baseline + exp1 + exp2
    python ../src/run_sb3_seeds.py --max-workers 3 --force
    python ../src/run_sb3_seeds.py --config configs/config_sb3_zoo.yaml \
        --arms baseline                               # alt recipe, own manifest

The trained method is RL-Zoo-tuned stock SB3 (algo.type: qrdqn via sb3_contrib
— SB3's best sample-efficient DQN-family method — or dqn), and every arm runs
the same FHRDQN/FHRQRDQN code path from src/agents/sb3_fhr.py: the baseline
arm forces fhr_weight 0.0, which is bit-for-bit the stock SB3 algorithm (same
RNG stream, same updates — asserted by tests/test_sb3_fhr.py), so a
exp<N>-vs-baseline gap is attributable to the FHR penalty alone.

Numbered experiments, manifest (cached/sb3_runs_manifest.json with keys seeds/
runs/overrides), per-child logs (cached/logs/sb3_<key>_seed<N>.log), skip/
--force semantics, and the launch_all()/load_runs()/record_final_videos()
notebook API all match run_fhrdqn_seeds.py. Run dirs land under cached/runs/
with the rewards.csv (episode,reward,steps), rank_stats.csv, hankel_sweep.csv,
train_diagnostics.csv and figures/ contract the result viewer app reads —
driven during training by FHRSB3Callback + training.run_analysis_tick, the
same analysis dispatch the classic dqn_training_loop uses.

Child mode (one training in this process; used internally by the launcher):

    python ../src/run_sb3_seeds.py --arm fhr --seed 44 \
        --agent-overrides '{"fhr_weight": 0.1, "fhr_order": 2}' --name-tag exp1
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
import yaml

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent   # experiments/stable_baselines_3/src
SRC = SCRIPTS_DIR.parents[2] / "src"                     # repo library code
CONFIG = "configs/config_sb3.yaml"
MANIFEST = "cached/sb3_runs_manifest.json"
LOGS = "cached/logs"
ARMS = ("baseline", "fhr")


def _config_suffix(config):
    """'' for the default config; 'zoo' for configs/config_sb3_zoo.yaml, etc.
    Keys the per-config manifest and log names so two configs sharing one
    experiment dir (e.g. tuned recipe vs stock-zoo reference) never collide."""
    stem = pathlib.Path(config).stem
    return stem.removeprefix("config_sb3").lstrip("_")


def _manifest_name(config):
    sfx = _config_suffix(config)
    return f"cached/sb3_runs_manifest{'_' + sfx if sfx else ''}.json"

# The FHR parameter set a numbered experiment may override — everything else
# (algo hyperparameters, net size, exploration schedule, ...) stays identical
# to the baseline so an exp<N>-vs-baseline gap is attributable to the FHR
# penalty alone. Mirrors agents.sb3_fhr.FHR_PARAMS.
FHR_PARAMS = ("fhr_weight", "fhr_order", "reward_lags",
              "warmup_grad_steps", "c_learning_rate", "c_predictor",
              "prioritized_replay", "per_alpha", "per_beta0",
              "rampdown_reward_threshold", "rampdown_penalty_threshold",
              "rampdown_penalty_topk", "rampdown_patience_eps",
              "rampdown_episodes")


def _algo_class(algo_type):
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from agents.sb3_fhr import FHRDQN, FHRQRDQN
    from agents.sb3_sac_fhr import FHRSAC, FHRSACD
    try:
        return {"dqn": FHRDQN, "qrdqn": FHRQRDQN,
                "sac": FHRSAC, "sacd": FHRSACD}[algo_type]
    except KeyError:
        raise ValueError("algo.type must be one of dqn|qrdqn|sac|sacd, "
                         f"got {algo_type!r}")


def _build_model(cfg, env, seed):
    """The SB3 model from the config's algo block (RL-Zoo hyperparameter
    names) + the agent block's FHR parameters."""
    algo = dict(cfg["algo"])
    algo_type = algo.pop("type")
    cls = _algo_class(algo_type)
    algo.pop("n_timesteps")
    policy_kwargs = {"net_arch": list(algo.pop("net_arch"))}
    # n_quantiles is meaningful to QR-DQN only; a config switched to
    # algo.type: dqn may (deliberately) keep the key around — never forward
    # it to DQNPolicy, which rejects unknown kwargs
    n_quantiles = algo.pop("n_quantiles", None)
    if algo_type == "qrdqn" and n_quantiles is not None:
        policy_kwargs["n_quantiles"] = n_quantiles
    # SAC-family-only keys: n_critics shapes the policy; target_entropy_scale
    # exists on SACD alone. Popped unconditionally so a config switched
    # between types can keep them around without crashing the constructor.
    n_critics = algo.pop("n_critics", None)
    if algo_type in ("sac", "sacd") and n_critics is not None:
        policy_kwargs["n_critics"] = int(n_critics)
    tes = algo.pop("target_entropy_scale", None)
    if algo_type == "sacd" and tes is not None:
        algo["target_entropy_scale"] = float(tes)
    fhr = {k: cfg["agent"][k] for k in FHR_PARAMS if k in cfg["agent"]}
    return cls("MlpPolicy", env, policy_kwargs=policy_kwargs, seed=seed,
               device=cfg["experiment"]["_device"], verbose=0, **algo, **fhr)


def _make_env(cfg, render_mode=None):
    """gym.make + the config's static observation wrappers, when present.

    environment.flatten_obs flattens a Dict observation space (the
    goal-conditioned robotics envs: {observation, achieved_goal,
    desired_goal} -> one Box) so MlpPolicy and the FHR episodic buffer work
    unchanged. environment.normalise.state {min, max} applies gymnasium's
    RescaleObservation (the same wrapper the classic base_env stack uses) — a
    fixed affine map, so unlike VecNormalize it is identical for the TD batch
    and the FHR lag observations. Must wrap every env a run touches (training,
    eval, analysis, videos) so the policy always sees one observation space.
    Robotics env ids (Fetch*, ...) register lazily on first use.
    """
    import gymnasium as gym
    name = cfg["environment"]["name"]
    try:
        env = gym.make(name, render_mode=render_mode)
    except (gym.error.NameNotFound, gym.error.NamespaceNotFound):
        import gymnasium_robotics                  # optional extra
        gym.register_envs(gymnasium_robotics)
        env = gym.make(name, render_mode=render_mode)
    if cfg["environment"].get("flatten_obs"):
        env = gym.wrappers.FlattenObservation(env)
    state = ((cfg["environment"].get("normalise") or {}).get("state") or {})
    if state:
        env = gym.wrappers.RescaleObservation(
            env, np.array(state["min"], dtype=np.float32),
            np.array(state["max"], dtype=np.float32))
    return env


def _make_analysis_env(cfg, render_mode=None):
    """A dedicated env for the analysis rollouts (the training env must not be
    disturbed mid-episode), with finite observation bounds for q_matrix_dqn's
    state grid when the config provides analysis.state_bounds."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from agents.sb3_fhr import BoundedObservations
    env = _make_env(cfg, render_mode=render_mode)
    bounds = (cfg.get("analysis") or {}).get("state_bounds")
    if bounds:
        env = BoundedObservations(env, bounds["low"], bounds["high"])
    return env


def run_one(arm, seed, timesteps=None, agent_overrides=None, name_tag=None,
            config=CONFIG):
    """Child mode: one full training run in this process (cwd = experiment
    dir). Baseline strips the FHR term (fhr_weight = 0.0); a numbered arm
    applies its FHR parameter set on top of the config's agent block. Returns
    the run directory."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import gymnasium as gym
    from stable_baselines3.common.monitor import Monitor
    from experiment import load_config, make_run_logger
    from agents.sb3_fhr import FHRSB3Callback, GreedyEvalCallback

    cfg = load_config(config, seed=seed)          # seeds torch / numpy / random
    if arm == "baseline":
        cfg["agent"]["fhr_weight"] = 0.0
    elif agent_overrides:
        cfg["agent"].update(agent_overrides)
    if name_tag:
        cfg["experiment"]["name"] += f"_{name_tag}"
    elif arm == "baseline":
        cfg["experiment"]["name"] += "_baseline"  # shared-baseline naming
    cfg["experiment"]["name"] += f"_seed{seed}"
    if timesteps is not None:                     # smoke-test override
        cfg["algo"]["n_timesteps"] = timesteps

    env = Monitor(_make_env(cfg))
    model = _build_model(cfg, env, seed)
    logger = make_run_logger(cfg, config_path=config, base_dir="cached")
    analysis_env = _make_analysis_env(cfg)
    callback = FHRSB3Callback(run_logger=logger,
                              analysis_config=cfg.get("analysis"),
                              analysis_env=analysis_env,
                              training_config=cfg.get("training"))
    callbacks = [callback]
    eval_cfg = (cfg.get("training") or {}).get("eval")
    eval_env = None
    if eval_cfg:
        # greedy-policy eval curve -> <run_dir>/eval.csv; fixed per-episode
        # reset seeds keep the curve paired across arms/variants, and the
        # deterministic policy draws no global RNG (training stream untouched)
        eval_env = _make_env(cfg)
        callbacks.append(GreedyEvalCallback(
            eval_env, logger.dir,
            freq_steps=eval_cfg.get("freq_steps", 5000),
            n_episodes=eval_cfg.get("n_episodes", 10),
            seed=eval_cfg.get("seed", 9000)))
    model.learn(total_timesteps=int(cfg["algo"]["n_timesteps"]),
                callback=callbacks)
    if eval_env is not None:
        eval_env.close()
    analysis_env.close()
    env.close()
    return logger.dir


def _config_seeds(exp_dir, config=CONFIG):
    with open(exp_dir / config) as f:
        cfg = yaml.safe_load(f)
    return list(cfg["experiment"].get("seeds") or [cfg["experiment"]["seed"]])


def _load_manifest(exp_dir, config=CONFIG):
    path = exp_dir / _manifest_name(config)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"seeds": [], "runs": {}}


def _run_recorded(exp_dir, manifest, arm, seed):
    rel = manifest["runs"].get(arm, {}).get(str(seed))
    return rel is not None and (exp_dir / rel / "rewards.csv").exists()


def _experiment_overrides(exp_dir, experiment, config=CONFIG):
    with open(exp_dir / config) as f:
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
            f"params, e.g. {{fhr_weight: 0.1, fhr_order: 2}} — got {overrides!r}")
    unknown = sorted(set(overrides) - set(FHR_PARAMS))
    if unknown:
        raise ValueError(f"experiment {experiment} overrides unknown FHR "
                         f"params {unknown} — allowed: {list(FHR_PARAMS)}")
    if float(overrides.get("fhr_weight", 1.0)) == 0.0:
        raise ValueError("fhr_weight 0 is the baseline arm, which every "
                         "launch already includes — drop it from the sweep")
    return overrides


def _arm_specs(exp_dir, experiments, config=CONFIG):
    baseline = ("baseline", ["--arm", "baseline"], {"fhr_weight": 0.0})
    if not experiments:
        return [baseline, ("fhr", ["--arm", "fhr"], None)]
    if len(set(experiments)) != len(experiments):
        raise ValueError(f"duplicate experiment numbers: {experiments}")
    specs = [baseline]
    for n in experiments:
        overrides = _experiment_overrides(exp_dir, n, config)
        specs.append((f"exp{n}", ["--arm", "fhr",
                                  "--agent-overrides", json.dumps(overrides),
                                  "--name-tag", f"exp{n}"], overrides))
    return specs


def launch_all(max_workers=6, force=False, exp_dir=None, timesteps=None,
               experiments=None, config=CONFIG, arms=None):
    """Fan out one subprocess per (arm, seed) from the config's seed list;
    pairs already recorded in the manifest are skipped unless force=True
    (force never touches the shared baseline in an --experiment launch).
    Returns the manifest dict; raises if any run fails (successes are kept,
    so a relaunch resumes from the failures).

    config selects an alternative config file in the experiment dir (e.g.
    "configs/config_sb3_zoo.yaml"); its runs are recorded in a per-config
    manifest (cached/sb3_runs_manifest_<suffix>.json) so recipes sharing the
    dir never collide. arms optionally restricts the launch to a subset of
    arm keys, e.g. arms=["baseline"] for a reference-baseline-only launch
    (keys: "baseline", "fhr", "exp<N>")."""
    exp_dir = pathlib.Path(exp_dir or pathlib.Path.cwd()).resolve()
    if isinstance(experiments, int):
        experiments = [experiments]
    manifest_name = _manifest_name(config)
    seeds = _config_seeds(exp_dir, config)
    manifest = _load_manifest(exp_dir, config)
    manifest["seeds"] = seeds
    specs = _arm_specs(exp_dir, experiments, config)
    if arms is not None:
        unknown = sorted(set(arms) - {k for k, _, _ in specs})
        if unknown:
            raise ValueError(f"arms {unknown} not in this launch's arm set "
                             f"{[k for k, _, _ in specs]}")
        specs = [s for s in specs if s[0] in arms]
    manifest["overrides"] = {**manifest.get("overrides", {}),
                             **{k: ov for k, _, ov in specs if ov is not None}}
    jobs = [(key, extra, s) for key, extra, _ in specs for s in seeds
            if (force and (key != "baseline" or not experiments))
            or not _run_recorded(exp_dir, manifest, key, s)]
    skipped = len(specs) * len(seeds) - len(jobs)
    if skipped:
        print(f"skipping {skipped} already-completed run(s) recorded in {manifest_name}")
    if not jobs:
        return manifest

    (exp_dir / LOGS).mkdir(parents=True, exist_ok=True)
    child_env = dict(os.environ, MPLBACKEND="Agg", TQDM_DISABLE="1",
                     OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                     OPENBLAS_NUM_THREADS="1")
    lock = threading.Lock()
    failures = []

    sfx = _config_suffix(config)

    def work(job):
        key, extra, seed = job
        log_path = exp_dir / LOGS / f"sb3_{sfx + '_' if sfx else ''}{key}_seed{seed}.log"
        cmd = [sys.executable, str(pathlib.Path(__file__).resolve()),
               *extra, "--seed", str(seed), "--config", config]
        if timesteps is not None:
            cmd += ["--timesteps", str(timesteps)]
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
        rel = os.path.relpath(os.path.join(exp_dir, run_dir), exp_dir)
        with lock:
            manifest["runs"].setdefault(key, {})[str(seed)] = rel
            with open(exp_dir / manifest_name, "w") as f:
                json.dump(manifest, f, indent=2)
        print(f"[{key} seed {seed}] done in {mins:.1f} min -> {rel}")

    print(f"launching {len(jobs)} run(s), {min(max_workers, len(jobs))} at a "
          f"time: {[(k, s) for k, _, s in jobs]}")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(work, jobs))
    if failures:
        raise RuntimeError(f"{len(failures)} run(s) failed (completed runs are "
                           f"kept in {manifest_name}; relaunch to retry): {failures}")
    return manifest


def load_runs(arm, exp_dir=None, config=CONFIG):
    """Manifest -> [{seed, cfg, rewards, steps, agent_overrides, run_dir}] in
    config-seed order — the structure the comparison notebooks expect, same as
    run_fhrdqn_seeds.load_runs. config selects the per-config manifest, as in
    launch_all."""
    exp_dir = pathlib.Path(exp_dir or pathlib.Path.cwd()).resolve()
    manifest_name = _manifest_name(config)
    manifest = _load_manifest(exp_dir, config)
    if not manifest["seeds"]:
        raise RuntimeError(f"no runs recorded in {manifest_name} under {exp_dir} — "
                           f"launch_all() first")
    runs = []
    for seed in manifest["seeds"]:
        rel = manifest["runs"].get(arm, {}).get(str(seed))
        if rel is None or not (exp_dir / rel / "rewards.csv").exists():
            raise RuntimeError(f"no completed {arm} run for seed {seed} in "
                               f"{manifest_name} — launch_all() it first "
                               f"({LOGS}/ has any failures)")
        run_dir = exp_dir / rel
        table = np.loadtxt(run_dir / "rewards.csv", delimiter=",", skiprows=1,
                           ndmin=2)
        with open(run_dir / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        runs.append({"seed": seed, "cfg": cfg, "rewards": table[:, 1],
                     "steps": table[:, 2] if table.shape[1] > 2 else None,
                     "agent_overrides": manifest.get("overrides", {}).get(arm),
                     "run_dir": run_dir})
    return runs


def load_run_model(run_dir, device="cpu", checkpoint="final"):
    """Rebuild the trained SB3 model + a QAgent-surface adapter from a run
    dir. Unlike the classic _load_run_agent, no manifest overrides are needed:
    the SB3 checkpoint zip is self-describing (FHR config and coefficients
    included) — only the algo class comes from the run's config.yaml copy.
    Returns (model, adapter)."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from agents.sb3_fhr import SB3QAgentAdapter
    run_dir = pathlib.Path(run_dir)
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cls = _algo_class(cfg["algo"]["type"])
    model = cls.load(run_dir / "checkpoints" / f"{checkpoint}.pt", device=device)
    make = getattr(model, "qagent_adapter", None)   # SAC family
    adapter = make(epsilon=0.0) if make else SB3QAgentAdapter(model, epsilon=0.0)
    return model, adapter


def record_final_videos(arm, exp_dir=None, config=CONFIG):
    """One greedy-policy episode per manifest run of `arm`, saved as
    <run_dir>/videos/epfinal-episode-0.mp4 -> [(seed, mp4 path)]."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from analysis.visualisations.rollout_video import record_greedy_episode
    out = []
    for r in load_runs(arm, exp_dir, config=config):
        model, adapter = load_run_model(r["run_dir"])
        env = _make_env(r["cfg"], render_mode="rgb_array")
        prefix = record_greedy_episode(adapter, env, str(r["run_dir"] / "videos"),
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
    parser.add_argument("--timesteps", type=int, default=None,
                        help="override algo.n_timesteps (smoke tests)")
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
                             "token, e.g. exp1")
    parser.add_argument("--config", default=CONFIG,
                        help="config file relative to the experiment dir "
                             f"(default {CONFIG}); alternative configs get "
                             "their own manifest, e.g. config_sb3_zoo.yaml -> "
                             "cached/sb3_runs_manifest_zoo.json")
    parser.add_argument("--arms", nargs="+", default=None,
                        help="launcher mode: restrict to these arm keys, e.g. "
                             "--arms baseline")
    args = parser.parse_args()
    if args.arm is not None:
        if args.seed is None:
            parser.error("--arm requires --seed")
        overrides = json.loads(args.agent_overrides) if args.agent_overrides else None
        run_dir = run_one(args.arm, args.seed, timesteps=args.timesteps,
                          agent_overrides=overrides, name_tag=args.name_tag,
                          config=args.config)
        print(f"RUN_DIR={run_dir}")
    else:
        launch_all(max_workers=args.max_workers, force=args.force,
                   timesteps=args.timesteps, experiments=args.experiment,
                   config=args.config, arms=args.arms)


if __name__ == "__main__":
    main()
