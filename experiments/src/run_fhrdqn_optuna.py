"""Optuna hyperparameter study for the FHR block — shared across experiments.

Like run_fhrdqn_seeds.py, run it with the experiment directory as cwd. All
per-env settings live in <experiment>/configs/study_fhrdqn.yaml (study name,
the fixed Q-net for the search, seeds, trial budget, search space, transfer
grid) — to set up a study for a new env, drop a study_fhrdqn.yaml next to its
config_fhrdqn.yaml in configs/ and run:

    cd experiments/classical_control/<env>
    python ../../src/run_fhrdqn_optuna.py              # stage 1: TPE search
    python ../../src/run_fhrdqn_optuna.py --transfer   # stage 2: best params vs
                                                       #   lambda=0 across net sizes
    python ../../src/run_fhrdqn_optuna.py --status     # progress / top trials

Stage 1 trains one subprocess per (trial, seed) — n_jobs trials in flight x
len(seeds) = concurrent trainings (default 3 x 2 = 6, which saturates the GPU
at these net sizes). Every trial scores the seed-mean of the best rolling-50
episode reward on the SAME seeds (common random numbers). A lambda=0 baseline
is run once per study and stored on it; `enqueue` points from the yaml are
evaluated first. The study lives in <out-dir>/study.db (sqlite): re-running
resumes it — stale RUNNING trials from a killed driver are marked FAILED on
open, and failed trials don't count against --n-trials.

Stage 2 reruns best-vs-baseline across transfer.sizes x transfer.seeds (put a
held-out seed there) at the config's full episode budget, writes
<out-dir>/transfer_results.json and prints a per-size delta table.

Children train with analysis and run-dir artifacts OFF; per-run rewards go to
<out-dir>/rewards/<tag>.npz, per-run logs to <out-dir>/logs/<tag>.log.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent   # experiments/src
SRC = SCRIPTS_DIR.parents[1] / "src"                     # repo library code
CONFIG = "configs/config_fhrdqn.yaml"
STUDY_FILE = "configs/study_fhrdqn.yaml"


def rolling_best(rewards, w=50):
    """Best rolling-w mean (the comparison notebooks' summary metric)."""
    x = np.asarray(rewards, dtype=float)
    if len(x) < w:
        return float(x.mean())
    return float(np.convolve(x, np.ones(w) / w, mode="valid").max())


# ---------------------------------------------------------------- child mode
def run_child(seed, overrides, tag, out_dir, episodes=None):
    """One full training in this process (cwd = experiment dir): overrides
    applied on top of config_fhrdqn.yaml, analysis + artifacts disabled.
    Prints METRIC=<float> as the last line."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    from experiment import load_config, build_env, build_agent, train
    from agents.fhrdqn_agent import FHRDQNAgent
    from run_fhrdqn_seeds import QNetwork      # same MLP as the comparison runs

    cfg = load_config(CONFIG, seed=seed)       # seeds torch / numpy / random
    for k, v in overrides.items():
        if k == "hidden_sizes":
            cfg["network"]["hidden_sizes"] = list(v)
        elif k in cfg["agent"]:
            cfg["agent"][k] = v
        else:
            raise ValueError(f"unknown override {k}")
    if episodes is not None:
        cfg["training"]["no_episodes"] = episodes
    cfg["analysis"] = {"ep_freq": 10**9, "methods": []}   # study runs stay lean

    env = build_env(cfg)
    nn_extra_kwargs = {"in_dim": env.observation_space.shape[0],
                       "out_dim": env.action_space.n,
                       "hidden_sizes": cfg["network"]["hidden_sizes"]}
    agent = build_agent(cfg, env, QNetwork, nn_extra_kwargs, agent_cls=FHRDQNAgent)
    rewards = train(cfg, agent, env, run_logger=None)

    out_dir = pathlib.Path(out_dir)
    (out_dir / "rewards").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "rewards" / f"{tag}.npz",
                        rewards=np.asarray(rewards, dtype=float),
                        seed=seed, overrides=json.dumps(overrides))
    print(f"METRIC={rolling_best(rewards)}")


# ------------------------------------------------------------- parent helpers
def _child_env():
    return dict(os.environ, MPLBACKEND="Agg", TQDM_DISABLE="1",
                OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1")


def spawn_run(exp_dir, seed, overrides, tag, out_dir, episodes=None):
    """Subprocess wrapper around child mode; returns the metric."""
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "logs" / f"{tag}.log"
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "--child",
           "--seed", str(seed), "--tag", tag, "--out-dir", str(out_dir),
           "--overrides", json.dumps(overrides)]
    if episodes is not None:
        cmd += ["--episodes", str(episodes)]
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, cwd=exp_dir, env=_child_env(),
                              stdout=lf, stderr=subprocess.STDOUT)
    for line in reversed(log_path.read_text().splitlines()):
        if line.startswith("METRIC="):
            return float(line.removeprefix("METRIC="))
    raise RuntimeError(f"run {tag} failed (exit {proc.returncode}) — see {log_path}")


def run_group(exp_dir, jobs, max_workers, out_dir, episodes=None):
    """jobs = [(tag, seed, overrides)]; runs them max_workers at a time."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(spawn_run, exp_dir, seed, ov, tag, out_dir, episodes)
                for tag, seed, ov in jobs]
        return [f.result() for f in futs]


def open_study(out_dir, spec):
    """Journal file storage — sqlite races under n_jobs > 1 ("Cannot tell a
    COMPLETE trial"); the journal backend is optuna's supported concurrent
    file storage. A legacy study.db is migrated into the journal once."""
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    out_dir.mkdir(parents=True, exist_ok=True)
    journal = out_dir / "study.journal"
    legacy = out_dir / "study.db"
    migrate = legacy.exists() and not journal.exists()
    storage = JournalStorage(JournalFileBackend(str(journal)))
    if migrate:
        try:
            optuna.copy_study(from_study_name=spec["study"]["name"],
                              from_storage=f"sqlite:///{legacy}",
                              to_storage=storage)
            print(f"migrated {legacy.name} -> {journal.name}")
        except Exception as e:   # start a fresh journal; sqlite left untouched
            print(f"sqlite -> journal migration skipped: {e}")
    return optuna.create_study(
        study_name=spec["study"]["name"], direction="maximize",
        storage=storage, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=0, constant_liar=True,
            n_startup_trials=spec["study"].get("n_startup_trials", 10)))


def reap_stale_running(study):
    """A killed driver leaves RUNNING trials behind; mark them FAILED so they
    neither block constant_liar forever nor look like live work."""
    from optuna.trial import TrialState
    stale = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
    for t in stale:
        study._storage.set_trial_state_values(t._trial_id, state=TrialState.FAIL)
    if stale:
        print(f"marked {len(stale)} stale RUNNING trial(s) as FAILED")


def suggest_params(trial, space):
    """search_space yaml -> optuna suggestions. A list is categorical; a
    {low, high, [step], [log]} dict is int when both bounds are ints, else
    float."""
    params = {}
    for name, spec in space.items():
        if isinstance(spec, list):
            params[name] = trial.suggest_categorical(name, spec)
        elif isinstance(spec["low"], int) and isinstance(spec["high"], int) \
                and not spec.get("log", False):
            params[name] = trial.suggest_int(name, spec["low"], spec["high"],
                                             step=spec.get("step", 1))
        else:
            params[name] = trial.suggest_float(name, spec["low"], spec["high"],
                                               log=spec.get("log", False))
    return params


# --------------------------------------------------------------- study stage
def run_study(args, spec, exp_dir):
    from optuna.trial import TrialState
    out_dir = pathlib.Path(args.out_dir)
    st = spec["study"]
    seeds = args.seeds or st["seeds"]
    episodes = args.episodes or st.get("episodes")
    hidden = st["hidden_sizes"]
    study = open_study(out_dir, spec)
    reap_stale_running(study)

    if "baseline_metrics" not in study.user_attrs:
        print(f"running lambda=0 baseline on seeds {seeds} ...")
        jobs = [(f"baseline_s{s}", s, {"fhr_weight": 0.0, "hidden_sizes": hidden})
                for s in seeds]
        metrics = run_group(exp_dir, jobs, len(jobs), out_dir, episodes)
        study.set_user_attr("baseline_metrics", metrics)
        study.set_user_attr("baseline_seeds", list(seeds))
    bm = study.user_attrs["baseline_metrics"]
    print(f"lambda=0 baseline ({hidden}): mean {np.mean(bm):.1f} "
          f"per-seed {[round(m, 1) for m in bm]}")

    if not any(t.state == TrialState.COMPLETE for t in study.trials):
        for point in st.get("enqueue", []):
            study.enqueue_trial(dict(point))

    def objective(trial):
        params = suggest_params(trial, spec["search_space"])
        jobs = [(f"t{trial.number}_s{s}", s, {**params, "hidden_sizes": hidden})
                for s in seeds]
        metrics = run_group(exp_dir, jobs, len(jobs), out_dir, episodes)
        for s, m in zip(seeds, metrics):
            trial.set_user_attr(f"metric_seed{s}", m)
        return float(np.mean(metrics))

    # n_jobs trials in flight x len(seeds) children = concurrent trainings.
    # catch: a crashed child fails its trial without stopping the study.
    study.optimize(objective, n_trials=args.n_trials or st["n_trials"],
                   n_jobs=args.n_jobs or st.get("n_jobs", 3),
                   catch=(RuntimeError,))
    print_status(args, spec)


# ------------------------------------------------------------ transfer stage
def run_transfer(args, spec, exp_dir):
    out_dir = pathlib.Path(args.out_dir)
    tr = spec["transfer"]
    seeds = args.seeds or tr["seeds"]
    study = open_study(out_dir, spec)
    best = study.best_trial
    print(f"best trial #{best.number}: value {best.value:.1f} params {best.params}")

    jobs, index = [], []
    for size in tr["sizes"]:
        for arm, ov in (("best", dict(best.params)), ("baseline", {"fhr_weight": 0.0})):
            for s in seeds:
                tag = f"transfer_{arm}_h{'x'.join(map(str, size))}_s{s}"
                jobs.append((tag, s, {**ov, "hidden_sizes": size}))
                index.append((str(size), arm, s))
    print(f"transfer: {len(jobs)} runs ({len(tr['sizes'])} sizes x 2 arms x "
          f"{len(seeds)} seeds), {tr.get('max_workers', 6)} at a time")
    # full config episode budget here unless --episodes is given explicitly
    metrics = run_group(exp_dir, jobs, tr.get("max_workers", 6), out_dir,
                        args.episodes)

    results = {}
    for (size, arm, s), m in zip(index, metrics):
        results.setdefault(size, {}).setdefault(arm, {})[s] = m
    with open(out_dir / "transfer_results.json", "w") as f:
        json.dump({"best_params": best.params, "seeds": seeds,
                   "results": results}, f, indent=2)

    print(f"\n{'hidden':>12} | {'baseline':>28} | {'best FHR':>28} | delta")
    for size in tr["sizes"]:
        r = results[str(size)]
        mb = np.mean(list(r["baseline"].values()))
        mf = np.mean(list(r["best"].values()))
        print(f"{'x'.join(map(str, size)):>12} "
              f"| {mb:>7.1f} {str([round(v, 1) for v in r['baseline'].values()]):>20} "
              f"| {mf:>7.1f} {str([round(v, 1) for v in r['best'].values()]):>20} "
              f"| {mf - mb:+.1f}")
    print(f"\nsaved {out_dir / 'transfer_results.json'}")


# -------------------------------------------------------------------- status
def print_status(args, spec):
    out_dir = pathlib.Path(args.out_dir)
    study = open_study(out_dir, spec)
    by_state = {}
    for t in study.trials:
        by_state[t.state.name] = by_state.get(t.state.name, 0) + 1
    done = [t for t in study.trials if t.value is not None]
    print(f"study '{spec['study']['name']}': {by_state}")
    if "baseline_metrics" in study.user_attrs:
        bm = study.user_attrs["baseline_metrics"]
        print(f"lambda=0 baseline ({spec['study']['hidden_sizes']}): "
              f"mean {np.mean(bm):.1f} per-seed {[round(m, 1) for m in bm]}")
    for t in sorted(done, key=lambda t: t.value, reverse=True)[:5]:
        print(f"  #{t.number:>3} value {t.value:>7.1f}  {t.params}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--child", action="store_true", help="internal: one training run")
    parser.add_argument("--seed", type=int, help="child mode: seed")
    parser.add_argument("--tag", help="child mode: rewards/log filename tag")
    parser.add_argument("--overrides", default="{}", help="child mode: JSON overrides")
    parser.add_argument("--transfer", action="store_true",
                        help="stage 2: best params vs lambda=0 across net sizes")
    parser.add_argument("--status", action="store_true", help="progress / top trials")
    parser.add_argument("--n-trials", type=int, default=None, help="override yaml n_trials")
    parser.add_argument("--n-jobs", type=int, default=None, help="override yaml n_jobs")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="override yaml seeds")
    parser.add_argument("--episodes", type=int, default=None,
                        help="override episode budget (smoke tests)")
    parser.add_argument("--out-dir", default=None,
                        help="default <cwd>/optuna_fhrdqn")
    args = parser.parse_args()

    exp_dir = pathlib.Path.cwd().resolve()
    if not (exp_dir / CONFIG).exists():
        sys.exit(f"run from an experiment dir containing {CONFIG} (cwd: {exp_dir})")
    args.out_dir = args.out_dir or str(exp_dir / "optuna_fhrdqn")

    if args.child:
        run_child(args.seed, json.loads(args.overrides), args.tag,
                  args.out_dir, args.episodes)
        return

    study_file = exp_dir / STUDY_FILE
    if not study_file.exists():
        sys.exit(f"no {STUDY_FILE} in {exp_dir} — create one (see the docstring "
                 f"and classical_control/dqn_acrobot_2_revised/{STUDY_FILE} "
                 f"for the schema)")
    with open(study_file) as f:
        spec = yaml.safe_load(f)

    if args.status:
        print_status(args, spec)
    elif args.transfer:
        run_transfer(args, spec, exp_dir)
    else:
        run_study(args, spec, exp_dir)


if __name__ == "__main__":
    main()
