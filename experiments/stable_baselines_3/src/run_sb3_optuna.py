"""Multi-objective Optuna study for the SB3 FHR block — rho and order only.

The SB3 twin of experiments/src/run_fhrdqn_optuna.py, restricted to the two
FHR knobs the tuning studies left open: the penalty weight (searched as
rho = lambda * penalty_raw / td_loss, the weighted-penalty/TD contribution
ratio from calibrate_fhr.py) and the recurrence order. Everything else —
recipe, c_learning_rate, reward_lags: false — is pinned by the env's
configs/config_sb3_study.yaml (the tuned config_sb3.yaml recipe with the
study seed list, wrank probes on, frequent Hankel sweeps and early stopping
disabled so every trial trains the full budget).

Run from the experiment directory:

    cd experiments/stable_baselines_3/mountaincar
    python ../src/run_sb3_optuna.py             # calibrate + baseline + TPE
    python ../src/run_sb3_optuna.py --status    # progress / Pareto front

Two objectives per trial, each the mean over the study seeds of the greedy
eval curve (<run_dir>/eval.csv, deterministic policy on fixed reset seeds):

  * SPEED (minimize): mean over objective.thresholds of the first env step
    whose eval mean_reward reaches the threshold; a threshold never reached
    costs 1.2 x the training budget (worse than reaching it on the last tick).
  * FINAL (maximize): mean of the last objective.final_evals eval ticks.

Stage 0 (once per study, stored in study user attrs) trains a PROBE per study
seed: the config's FHR block with an infinite warm-up, which trains
bit-identically to lambda=0 while train_diagnostics.csv logs penalty_raw next
to td_loss. The probes provide both the lambda calibration
(td_over_penalty = tail-median ratio, seed mean; fhr_weight = rho *
td_over_penalty) and the baseline reference for the two objectives.
enqueue points are given as {fhr_weight, fhr_order} and converted to rho with
that same ratio. Each trial then trains len(seeds) children; every child run
dir (under <out-dir>/runs/) carries the full wrank + Hankel-sweep
instrumentation of the study config.

The study lives in <out-dir>/study.journal (concurrency-safe journal
storage); re-running resumes it, stale RUNNING trials from a killed driver
are failed on open, and a crashed child fails its trial without stopping the
study.
"""
import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent   # stable_baselines_3/src
CONFIG = "configs/config_sb3_study.yaml"
STUDY_FILE = "configs/study_sb3_fhr.yaml"
NEVER_HIT_FACTOR = 1.2       # censored time-to-threshold, x training budget
PROBE_WARMUP = 10 ** 9       # infinite warm-up == bit-identical to lambda=0


# ---------------------------------------------------------------- child mode
def run_child(args):
    """One full training in this process (cwd = experiment dir), via
    run_sb3_seeds.run_one against the study config. Prints RUN_DIR=<path> as
    its last line (run_one's contract)."""
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import run_sb3_seeds as runner
    overrides = json.loads(args.overrides) if args.overrides else None
    run_dir = runner.run_one(args.arm, args.seed, timesteps=args.timesteps,
                             agent_overrides=overrides, name_tag=args.tag,
                             config=CONFIG, base_dir=args.base_dir)
    print(f"RUN_DIR={run_dir}")


# ------------------------------------------------------------- parent helpers
def _child_env():
    return dict(os.environ, MPLBACKEND="Agg", TQDM_DISABLE="1",
                OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1")


def spawn_run(exp_dir, out_dir, tag, seed, overrides, arm="fhr",
              timesteps=None):
    """Subprocess wrapper around child mode; returns the run dir (absolute)."""
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "logs" / f"{tag}_s{seed}.log"
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "--child",
           "--arm", arm, "--seed", str(seed), "--tag", tag,
           "--base-dir", str(out_dir),
           "--overrides", json.dumps(overrides or {})]
    if timesteps is not None:
        cmd += ["--timesteps", str(timesteps)]
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, cwd=exp_dir, env=_child_env(),
                              stdout=lf, stderr=subprocess.STDOUT)
    run_dir = None
    for line in reversed(log_path.read_text().splitlines()):
        if line.startswith("RUN_DIR="):
            run_dir = line.removeprefix("RUN_DIR=")
            break
    if proc.returncode != 0 or run_dir is None:
        raise RuntimeError(f"run {tag} seed {seed} failed "
                           f"(exit {proc.returncode}) — see {log_path}")
    run_dir = pathlib.Path(run_dir)
    return run_dir if run_dir.is_absolute() else exp_dir / run_dir


def run_group(exp_dir, out_dir, jobs, timesteps=None):
    """jobs = [(tag, seed, overrides, arm)]; all in flight at once (one trial's
    seeds, or the probe stage). Returns run dirs in job order."""
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futs = [pool.submit(spawn_run, exp_dir, out_dir, tag, seed, ov, arm,
                            timesteps)
                for tag, seed, ov, arm in jobs]
        return [f.result() for f in futs]


# ------------------------------------------------------------------- metrics
def eval_metrics(run_dir, thresholds, final_evals, budget):
    """<run_dir>/eval.csv -> (speed, final): mean env-steps-to-threshold over
    the ladder (censored at NEVER_HIT_FACTOR x budget) and the mean of the
    last final_evals greedy-eval means."""
    rows = list(csv.DictReader(open(pathlib.Path(run_dir) / "eval.csv")))
    steps = np.array([float(r["env_steps"]) for r in rows])
    mr = np.array([float(r["mean_reward"]) for r in rows])
    final = float(np.mean(mr[-final_evals:]))
    tts = []
    for th in thresholds:
        hit = np.nonzero(mr >= th)[0]
        tts.append(float(steps[hit[0]]) if len(hit)
                   else float(budget) * NEVER_HIT_FACTOR)
    return float(np.mean(tts)), final


def _tail_median(rows, col, tail_frac):
    vals = np.array([float(r[col]) for r in rows])
    vals = vals[np.isfinite(vals)]
    tail = vals[int(len(vals) * (1 - tail_frac)):]
    return float(np.median(tail)) if len(tail) else float("nan")


def diag_medians(run_dir, tail_frac=0.5):
    """train_diagnostics.csv tail medians -> {td_loss, penalty_raw,
    penalty_weighted} (the realized-rho / calibration ingredients)."""
    rows = list(csv.DictReader(open(pathlib.Path(run_dir)
                                    / "train_diagnostics.csv")))
    return {c: _tail_median(rows, c, tail_frac)
            for c in ("td_loss", "penalty_raw", "penalty_weighted")}


# --------------------------------------------------------------------- study
def open_study(out_dir, spec):
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(out_dir / "study.journal")))
    return optuna.create_study(
        study_name=spec["study"]["name"],
        directions=["minimize", "maximize"],      # (speed, final)
        storage=storage, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=0, constant_liar=True,
            n_startup_trials=spec["study"].get("n_startup_trials", 8)))


def reap_stale_running(study):
    from optuna.trial import TrialState
    stale = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
    for t in stale:
        study._storage.set_trial_state_values(t._trial_id,
                                              state=TrialState.FAIL)
    if stale:
        print(f"marked {len(stale)} stale RUNNING trial(s) as FAILED")


def _budget(exp_dir, timesteps):
    if timesteps is not None:
        return timesteps
    with open(exp_dir / CONFIG) as f:
        return int(yaml.safe_load(f)["algo"]["n_timesteps"])


def ensure_probes(study, spec, exp_dir, out_dir, seeds, args):
    """Stage 0: one probe per study seed (config FHR block, infinite warm-up
    == bit-identical to lambda=0). Stores the lambda calibration ratio and the
    baseline objectives on the study."""
    if "td_over_penalty" in study.user_attrs:
        return
    obj = spec["objective"]
    budget = _budget(exp_dir, args.timesteps)
    cal = spec["study"].get("calibration", {})
    tail_frac = cal.get("tail_frac", 0.5)
    reuse = cal.get("reuse_probe_dirs")
    if reuse:
        # probes from a superseded study of the same config (bit-identical to
        # lambda=0, so their diagnostics and eval curves carry over), listed
        # in study-seed order
        run_dirs = [exp_dir / r for r in reuse]
        print(f"probe stage: reusing {len(run_dirs)} recorded probe run(s)")
    else:
        print(f"probe stage: {len(seeds)} calibration/baseline run(s) on "
              f"seeds {seeds} ...")
        jobs = [("probe", s, {"warmup_grad_steps": PROBE_WARMUP}, "fhr")
                for s in seeds]
        run_dirs = run_group(exp_dir, out_dir, jobs, args.timesteps)
    ratios, speeds, finals = [], [], []
    for rd in run_dirs:
        med = diag_medians(rd, tail_frac)
        ratios.append(med["td_loss"] / med["penalty_raw"])
        sp, fi = eval_metrics(rd, obj["thresholds"], obj["final_evals"],
                              budget)
        speeds.append(sp)
        finals.append(fi)
    # The probe ratio assumes the baseline-equilibrium penalty magnitude is
    # representative — false where the baseline barely learns (flat Q ->
    # near-zero probe penalty, MountainCar) or where the active penalty
    # collapses penalty_raw by orders of magnitude (Acrobot). calibration.
    # td_over_penalty_override pins the mapping from measured FHR-active runs
    # (or a chosen anchor) instead; the probe ratio is kept for reference.
    override = cal.get("td_over_penalty_override")
    study.set_user_attr("td_over_penalty",
                        float(override) if override is not None
                        else float(np.mean(ratios)))
    study.set_user_attr("td_over_penalty_probe", float(np.mean(ratios)))
    study.set_user_attr("td_over_penalty_per_seed",
                        [float(r) for r in ratios])
    study.set_user_attr("baseline_speed_per_seed", speeds)
    study.set_user_attr("baseline_final_per_seed", finals)
    study.set_user_attr("probe_seeds", list(seeds))
    study.set_user_attr(
        "probe_run_dirs",
        [os.path.relpath(rd, exp_dir) for rd in run_dirs])


def run_study(args, spec, exp_dir):
    from optuna.trial import TrialState
    out_dir = pathlib.Path(args.out_dir)
    st = spec["study"]
    obj = spec["objective"]
    space = spec["search_space"]
    seeds = args.seeds or st["seeds"]
    budget = _budget(exp_dir, args.timesteps)
    study = open_study(out_dir, spec)
    reap_stale_running(study)
    ensure_probes(study, spec, exp_dir, out_dir, seeds, args)

    ratio = study.user_attrs["td_over_penalty"]
    print(f"lambda calibration: fhr_weight = rho * {ratio:.4g}  "
          f"(rho in [{space['rho']['low']:g}, {space['rho']['high']:g}])")
    print(f"baseline (probe): speed {np.mean(study.user_attrs['baseline_speed_per_seed']):.0f} "
          f"final {np.mean(study.user_attrs['baseline_final_per_seed']):.1f}")

    if not any(t.state == TrialState.COMPLETE for t in study.trials):
        for point in st.get("enqueue", []):
            rho = float(np.clip(point["fhr_weight"] / ratio,
                                space["rho"]["low"], space["rho"]["high"]))
            study.enqueue_trial({"rho": rho,
                                 "fhr_order": point["fhr_order"]})
            print(f"enqueued fhr_weight {point['fhr_weight']} "
                  f"order {point['fhr_order']} as rho {rho:.3g}")

    def objective(trial):
        rho = trial.suggest_float("rho", space["rho"]["low"],
                                  space["rho"]["high"], log=True)
        order = trial.suggest_int("fhr_order", space["fhr_order"]["low"],
                                  space["fhr_order"]["high"])
        lam = rho * ratio
        trial.set_user_attr("fhr_weight", lam)
        overrides = {"fhr_weight": lam, "fhr_order": order}
        jobs = [(f"t{trial.number}", s, overrides, "fhr") for s in seeds]
        run_dirs = run_group(exp_dir, out_dir, jobs, args.timesteps)
        speeds, finals, rho_real = [], [], []
        for s, rd in zip(seeds, run_dirs):
            sp, fi = eval_metrics(rd, obj["thresholds"], obj["final_evals"],
                                  budget)
            med = diag_medians(rd)
            speeds.append(sp)
            finals.append(fi)
            rho_real.append(med["penalty_weighted"] / med["td_loss"])
            trial.set_user_attr(f"speed_seed{s}", sp)
            trial.set_user_attr(f"final_seed{s}", fi)
            trial.set_user_attr(f"run_dir_seed{s}",
                                os.path.relpath(rd, exp_dir))
        trial.set_user_attr("realized_rho", float(np.mean(rho_real)))
        return float(np.mean(speeds)), float(np.mean(finals))

    # n_jobs trials in flight x len(seeds) children = concurrent trainings;
    # a crashed child fails its trial without stopping the study.
    study.optimize(objective, n_trials=args.n_trials or st["n_trials"],
                   n_jobs=args.n_jobs or st.get("n_jobs", 1),
                   catch=(RuntimeError,))
    print_status(args, spec)


# -------------------------------------------------------------------- status
def print_status(args, spec):
    out_dir = pathlib.Path(args.out_dir)
    study = open_study(out_dir, spec)
    by_state = {}
    for t in study.trials:
        by_state[t.state.name] = by_state.get(t.state.name, 0) + 1
    print(f"study '{spec['study']['name']}': {by_state}")
    if "td_over_penalty" in study.user_attrs:
        print(f"lambda = rho * {study.user_attrs['td_over_penalty']:.4g}; "
              f"baseline speed "
              f"{np.mean(study.user_attrs['baseline_speed_per_seed']):.0f} "
              f"final "
              f"{np.mean(study.user_attrs['baseline_final_per_seed']):.1f}")
    done = [t for t in study.trials if t.values is not None]
    pareto = {t.number for t in study.best_trials}
    for t in sorted(done, key=lambda t: t.values[1], reverse=True)[:10]:
        star = "*" if t.number in pareto else " "
        print(f" {star}#{t.number:>3} speed {t.values[0]:>8.0f} "
              f"final {t.values[1]:>8.1f}  rho {t.params['rho']:.3g} "
              f"(lambda {t.user_attrs.get('fhr_weight', float('nan')):.3g}, "
              f"realized rho {t.user_attrs.get('realized_rho', float('nan')):.3g}) "
              f"order {t.params['fhr_order']}")
    if pareto:
        print(f"Pareto front (*): {sorted(pareto)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--child", action="store_true",
                        help="internal: one training run")
    parser.add_argument("--arm", default="fhr", choices=("baseline", "fhr"),
                        help="child mode: run_sb3_seeds arm")
    parser.add_argument("--seed", type=int, help="child mode: seed")
    parser.add_argument("--tag", help="child mode: run-name tag")
    parser.add_argument("--overrides", default="",
                        help="child mode: JSON FHR overrides")
    parser.add_argument("--base-dir", default=None,
                        help="child mode: run-dir base (the study out-dir)")
    parser.add_argument("--status", action="store_true",
                        help="progress / Pareto front")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--timesteps", type=int, default=None,
                        help="override algo.n_timesteps (smoke tests)")
    parser.add_argument("--out-dir", default=None,
                        help="default <cwd>/optuna_sb3_fhr")
    args = parser.parse_args()

    exp_dir = pathlib.Path.cwd().resolve()
    if not (exp_dir / CONFIG).exists():
        sys.exit(f"run from an experiment dir containing {CONFIG} "
                 f"(cwd: {exp_dir})")
    args.out_dir = args.out_dir or str(exp_dir / "optuna_sb3_fhr")

    if args.child:
        run_child(args)
        return

    with open(exp_dir / STUDY_FILE) as f:
        spec = yaml.safe_load(f)
    if args.status:
        print_status(args, spec)
    else:
        run_study(args, spec, exp_dir)


if __name__ == "__main__":
    main()
