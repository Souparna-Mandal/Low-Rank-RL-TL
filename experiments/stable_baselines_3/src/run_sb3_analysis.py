"""Post-hoc "run every analysis framework" pass over completed SB3 FHR runs.

The offline twin of FHRSB3Callback's in-training analysis tick: reload a
finished run's checkpoint (checkpoints/final.pt, falling back to latest.pt
with a note) through run_sb3_seeds.load_run_model, attach a RunLogger to the
EXISTING run directory, and drive training.run_analysis_tick with the run's
own config.yaml — Q-matrix rank -> rank_stats.csv + figures, Hankel sweep ->
hankel_sweep.csv + trajectories/, AR value probe -> autoregressive_*.csv +
rollouts. Then N greedy evaluation episodes (fresh env, seeds 30000+i,
mean +/- std printed) and, unless --skip-video, one greedy rollout mp4 under
<run_dir>/videos/.

Run with the experiment directory as cwd (all paths, like run_sb3_seeds.py,
resolve against cwd):

    cd experiments/stable_baselines_3/cartpole
    python ../src/run_sb3_analysis.py --run-dir cached/runs/<name>
    python ../src/run_sb3_analysis.py --arm baseline           # every seed
    python ../src/run_sb3_analysis.py --arm exp1 --seed 44
    python ../src/run_sb3_analysis.py --all                    # every manifest run
    python ../src/run_sb3_analysis.py --all --skip-video --eval-episodes 50

--arm/--all resolve run directories from cached/sb3_runs_manifest.json (the
arms run_sb3_seeds.py records: baseline, fhr, exp<N>); --run-dir analyses one
directory whether or not any manifest mentions it. The analysis tick rolls the
checkpoint's own exploration rate (adapter.epsilon = model.exploration_rate —
the eps-greedy convention the in-training ticks use, so post-hoc rows are
directly comparable to late-training ones); the evaluation episodes and the
video are greedy (epsilon 0).

Every CSV row and figure the tick produces is tagged with --episode-label
(default: the last episode index in the run's rewards.csv). The tick APPENDS:
re-running with the same label adds another set of rows for that label — the
result viewer dedups nothing — so pass a fresh label (e.g. a sentinel well
past the last training episode) when the reruns must stay distinguishable.
"""
import argparse
import csv
import os
import pathlib
import sys

os.environ.setdefault("MPLBACKEND", "Agg")   # post-hoc CLI: never open a GUI

import numpy as np
import yaml

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent   # experiments/stable_baselines_3/src
SRC = SCRIPTS_DIR.parents[2] / "src"                     # repo library code
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_sb3_seeds
from run_sb3_seeds import MANIFEST


def _default_episode_label(run_dir):
    """Last episode index in the run's rewards.csv (0 if absent/empty)."""
    path = run_dir / "rewards.csv"
    if path.exists():
        table = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        if table.size:
            return int(table[-1, 0])
    print(f"[{run_dir.name}] no rewards.csv rows — using episode label 0")
    return 0


def _load_model(run_dir, device):
    """(model, adapter, checkpoint name): final.pt, else latest.pt with a
    printed note. The adapter starts at the checkpoint's own exploration rate
    (the in-training tick convention); analyse_run switches it to greedy for
    the evaluation episodes and the video."""
    checkpoint = "final"
    if not (run_dir / "checkpoints" / "final.pt").exists():
        if not (run_dir / "checkpoints" / "latest.pt").exists():
            raise FileNotFoundError(
                f"{run_dir}/checkpoints has neither final.pt nor latest.pt")
        print(f"[{run_dir.name}] checkpoints/final.pt missing — "
              f"falling back to latest.pt")
        checkpoint = "latest"
    model, adapter = run_sb3_seeds.load_run_model(run_dir, device=device,
                                                  checkpoint=checkpoint)
    adapter.epsilon = float(model.exploration_rate)
    return model, adapter, checkpoint


def _attach_run_logger(run_dir):
    """RunLogger attached to the EXISTING run dir: exp_dir is the run dir's
    grandparent (the cached/ dir), run_id its basename, config_path=None so
    the run's config.yaml copy is not clobbered."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from analysis.run_logger import RunLogger
    if run_dir.parent.name != "runs":
        raise ValueError(f"{run_dir} is not a <cached>/runs/<name> run dir")
    return RunLogger(run_dir.parent.parent, config_path=None,
                     run_id=run_dir.name)


def _greedy_episode_return(adapter, env, seed):
    """One greedy episode's summed reward (the _greedy_episode_return protocol
    of src/training.py, on the SB3 adapter's act_greedy)."""
    import torch
    state, _ = env.reset(seed=seed)
    total, terminated, truncated = 0.0, False, False
    while not (terminated or truncated):
        state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                                  device=adapter.device).unsqueeze(0)
        action = adapter.act_greedy(state_t)
        state, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
    return total


def _hankel_q_eff_rank(run_dir, episode_label):
    """Mean full-rollout 'Hankel Q' eff_rank logged at episode_label (the
    rows the tick just appended), or None when the sweep wrote none."""
    path = run_dir / "hankel_sweep.csv"
    if not path.exists():
        return None
    best = {}                                 # rollout -> (sub_len, eff_rank)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if (row["matrix"] != "Hankel Q"
                    or int(row["episode"]) != episode_label):
                continue
            sub_len = int(row["sub_len"])
            if row["rollout"] not in best or sub_len > best[row["rollout"]][0]:
                best[row["rollout"]] = (sub_len, float(row["eff_rank"]))
    if not best:
        return None
    return float(np.mean([er for _, er in best.values()]))


def analyse_run(run_dir, device="cpu", episode_label=None, eval_episodes=20,
                skip_video=False):
    """The full post-hoc pass on one run dir; returns a summary dict
    {run_dir, checkpoint, episode_label, eval_mean, eval_std, hankel_q_eff_rank,
    video} (video None when skipped)."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import gymnasium as gym
    from training import run_analysis_tick
    from analysis.visualisations.rollout_video import record_greedy_episode

    run_dir = pathlib.Path(run_dir).resolve()
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    if episode_label is None:
        episode_label = _default_episode_label(run_dir)
    model, adapter, checkpoint = _load_model(run_dir, device)
    logger = _attach_run_logger(run_dir)

    print(f"=== {run_dir.name}: analysis tick @ episode label {episode_label} "
          f"(checkpoint {checkpoint}.pt, device {model.device}) ===")
    analysis_env = run_sb3_seeds._make_analysis_env(cfg)
    run_analysis_tick(adapter, analysis_env, cfg.get("analysis") or {},
                      logger, episode_label)
    analysis_env.close()

    adapter.epsilon = 0.0                     # greedy from here: eval + video
    eval_env = gym.make(cfg["environment"]["name"])
    returns = [_greedy_episode_return(adapter, eval_env, seed=30_000 + i)
               for i in range(eval_episodes)]
    eval_env.close()
    eval_mean, eval_std = float(np.mean(returns)), float(np.std(returns))

    video = None
    if not skip_video:
        video_env = run_sb3_seeds._make_analysis_env(cfg,
                                                     render_mode="rgb_array")
        prefix = record_greedy_episode(adapter, video_env,
                                       str(run_dir / "videos"),
                                       episode=episode_label, seed=30_000)
        video_env.close()
        video = run_dir / "videos" / f"{prefix}-episode-0.mp4"

    hankel_q = _hankel_q_eff_rank(run_dir, episode_label)
    print(f"[{run_dir.name}] eval {eval_mean:.1f} +/- {eval_std:.1f} over "
          f"{eval_episodes} greedy episodes"
          + (f" | Hankel-Q eff_rank {hankel_q:.2f}" if hankel_q is not None else "")
          + (f" | video {video}" if video is not None else ""))
    return {"run_dir": run_dir, "checkpoint": checkpoint,
            "episode_label": episode_label, "eval_mean": eval_mean,
            "eval_std": eval_std, "hankel_q_eff_rank": hankel_q,
            "video": video}


def _resolve_run_dirs(exp_dir, args):
    """CLI selection -> ordered list of run dirs. --run-dir needs no manifest;
    --arm/--all read cached/sb3_runs_manifest.json, raising on gaps (same
    contract as run_sb3_seeds.load_runs)."""
    if args.run_dir:
        run_dir = (exp_dir / args.run_dir).resolve()
        if not (run_dir / "config.yaml").exists():
            raise FileNotFoundError(f"{run_dir} has no config.yaml — not a "
                                    f"completed run dir")
        return [run_dir]
    manifest = run_sb3_seeds._load_manifest(exp_dir)
    runs = manifest.get("runs", {})
    if not runs:
        raise RuntimeError(f"no runs recorded in {MANIFEST} under {exp_dir} — "
                           f"train first, or pass --run-dir")
    if args.all:
        return [(exp_dir / rel).resolve() for arm in sorted(runs)
                for _, rel in sorted(runs[arm].items(), key=lambda kv: int(kv[0]))]
    if args.arm not in runs:
        raise ValueError(f"arm {args.arm!r} not in {MANIFEST} "
                         f"(recorded: {sorted(runs)})")
    seeds = ([args.seed] if args.seed is not None
             else manifest.get("seeds") or sorted(runs[args.arm], key=int))
    out = []
    for seed in seeds:
        rel = runs[args.arm].get(str(seed))
        if rel is None:
            raise ValueError(f"no {args.arm} run for seed {seed} in {MANIFEST} "
                             f"(recorded seeds: {sorted(runs[args.arm], key=int)})")
        out.append((exp_dir / rel).resolve())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--run-dir",
                       help="one run dir, e.g. cached/runs/<name> (works for "
                            "runs no manifest mentions)")
    which.add_argument("--arm",
                       help="manifest arm (baseline, fhr, exp<N>): every seed "
                            "of the arm, or one with --seed")
    which.add_argument("--all", action="store_true",
                       help="every run recorded in the manifest")
    parser.add_argument("--seed", type=int, default=None,
                        help="with --arm: analyse only this seed's run")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"),
                        default="cpu",
                        help="device to load the model on (default cpu)")
    parser.add_argument("--episode-label", type=int, default=None,
                        help="episode tag for the appended CSV rows / figure "
                             "names (default: last episode in rewards.csv)")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="greedy evaluation episodes, seeds 30000+i "
                             "(default 20)")
    parser.add_argument("--skip-video", action="store_true",
                        help="skip the greedy rollout mp4")
    args = parser.parse_args()
    if args.seed is not None and args.arm is None:
        parser.error("--seed requires --arm")

    exp_dir = pathlib.Path.cwd().resolve()
    run_dirs = _resolve_run_dirs(exp_dir, args)
    print(f"analysing {len(run_dirs)} run(s): "
          f"{[str(d.relative_to(exp_dir)) if d.is_relative_to(exp_dir) else str(d) for d in run_dirs]}")
    summaries = [analyse_run(d, device=args.device,
                             episode_label=args.episode_label,
                             eval_episodes=args.eval_episodes,
                             skip_video=args.skip_video)
                 for d in run_dirs]

    print("\n---- summary ----")
    for s in summaries:
        hq = (f"{s['hankel_q_eff_rank']:.2f}"
              if s["hankel_q_eff_rank"] is not None else "n/a")
        print(f"{s['run_dir'].name}: ep label {s['episode_label']} "
              f"({s['checkpoint']}.pt) | eval {s['eval_mean']:.1f} +/- "
              f"{s['eval_std']:.1f} | Hankel-Q eff_rank {hq}"
              + (f" | {s['video'].name}" if s["video"] else ""))
    return summaries


if __name__ == "__main__":
    main()
