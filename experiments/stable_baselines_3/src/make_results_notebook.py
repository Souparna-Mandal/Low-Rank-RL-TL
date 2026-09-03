"""Generate a results notebook for one SB3 FHR family from the shared template.

Every `exp*_results*.ipynb` under experiments/stable_baselines_3 is a thin
call site into src/analysis/visualisations/fhr_figures.py; this script writes
one such notebook so a new family (a new env, a new algorithm, a new config)
gets the full report - training / greedy-eval curves, the sample-efficiency
ladder, final returns, every internals diagnostic, the loss- AND
gradient-stream rho ratios with the lambda-selection table, rollout Hankel
spectra, the penalised-window rank probe, compute cost, videos - with EVERY
figure saved as its own file under figures/<family>/ (one copy-pastable image
per cell). Run from the experiment dir:

    cd experiments/stable_baselines_3/ant
    python ../src/make_results_notebook.py --config configs/config_sb3_td3.yaml

Defaults: the manifest is derived from the config name (run_sb3_seeds
._manifest_name), the notebook is exp1_fhr<algo>_results.ipynb, the title
"FHR-<ALGO> on <env>". Re-running overwrites the notebook's cells (execute it
afterwards with `jupyter nbconvert --to notebook --execute --inplace`).
"""
import argparse
import pathlib
import sys

import nbformat as nbf
import yaml

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
import run_sb3_seeds  # noqa: E402

ALGO_LABEL = {"sac": "SAC", "sacd": "SAC-Discrete", "dqn": "DQN",
              "qrdqn": "QR-DQN", "td3": "TD3"}

# One paragraph per env, in the voice of the existing notebooks.
BLURBS = {
    "Ant-v5": (
        "MuJoCo `Ant-v5` (obs 105, actions 8). Ant **terminates early** on "
        "unhealthy states, so episodes are variable-length and short early in "
        "training - the FHR episodic buffer's predecessor lookup and the "
        "order-8 arms see genuinely truncated recurrence windows here."),
    "HalfCheetah-v5": (
        "MuJoCo `HalfCheetah-v5` (obs 17, actions 6). It never terminates early "
        "(fixed 1000-step episodes), so the recurrence residual is never "
        "truncated by a done flag - the cleanest read of the penalty."),
    "Swimmer-v5": (
        "MuJoCo `Swimmer-v5` (obs 8, actions 2, gamma 0.9999 per the zoo). The "
        "SAC family here came back mechanism-ambiguous (every arm within noise "
        "of the baseline); this family is the lower-variance re-read."),
    "Pendulum-v1": (
        "`Pendulum-v1`, the validation family of the track: 20k steps, two "
        "seeds, every code path (learned c, c(s,a), PER, frozen-theory c) and "
        "both probes on a dense cadence."),
    "MountainCarContinuous-v0": (
        "`MountainCarContinuous-v0`: sparse reward (+100 at the goal), "
        "999-step truncation, Ornstein-Uhlenbeck exploration noise per the zoo "
        "recipe. The env is bimodal - a seed either finds the goal or never "
        "does - so read the per-seed dots, not only the means."),
}

TD3_NOTE = """
**Penalty scale (TD3-native).** Stock TD3's critic loss is the plain sum over
the twin critics, $\\sum_i \\mathrm{MSE}_i$, and the FHR term joins on the same
footing: $L = \\sum_i \\mathrm{MSE}_i + \\lambda \\sum_i \\mathrm{Huber}_i$ - per
critic $\\mathrm{MSE}_i + \\lambda\\,\\mathrm{Huber}_i$, with no division by the
number of critics. `penalty_raw` logs $\\sum_i \\mathrm{Huber}_i$ and `td_loss`
the stock $\\sum_i \\mathrm{MSE}_i$, so $\\rho_{loss} = \\lambda\\,$penalty / TD is
the ratio of the two terms exactly as optimised. $\\lambda$ is therefore a
TD3-scale knob and **not numerically comparable to the SAC families'**
$\\lambda$; the cross-algorithm quantities are the stream ratios of section 5b.

**Why TD3.** The target is the plain Bellman backup (no entropy term with a
moving temperature inside the value the recurrence is fitted to), so the
on-trajectory recurrence is exact up to the zero-mean smoothing noise and the
frozen-theory control $c = (1 + 1/\\gamma, -1/\\gamma)$ is a genuine identity
test. The actor is deterministic and trained on $Q_1(s, \\pi(s))$ alone every
`policy_delay` critic steps, so the FHR-shaped $Q_1$ feeds straight into the
deterministic policy gradient. Exploration is a fixed action noise, so
`rewards.csv` is the noisy behaviour policy while `eval.csv` is the
deterministic actor.
"""

COMMON_NOTE = """
The **baseline arm is bit-for-bit stock SB3 {algo}** - same RNG stream, same
updates, with both probes on (asserted in `tests/test_sb3_{test}_fhr.py`) - so
an arm-vs-baseline gap is attributable to the FHR penalty alone. Every arm
below is exactly what `{config}` defines under `experiment.fhr_experiments`.

**Every figure is saved individually** under `figures/{family}/` (PDF + PNG,
300 dpi) so each can be copy-pasted on its own; the file name is printed under
each cell.
"""


def md(src):
    return nbf.v4.new_markdown_cell(src.strip("\n"))


def code(src):
    return nbf.v4.new_code_cell(src.strip("\n"))


def build(config, manifest, title, cfg, out_name):
    algo = str(cfg["algo"]["type"]).lower()
    algo_label = ALGO_LABEL.get(algo, algo.upper())
    env = cfg["environment"]["name"]
    family = cfg["experiment"]["name"]
    seeds = cfg["experiment"].get("seeds") or [cfg["experiment"]["seed"]]
    n_steps = int(cfg["algo"]["n_timesteps"])
    n_act = max(1, len(cfg["analysis"]["methods"][0]["outputs"]) - 1)
    blurb = BLURBS.get(env, f"`{env}`.")
    test = "td3" if algo == "td3" else ("sac" if algo in ("sac", "sacd") else "")
    cells = []

    cells.append(md(f"""
# {title}

{blurb} Stock SB3 {algo_label} vs **FHR{algo_label}** on the recipe in
`{config}` ({n_steps:,} steps, seeds {seeds}).
{TD3_NOTE if algo == "td3" else ""}
{COMMON_NOTE.format(algo=algo_label, test=test, config=config, family=family)}
"""))

    cells.append(md("## 0 · Launch - train whatever the config defines"))
    cells.append(code(f"""
import pathlib, sys
SRC_RUNNERS = pathlib.Path.cwd().parent / "src"
if str(SRC_RUNNERS) not in sys.path:
    sys.path.insert(0, str(SRC_RUNNERS))
import run_sb3_seeds as runner
import yaml

CONFIG = "{config}"
MANIFEST = "{manifest}"
# Every arm below - launched and analysed - is exactly what the config's
# experiment.fhr_experiments block currently defines; nothing is hardcoded.
EXPERIMENTS = sorted(int(k) for k in
                     (yaml.safe_load(open(CONFIG))["experiment"]
                      .get("fhr_experiments") or {{}}))
print("config experiments:", EXPERIMENTS)

LAUNCH = False        # the waves are launched detached from a shell (README);
FORCE_EXP = False     # flip to launch (or resume the missing pairs) from here
if LAUNCH:
    manifest = runner.launch_all(max_workers=6, force=FORCE_EXP,
                                 experiments=EXPERIMENTS, config=CONFIG)
    print({{k: sorted(v) for k, v in manifest["runs"].items()}})
"""))

    cells.append(md("""
## Setup - the figure toolkit

Every figure in this notebook comes from
[`analysis.visualisations.fhr_figures`](../../../src/analysis/visualisations/fhr_figures.py),
so every family plots the same way and a fix lands once. Conventions:

* **Colour identifies the arm**, never a single hyper-parameter; the baseline
  is near-black; **dashed = frozen-c control**, dotted = c(s,a), dash-dot = PER.
* **The seed band is mean ± 1 s.e.m.** (`ff.BAND`: "sem" | "ci95" | "iqr" | "minmax").
* **Sample efficiency is read off the training stream** (`rewards.csv`), not
  the greedy-eval curve.
* **One figure per cell, saved individually** (`F.save` prints the paths).
"""))
    cells.append(code("""
import numpy as np
import matplotlib.pyplot as plt

REPO = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
            if (p / "src" / "analysis").is_dir())
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
from analysis.visualisations import fhr_figures as ff

ff.set_pub_style()
ff.BAND = "sem"          # seed band: "sem" | "ci95" | "iqr" | "minmax"

F = ff.load_family(CONFIG, MANIFEST)
F.summary()

# The sample-efficiency ladder (training stream, section 3). Pass step= /
# start= to auto_thresholds to pin it explicitly.
THRESHOLDS = F.auto_thresholds()
print("thresholds:", THRESHOLDS)
"""))

    cells.append(md("""
## 1 · Training curves - the episodes as trained

`rewards.csv`: every training episode's return exactly as the behaviour
policy experienced it (exploration noise included), against cumulative env
steps, rolling-mean smoothed. One figure per arm against the shared baseline,
then the overlay. Dots mark where the seed-mean curve first crosses each
threshold of the sample-efficiency ladder (tabulated in section 3).
"""))
    cells.append(code("""
for a in F.fhr_arms:
    fig = F.fig_training(a.key, thresholds=THRESHOLDS)
    F.save(fig, f"01_training_{a.key}")
    plt.show()
"""))
    cells.append(code("""
fig = F.fig_overlay("train")
F.save(fig, "01_training_all")
plt.show()
"""))

    cells.append(md("""
## 2 · Learning curves - greedy evaluation

`eval.csv`: the deterministic actor on fixed reset seeds, so the curves are
paired across arms and the training stream is untouched. One figure per arm
against the baseline, then the overlay.
"""))
    cells.append(code("""
for a in F.fhr_arms:
    fig = F.fig_eval(a.key)
    F.save(fig, f"02_eval_{a.key}")
    plt.show()
"""))
    cells.append(code("""
fig = F.fig_overlay("eval")
F.save(fig, "02_eval_all")
plt.show()
"""))

    cells.append(md("""
## 3 · Sample efficiency - the claim FHR actually makes

For each arm and each threshold: the first env step at which that seed's
rolling-mean training return reaches it, averaged over seeds. A threshold
only some seeds reach is marked (open marker / `*n`) - the mean over the seeds
that made it is biased low and is deliberately kept off the line.
"""))
    cells.append(code("_ = F.table_sample_efficiency(THRESHOLDS)"))
    cells.append(code("""
fig = F.fig_steps_to_threshold(THRESHOLDS)
F.save(fig, "03_steps_to_threshold")
plt.show()
"""))
    cells.append(code("""
# The same numbers as a ratio: > 1 means the arm reached that return in fewer
# environment steps than the baseline.
fig = F.fig_speedup(THRESHOLDS)
F.save(fig, "03_speedup")
plt.show()
"""))

    cells.append(md("""
## 4 · Final performance - one figure per lambda

Final greedy-eval return, one figure per lambda rung. Bar = seed mean,
whisker = ± 1 s.e.m., open dots = the individual seeds, percentage = change
against the baseline of the same figure.
"""))
    cells.append(code("""
for lam in F.lambdas:
    fig = F.fig_final(lam)
    F.save(fig, f"04_final_lambda{lam:g}")
    plt.show()
"""))
    cells.append(code("""
# The learned-c sweep as one lambda x order map (skipped when the family has
# only one lambda or one order).
if len(F.lambdas) > 1 and len(F.orders) > 1:
    fig = F.fig_grid()
    F.save(fig, "04_grid")
    plt.show()
"""))

    cells.append(md(f"""
## 5 · FHR + {algo_label} internals - one figure per diagnostic

`train_diagnostics.csv` is one row per gradient burst; `F.prime_diag_cache()`
bins each run once (per-bin median, cached next to the run), so every panel is
a few hundred points per curve. Each diagnostic is its own figure: the
weighted penalty, the critic TD loss, the recurrence residual, the companion
spectral radius and $\\sum c$ (recurrence health), $c(s,a)$ spread on the
state-conditioned arms, the actor loss, the penalty batch size and NaN skips.
The stream ratios have their own section (5b).
"""))
    cells.append(code("""
F.prime_diag_cache()      # first call: a few seconds per run. Then cached.
STREAM = {"rho", "grad_rho", "loss_ratio", "grad_ratio", "grad_cos",
          "grad_norm_td", "grad_norm_pen"}
for col, fig in F.fig_internals_individual():
    if col in STREAM:
        plt.close(fig)
        continue
    F.save(fig, f"05_{col}")
    plt.show()
"""))

    cells.append(md("""
## 5b · The two streams - rho by losses and by gradients

How hard each loss term pushes the critic, measured on the same batch:

* $\\rho_{loss} = \\lambda\\cdot$penalty / TD and its unweighted form
  penalty / TD - the ratio of the two loss terms;
* $\\rho_{grad} = \\lambda\\,\\|\\nabla_\\theta$penalty$\\| / \\|\\nabla_\\theta$TD$\\|$
  and its unweighted form - the ratio of the two **gradient** streams on the
  critic parameters (`agent.grad_probe_every`), plus their cosine (negative =
  the streams conflict) and the two norms.

The unweighted ratios are drawn for the **baseline too**: there the penalty
never enters the loss, so its ratio is the free calibration signal, and the
table converts it into the $\\lambda$ that would put a target ratio on the
critic - by gradients (the scale-free choice) and by losses. This is the
quantity that transfers between algorithms whose TD losses sit on different
scales; $\\lambda$ itself does not.
"""))
    cells.append(code("""
for col, fig in F.fig_internals_individual():
    if col not in STREAM:
        plt.close(fig)
        continue
    F.save(fig, f"05b_rho_{col}")
    plt.show()
"""))
    cells.append(code("""
F.table_rho()
print()
RHO = F.table_rho_streams(tail=0.5, targets=(0.1, 0.5, 1.0))
"""))

    cells.append(md(f"""
## 6 · Rollout Hankel rank - the critic trace and the policy itself

Stacked per-rollout Hankels of the min-twin critic trace $Q(s_t, \\pi(s_t))$
and of each of the {n_act} action dimension(s) of $\\pi(s_t)$, from the
**converged** deterministic policy (one seed per arm). A rank-$r$ Hankel
sequence satisfies an order-$r$ recurrence, so the measured rank of a
converged policy is the smallest order the penalty can enforce without
fighting the solution. One figure per signal.
"""))
    cells.append(code(f"""
_ = F.table_rollout_hankel(runner)
fig = F.fig_rollout_hankel_single(runner, which="q")
F.save(fig, "06_rollout_hankel_q")
plt.show()
for j in range({n_act}):
    fig = F.fig_rollout_hankel_single(runner, which="pi", dim=j)
    F.save(fig, f"06_rollout_hankel_pi{{j}}")
    plt.show()
"""))

    cells.append(md("""
## 7 · Penalised-window Hankel rank - the in-training probe

Rank measured **where the penalty is applied**: on sampled replay windows
(anchor + `window_rank_lags` same-episode predecessors, online critics,
buffer actions). The probe runs for every arm **including the baseline**,
which measures the same windows, so its curve is the control: if FHR operates
as a rank constraint its arms push the window rank and the penalty-block tail
ratio *below* the baseline on exactly these windows. One figure per arm per
metric, then the all-arm overlay per metric. (In-training measurement - it
cannot be back-filled; `F.window_probe_status()` says which runs carry it.)
"""))
    cells.append(code("_ = F.window_probe_status()"))
    cells.append(code("""
if F.has_window_probe:
    for mkey, mtitle, mscale in F.WINDOW_KEYS:
        for a in F.fhr_arms:
            fig = F.fig_window_rank(a.key, keys=[(mkey, mtitle, mscale)])
            F.save(fig, f"07_window_{mkey}_{a.key}")
            plt.show()
        fig = F.fig_window_rank_overlay(keys=[(mkey, mtitle, mscale)])
        F.save(fig, f"07_window_{mkey}_all")
        plt.show()
    print()
    F.table_window_rank()
"""))

    cells.append(md("## 8 · Compute cost"))
    cells.append(code("""
import os
rows = []
for k, a in F.arms.items():
    for s, d in F.run_dirs(k):
        ck = d / "checkpoints" / "final.pt"
        if ck.exists():
            rows.append((a.plain, s, (os.path.getmtime(ck)
                                      - os.path.getmtime(d / "config.yaml")) / 60))
for label, s, mins in rows:
    print(f"{label:30s} seed {s}: {mins:6.1f} min")
base_name = F.baseline.plain if F.baseline else None
base = [m for l, _, m in rows if l == base_name]
fhr = [m for l, _, m in rows if l != base_name]
if base and fhr:
    print(f"\\nbaseline mean {np.mean(base):.1f} min; FHR-arm mean "
          f"{np.mean(fhr):.1f} min (overhead x{np.mean(fhr)/np.mean(base):.2f})")
"""))

    cells.append(md("""
## 9 · Final policy rollouts - videos

`record_final_videos` reloads each run's **final checkpoint**, rebuilds the
same env wrapper stack with `render_mode="rgb_array"`, and rolls **one greedy
episode** of the deterministic actor (the policy the `eval.csv` curves score)
through gymnasium's `RecordVideo` -> `<run_dir>/videos/epfinal-episode-0.mp4`.
"""))
    cells.append(code("""
import os
os.environ.setdefault("MUJOCO_GL", "egl")   # headless MuJoCo rendering
from IPython.display import Video, display
from run_sb3_seeds import record_final_videos

for k, a in F.arms.items():
    try:
        vids = record_final_videos(k, config=CONFIG)
    except RuntimeError as e:   # arm not finished in this config's manifest yet
        print(f"{a.plain} ({k}): skipped - {e}")
        continue
    for seed, path in vids:
        print(f"{a.plain} ({k}) - seed {seed}: {path}")
        display(Video(str(path), embed=True, width=420))
"""))

    cells.append(md("""
## 10 · Findings

*(fill in after the family completes)*
"""))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "lowrank-rl", "language": "python",
                       "name": "lowrank-rl"},
        "language_info": {"name": "python"},
    }
    return nb


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/config_sb3_td3.yaml",
                    help="config path relative to the experiment dir")
    ap.add_argument("--manifest", default=None,
                    help="manifest path (default: derived from the config name)")
    ap.add_argument("--out", default=None,
                    help="notebook file name (default exp1_fhr<algo>_results.ipynb)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--exp-dir", default=".")
    args = ap.parse_args()
    exp_dir = pathlib.Path(args.exp_dir).resolve()
    cfg = yaml.safe_load(open(exp_dir / args.config))
    algo = str(cfg["algo"]["type"]).lower()
    manifest = args.manifest or run_sb3_seeds._manifest_name(args.config)
    out = exp_dir / (args.out or f"exp1_fhr{algo}_results.ipynb")
    title = args.title or (f"FHR-{ALGO_LABEL.get(algo, algo.upper())} on "
                           f"{cfg['environment']['name'].split('-')[0]}")
    nb = build(args.config, manifest, title, cfg, out.name)
    nbf.write(nb, str(out))
    print(f"wrote {out} ({len(nb['cells'])} cells; manifest {manifest})")


if __name__ == "__main__":
    main()
