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
"FHR-<ALGO> on <env>". Re-running on an existing notebook is an APPEND-ONLY
merge: every existing cell (and its outputs) is kept and only template cells
that are missing are inserted in section order, so a notebook only ever
gains analyses (--fresh overwrites instead). Execute afterwards with
`jupyter nbconvert --to notebook --execute --inplace` - WITHOUT
MPLBACKEND=Agg in the environment, or figures are saved but not shown
inline (the notebook carries a %matplotlib inline cell as belt and braces).
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

    cells.append(code("""
# Inline figures no matter how the notebook is executed (an MPLBACKEND=Agg in
# the environment would otherwise silence plt.show() under nbconvert).
%matplotlib inline
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
### Selecting arms and paper mode

Every figure and table below honours one selection: `SELECT = None` plots
every arm; `SELECT = ["exp2", "exp5"]` restricts all of them to those arms
(the baseline always stays as the reference). Any single call can still
override it with `arms=[...]`. `ff.PAPER = True` strips titles from the
SAVED files only (the caption carries them); the inline figures keep theirs.
"""))
    cells.append(code("""
SELECT = None          # e.g. ["exp2", "exp5", "exp7"] - None = all arms
F.select(SELECT)
ff.PAPER = False       # True: saved PDF/PNG without titles, for the paper
print("arms in every figure:", [a.key for a in F.fhr_arms], "+ baseline")
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
### 1b · Every seed - what the mean band hides

Thin lines are the individual seeds, the bold line their mean, grey dashed
the baseline's mean: collapses, late escapes and bimodal seeds show here and
nowhere else. One figure per arm, then the small-multiples overview.
"""))
    cells.append(code("""
for a in F.fhr_arms:
    fig = F.fig_seed_curves_arm(a.key, "train")
    F.save(fig, f"01b_seeds_train_{a.key}")
    plt.show()
fig = F.fig_seed_curves("train")
F.save(fig, "01b_seeds_train_all")
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
    cells.append(code("""
for a in F.fhr_arms:
    fig = F.fig_seed_curves_arm(a.key, "eval")
    F.save(fig, f"02b_seeds_eval_{a.key}")
    plt.show()
fig = F.fig_seed_curves("eval")
F.save(fig, "02b_seeds_eval_all")
plt.show()
"""))
    cells.append(md("""
### 2c · Escape rate - how many seeds made it, and when

The fraction of seeds at or above a return bar as training proceeds (the
bar defaults to half-way between the lowest starting level and the best
final level; pass `threshold=` to pin it). This is the "4 of 5 seeds escape
the local optimum" claim as a curve, per arm.
"""))
    cells.append(code("""
fig = F.fig_escape_rate()
F.save(fig, "02c_escape_rate")
plt.show()
_ = F.table_escape()
"""))
    cells.append(code("""
# One row per seed: first env step at the solved / escape bar, final and best
# greedy-eval return - the honest per-seed picture behind the means.
_ = F.table_per_seed()
"""))
    cells.append(md("""
### 2e · Per-lambda panels - frozen-theory c vs learned c at the same lambda

For each lambda rung, the baseline against that rung's learned-c and
frozen-c arms only, so the mechanism comparison (does the Bellman-theory
recurrence alone do it, or does the data-fitted c matter?) is one clean
panel per lambda.
"""))
    cells.append(code("""
for lam in F.lambdas:
    keys = [a.key for a in F.fhr_arms if np.isclose(a.lam, lam)
            and a.kind in ("global", "frozen")]
    if keys:
        fig = F.fig_overlay("eval", arms=[F.baseline.key] + keys if F.baseline else keys,
                            title=f"{F.env} - lambda = {lam:g}: learned vs frozen-theory c (eval)")
        F.save(fig, f"02e_lambda{lam:g}_learned_vs_frozen")
        plt.show()
"""))
    cells.append(code("""
# Baseline-normalised overlay: 1.0 = the baseline's final level, so this
# panel is comparable across environments.
fig = F.fig_normalised_overlay("eval")
F.save(fig, "02d_eval_normalised")
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
### 3b · Speed-up with paired bootstrap CIs, and the greedy-eval read

The ratio of IQM steps-to-threshold (baseline / arm) on a log2 axis with a
seed-PAIRED bootstrap CI (seed ids are shared across arms), and explicit
censoring marks ('k/n') where only some seeds reached the bar. Then the same
sample-efficiency read on the greedy-eval stream (the older notebooks'
convention), for comparison with the training-stream numbers above.
"""))
    cells.append(code("""
fig = F.fig_speedup_paired(THRESHOLDS)
F.save(fig, "03b_speedup_paired")
plt.show()
"""))
    cells.append(code("""
EVAL_THRESHOLDS = F.auto_thresholds(stream="eval")
_ = F.table_sample_efficiency(EVAL_THRESHOLDS, stream="eval")
fig = F.fig_steps_to_threshold(EVAL_THRESHOLDS, stream="eval")
F.save(fig, "03b_steps_to_threshold_eval")
plt.show()
for a in F.fhr_arms:
    fig = F.fig_eval(a.key, thresholds=EVAL_THRESHOLDS)
    F.save(fig, f"03b_eval_ladder_{a.key}")
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
    cells.append(md("""
### 4b · All arms at a glance - IQM with bootstrap confidence intervals

Every arm on one axis: the individual seeds (open dots), the interquartile
mean (the aggregate rliable recommends - robust to one collapsed or one
lucky seed) with its 95% bootstrap CI, and the baseline IQM as the dashed
reference. This is the single figure that summarises a family.
"""))
    cells.append(code("""
fig = F.fig_final_strip()
F.save(fig, "04b_final_strip")
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
    cells.append(md("""
### 5c · Choosing lambda - how hard the penalty pushes vs what it buys

Each FHR arm's tail-median stream ratio (x) against its final-return gain
over the baseline (y, %): by gradients ($\\rho_{grad}$) and by losses
($\\rho_{loss}$). Arms that barely move the critic sit on the left, the
over-penalised ones on the right; the sweet spot is the top of the arc.
"""))
    cells.append(code("""
fig = F.fig_rho_vs_gain("grad")
F.save(fig, "05c_rho_grad_vs_gain")
plt.show()
fig = F.fig_rho_vs_gain("loss")
F.save(fig, "05c_rho_loss_vs_gain")
plt.show()
"""))
    cells.append(md("""
### 5d · The learned recurrence - coefficients and their roots

Where the learned $c$ goes: each coefficient over training against the
Bellman-theory value $c = (1 + 1/\\gamma, -1/\\gamma, 0, \\dots)$ (dotted),
and the end-of-training companion roots of $z^r - \\sum_j c_j z^{r-j}$ in
the complex plane against the theory roots $\\{1, 1/\\gamma\\}$. Frozen-c
arms sit on the theory by construction; learned arms show whether the data
pulls the recurrence away from the Bellman poles.
"""))
    cells.append(code("""
for j in range(1, max(F.orders) + 1):
    fig = F.fig_coefficient(j)
    F.save(fig, f"05d_c_{j}")
    plt.show()
fig = F.fig_companion_roots()
F.save(fig, "05d_companion_roots")
plt.show()
"""))
    cells.append(code("""
# Frozen-c control check: the constant must never leave its theory init
# (max |c - theory| ~ float32 rounding); learned arms listed for contrast.
_ = F.table_coefficient_drift()
"""))
    cells.append(md("""
### 5e · Stream ratios at a glance

The seven stream time series of section 5b as one dot plot: per arm the
tail medians of $\\rho_{grad}$, $\\rho_{loss}$ (log axes, shaded = target
range) and the gradient cosine, seeds as open dots, median as a diamond;
the baseline row shows its unweighted ratios.
"""))
    cells.append(code("""
fig = F.fig_stream_summary()
F.save(fig, "05e_stream_summary")
plt.show()
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
    cells.append(md("""
### 7b · Compression vs return

Per seed: the final-quarter mean of a penalised-window metric (x) against
the final return (y), arm means as large markers with $\\pm$ 1 s.e.m. both
ways. Does compressing the penalised windows go with better policies, or
is it a tax?
"""))
    cells.append(code("""
if F.has_window_probe:
    fig = F.fig_window_vs_return("pen_tail_ratio")
    F.save(fig, "07b_pen_tail_vs_return")
    plt.show()
    fig = F.fig_window_vs_return("rank999")
    F.save(fig, "07b_rank_vs_return")
    plt.show()
"""))
    cells.append(md("""
### 7c · The window spectrum itself, and compression relative to the baseline

The shape of the penalised-window spectrum late in training ($\\sigma_i /
\\sigma_1$ and the consecutive decay $\\sigma_{i+1}/\\sigma_i$, median over
seeds and critics), the penalty block's $\\sigma_{r+1}/\\sigma_r$ over
training, and each FHR arm's window metrics divided by the baseline's on the
same windows (rule at 1 = no compression). Then one panel per lambda with
that lambda's learned and frozen-c arms against the baseline.
"""))
    cells.append(code("""
if F.has_window_probe:
    for which, ratio, name in (("sv", "normalised", "07c_window_spectrum"),
                               ("sv", "consecutive", "07c_window_spectrum_decay"),
                               ("pen_sv", "normalised", "07c_penalty_block_spectrum")):
        fig = F.fig_window_spectrum_profile(which=which, ratio=ratio)
        F.save(fig, name)
        plt.show()
    fig = F.fig_window_pen_ratio()
    F.save(fig, "07c_penalty_block_ratio_rr1")
    plt.show()
    for mkey, _, _ in F.WINDOW_KEYS:
        fig = F.fig_window_compression(mkey)
        F.save(fig, f"07c_compression_{mkey}")
        plt.show()
    for lam in F.lambdas:
        keys = [a.key for a in F.fhr_arms if np.isclose(a.lam, lam)
                and a.kind in ("global", "frozen")]
        if keys:
            for mkey, mtitle, mscale in F.WINDOW_KEYS:
                fig = F.fig_window_rank(keys, keys=[(mkey, mtitle, mscale)])
                F.save(fig, f"07c_window_{mkey}_lambda{lam:g}")
                plt.show()
"""))
    cells.append(md("""
### 7d · Rollout Hankel rank during training

The greedy-rollout Hankel(Q) spectrum from the periodic analysis sweep,
over training rather than only at the end: $\\sigma_3/\\sigma_2$ vs episode
per arm, and the late-training spectrum. Families launched before the sweep
logged singular values render a notice instead.
"""))
    cells.append(code("""
fig = F.fig_rollout_hankel_sweep()
F.save(fig, "07d_rollout_sweep_s3_s2")
plt.show()
fig = F.fig_rollout_spectrum_late()
F.save(fig, "07d_rollout_spectrum_late")
plt.show()
"""))

    cells.append(md("""
## 8 · Paper figures

The headline forms: the learning curve with IQM and a 95% bootstrap CI band
and short labels (draw several envs into one row by passing `ax=`), and the
two low-rank spectra side by side - the converged policy's rollout Hankel
over ALL seeds (median, min-max band) and the final penalised-window
spectrum - with rules at the penalty orders. Set `ff.PAPER = True` above to
save these without titles.
"""))
    cells.append(code("""
fig = F.fig_learning_curves_paper("eval", letter="(a)")
F.save(fig, "08_learning_curve_paper")
plt.show()
fig = F.fig_spectra_paper(runner)
F.save(fig, "08_spectra_paper")
plt.show()
"""))
    cells.append(md("## 8b · Compute cost"))
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
    cells.append(code("""
# Custom video eval: one greedy episode of a chosen TRAINED seed's checkpoint
# on an arbitrary env reset seed (need not be a training seed).
CUSTOM_VIDEO = False
TRAIN_SEED, EVAL_SEED = F.seeds[0], 16
VIDEO_ARMS = [F.baseline.key] + [a.key for a in F.fhr_arms] if F.baseline else [a.key for a in F.fhr_arms]
if CUSTOM_VIDEO:
    from analysis.visualisations.rollout_video import record_greedy_episode
    for arm in VIDEO_ARMS:
        d = dict(F.run_dirs(arm)).get(str(TRAIN_SEED))
        if d is None:
            print(f"{arm}: no run for seed {TRAIN_SEED}")
            continue
        _, adapter = runner.load_run_model(d)
        env = runner._make_env(F.cfg, render_mode="rgb_array")
        prefix = record_greedy_episode(adapter, env, str(d / "videos"),
                                       episode=f"eval{EVAL_SEED}", seed=int(EVAL_SEED))
        env.close()
        display(Video(str(d / "videos" / f"{prefix}-episode-0.mp4"), embed=True, width=420))
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


def cell_key(cell):
    """A content-derived identity for a cell: its full source with comments
    and blank lines dropped and whitespace collapsed (a first-line key
    collided - two template cells start with `for lam in F.lambdas:`)."""
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    lines = [l.rstrip() for l in src.splitlines()]
    if cell["cell_type"] != "markdown":
        lines = [l for l in lines if l.strip() and not l.lstrip().startswith("#")]
    body = " ".join(" ".join(l.split()) for l in lines if l.strip())
    return f"{cell['cell_type']}:{body}"


def merge(existing, fresh):
    """Append-only merge: every cell of the existing notebook is kept as is
    (outputs included); template cells whose key is absent are inserted
    right after the existing cell that corresponds to the template's
    preceding cell, so section order is preserved. Returns (nb, n_added)."""
    cells = list(existing["cells"])
    keys = [cell_key(c) for c in cells]
    n_added, cursor = 0, 0          # cursor: slot right after the last
    for cell in fresh["cells"]:     # template cell matched or inserted
        k = cell_key(cell)
        if k in keys:
            cursor = keys.index(k) + 1
            continue
        cells.insert(cursor, cell)
        keys.insert(cursor, k)
        cursor += 1
        n_added += 1
    existing["cells"] = cells
    return existing, n_added


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
    ap.add_argument("--fresh", action="store_true",
                    help="overwrite the notebook instead of the default "
                         "append-only merge (existing cells and their outputs "
                         "are kept; only template cells that are missing get "
                         "inserted, in section order)")
    args = ap.parse_args()
    exp_dir = pathlib.Path(args.exp_dir).resolve()
    cfg = yaml.safe_load(open(exp_dir / args.config))
    algo = str(cfg["algo"]["type"]).lower()
    manifest = args.manifest or run_sb3_seeds._manifest_name(args.config)
    out = exp_dir / (args.out or f"exp1_fhr{algo}_results.ipynb")
    title = args.title or (f"FHR-{ALGO_LABEL.get(algo, algo.upper())} on "
                           f"{cfg['environment']['name'].split('-')[0]}")
    nb = build(args.config, manifest, title, cfg, out.name)
    if out.exists() and not args.fresh:
        existing = nbf.read(str(out), as_version=4)
        nb, n_added = merge(existing, nb)
        nbf.write(nb, str(out))
        print(f"merged {n_added} new cell(s) into {out} ({len(nb['cells'])} "
              f"cells; existing cells kept; manifest {manifest})")
    else:
        nbf.write(nb, str(out))
        print(f"wrote {out} ({len(nb['cells'])} cells; manifest {manifest})")


if __name__ == "__main__":
    main()
