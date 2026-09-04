"""Poster-quality spectrum figures from a real SB3 MountainCar baseline run.

The logger only stores the top 12 singular values per sweep row
(RunLogger.SWEEP_N_SV), so the FULL spectra are recomputed here from the raw
rollout traces the run saved (trajectories/ep*_seed*.npz), rebuilding the same
Hankel matrix as src/analysis/low_rank/hankel_policy.py:
    mid = len(tau)//2;  scipy.linalg.hankel(tau[:mid+1], tau[mid:])
A sanity check asserts the recomputed top-12 match the sv_01.. columns logged
in hankel_sweep.csv before anything is plotted.

Renders:
  1. spectrum_evolution        — normalised log-scree per checkpoint, viridis + colorbar
  2. spectrum_start_vs_final   — normalised linear main panel + log-scree inset
Each as PDF (vector, for the poster), SVG and 300-dpi PNG.
"""
import csv
import pathlib
import collections
import re

import numpy as np
import scipy.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

ROOT = pathlib.Path("/home/souparna/Low-Rank-RL-TL")
import sys
EXP = sys.argv[1] if len(sys.argv) > 1 else "mountaincar"
RUN_NAME = (sys.argv[2] if len(sys.argv) > 2
            else "sb3_mountaincar_lagsrc_baseline_seed44_20260829-002641")
RUN = ROOT / "experiments/stable_baselines_3" / EXP / "cached/runs" / RUN_NAME
OUT = ROOT / "experiments/stable_baselines_3" / EXP / "cached/poster_figures"
MATRIX = "Hankel Q"
TRAJ_KEY = "hankel-q"      # npz key RunLogger.save_trajectory slugifies the name to

# ---- full spectra recomputed from the stored rollout traces ----
TRAJ_RE = re.compile(r"^ep(\d{6})_seed(\d+)\.npz$")
raw = collections.defaultdict(dict)          # ep -> seed -> full sv array
shapes = []
for p in sorted((RUN / "trajectories").glob("*.npz")):
    m = TRAJ_RE.match(p.name)
    if not m:
        continue
    ep, seed = int(m.group(1)), int(m.group(2))
    tau = np.load(p)[TRAJ_KEY]
    mid = int(len(tau) / 2)
    hk = scipy.linalg.hankel(tau[: mid + 1], tau[mid:])
    sv = np.linalg.svd(hk, compute_uv=False)
    raw[ep][seed] = sv
    shapes.append(hk.shape)
eps = sorted(raw)
print("episodes:", eps)
print("rollout seeds:", sorted(raw[eps[0]]))
print("spectrum lengths at ep0 / final:",
      [len(v) for v in raw[eps[0]].values()],
      [len(v) for v in raw[eps[-1]].values()])

# ---- sanity check: recomputed top-12 must match the logged sv columns
# (runs from before the sv columns existed simply skip the check) ----
logged = {}
for r in csv.DictReader(open(RUN / "hankel_sweep.csv")):
    if r["matrix"] != MATRIX or "sv_01" not in r:
        continue
    logged[(int(r["episode"]), int(r["seed"]))] = np.array(
        [float(r[f"sv_{i:02d}"]) for i in range(1, 13)])
checked = 0
for (ep, seed), sv12 in logged.items():
    got = raw[ep][seed][:12]
    assert np.allclose(got, sv12, rtol=1e-4), (ep, seed, got[:3], sv12[:3])
    checked += 1
print(f"sanity check: recomputed top-12 matches hankel_sweep.csv "
      f"for all {checked} (episode, rollout) rows"
      if checked else
      "sanity check skipped: this run predates the sv_* columns")

# mean normalised spectrum across rollouts, truncated to the episode's
# shortest rollout so the mean is well-defined
curves = {}
for e in eps:
    svs = [v / v[0] for v in raw[e].values()]
    n = min(len(v) for v in svs)
    curves[e] = np.mean([v[:n] for v in svs], axis=0)
n_max = max(len(c) for c in curves.values())
print("curve lengths:", {e: len(curves[e]) for e in eps})

# ---- full-rank reference: mean spectrum of i.i.d. Gaussian same-shape
# Hankels, at the LARGEST shape seen (the reference must cover the longest
# spectrum; on envs where episodes shrink or grow with training the most
# common shape can be a degenerate few-step episode)
nr, nc = max(shapes, key=lambda s: min(s))
rng = np.random.default_rng(0)
ref = np.mean([np.linalg.svd(rng.standard_normal((nr, nc)), compute_uv=False)
               for _ in range(60)], axis=0)
ref = (ref / ref[0])[:n_max]
print(f"full-rank ref ({nr}x{nc}): {len(ref)} values, tail {ref[-1]:.3f}")

def nice_step(n):
    for s in (1, 2, 5, 10, 20, 50, 100, 200, 500):
        if n / s <= 6:
            return s
    return 1000


# When spectrum lengths span an order of magnitude (episode length grows or
# shrinks drastically with training, e.g. CartPole 5 -> 250 steps), a linear
# index axis crams the short spectra into a sliver — switch to a log index
# axis; otherwise (MountainCar-like data) stay linear.
n_min = min(len(c) for c in curves.values())
XLOG = n_max >= 8 * max(1, n_min)
if XLOG:
    XT = [t for t in (1, 2, 5, 10, 25, 50, 100, 250, 500) if t <= n_max]
else:
    XT = [1] + list(range(nice_step(n_max), n_max + 1, nice_step(n_max)))
print("x axis:", "log" if XLOG else "linear", "ticks", XT)

GRAY = "#4a4a4a"

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 15, "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5, "legend.fontsize": 12.5,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.1,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.7,
    "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.dpi": 300,
})
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, stem):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}")
    print("wrote", OUT / f"{stem}.[pdf|svg|png]")


def x_of(c):
    return np.arange(1, len(c) + 1)


def setup_x(a, labelsize=None):
    if XLOG:
        a.set_xscale("log")
        a.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        a.set_xlim(0.92, n_max * 1.08)
    else:
        a.set_xlim(0, n_max + 1)
    a.set_xticks(XT)
    if labelsize:
        a.tick_params(labelsize=labelsize)


# ================= 1. spectrum evolution =================
fig, ax = plt.subplots(figsize=(8.4, 5.2))
cmap = plt.get_cmap("viridis")
norm = Normalize(vmin=eps[0], vmax=eps[-1])
for e in eps:
    ax.plot(x_of(curves[e]), curves[e], color=cmap(norm(e)), lw=2.1,
            solid_capstyle="round", zorder=2)
ax.plot(x_of(ref), ref, ls=(0, (5, 3)), color=GRAY, lw=2.3, zorder=3)
# label the reference where there is empty space: above it on a linear axis,
# below it on a log axis (where the curve hugs the top of the panel)
_ann_i = max(2, int(len(ref) ** 0.62) if XLOG else int(len(ref) * 0.55))
ax.annotate("full-rank Hankel sequence", xy=(_ann_i, ref[_ann_i - 1]),
            xytext=(0, -18 if XLOG else 12), textcoords="offset points",
            ha="center", va="top" if XLOG else "baseline",
            color=GRAY, fontsize=12.5)
ax.set_yscale("log")
ax.set_xlabel("singular value index $i$")
ax.set_ylabel(r"$\sigma_i \, / \, \sigma_1$")
setup_x(ax)
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
cb.set_label("training episode")
cb.outline.set_visible(False)
fig.tight_layout()
save(fig, "spectrum_evolution")
plt.close(fig)

# ================= 2. start vs final, log-scree inset =================
BLUE, ORANGE = "#0173b2", "#de8f05"
e0, eF = eps[0], eps[-1]
fig, ax = plt.subplots(figsize=(8.6, 6.0))


def draw(a, lw):
    a.plot(x_of(ref), ref, ls=(0, (5, 3)), color=GRAY, lw=lw, zorder=2)
    a.plot(x_of(curves[e0]), curves[e0], color=ORANGE, lw=lw, zorder=3)
    a.plot(x_of(curves[eF]), curves[eF], color=BLUE, lw=lw, zorder=4)


draw(ax, 2.5)
ax.set_xlabel("singular value index $i$")
ax.set_ylabel(r"$\sigma_i \, / \, \sigma_1$")
setup_x(ax)
ax.set_ylim(-0.03, 1.06)
ax.legend([plt.Line2D([], [], color=ORANGE, lw=2.5),
           plt.Line2D([], [], color=BLUE, lw=2.5),
           plt.Line2D([], [], color=GRAY, ls=(0, (5, 3)), lw=2.5)],
          [f"episode {e0} (start)", f"episode {eF} (trained)",
           "full-rank Hankel sequence"],
          loc="lower left",
          bbox_to_anchor=(0.5 if XLOG else 0.085, 0.115),
          handlelength=2.2, borderaxespad=0, labelspacing=0.55)

# on a log index axis the reference hugs the panel top, so the free region
# is the centre-right below it (ending before its right-edge dive)
axin = ax.inset_axes([0.31, 0.33, 0.41, 0.38] if XLOG
                     else [0.55, 0.56, 0.43, 0.38])
draw(axin, 1.9)
axin.set_yscale("log")
setup_x(axin, labelsize=10)
axin.grid(alpha=0.25, lw=0.6)
axin.spines[["top", "right"]].set_visible(True)
for s in axin.spines.values():
    s.set_edgecolor("#b9b7b0")
axin.text(*(0.5, 0.07) if XLOG else (0.035, 0.07), "log scale",
          transform=axin.transAxes, ha="center" if XLOG else "left",
          fontsize=10.5, color="#555555")
fig.tight_layout()
save(fig, "spectrum_start_vs_final")
plt.close(fig)
