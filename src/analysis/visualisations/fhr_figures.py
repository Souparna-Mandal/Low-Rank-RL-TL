"""Publication-quality figures for the SB3 FHR-SAC comparison families.

One module behind every `exp*_results*.ipynb` in experiments/stable_baselines_3,
so the four MuJoCo notebooks (HalfCheetah, Ant, Swimmer, HumanoidStandup) plot
the same family the same way and a figure only has to be fixed once.

Design rules, all of them deliberate:

* **Colour identifies the ARM, never a single hyper-parameter.** The older
  notebooks coloured by lambda; in a family whose arms all share one lambda
  (Ant's tuned grid is entirely lambda = 0.1) that painted every curve the same
  colour. Here each non-baseline arm draws the next colour from an Okabe-Ito
  colour-blind-safe palette, the baseline is always near-black, and the
  learned-c / frozen-c distinction is carried by the line style.
* **One figure per call.** Each `fig_*` returns a standalone Figure sized for a
  paper column, so a notebook cell produces exactly one copy-pastable image.
* **Bounded variance.** The seed band defaults to mean +- 1 s.e.m. rather than
  the seed min-max envelope, which on 5 seeds is a ragged outlier trace. Set
  `BAND` (or pass `band=`) to "sem" | "ci95" | "iqr" | "minmax".
* **Sample efficiency is read off the TRAINING stream** (rewards.csv, the
  episodes as trained), not the greedy-eval curve - see `steps_to_thresholds`.
* **Diagnostics are read once and cached.** train_diagnostics.csv is one row
  per gradient step (~1e6 rows, ~180 MB per run); `load_diag` bins it to a few
  hundred points and caches the result next to the run, so the internals
  figures load in milliseconds after the first pass.
"""
import csv
import json
import pathlib

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from analysis.low_rank.window_rank import arm_tick_metrics

__all__ = ["set_pub_style", "load_family", "Family", "OKABE_ITO", "BAND"]

# Okabe-Ito, the standard colour-blind-safe qualitative set. Yellow (#F0E442)
# is dropped: it is illegible as a thin line on white.
OKABE_ITO = ["#0072B2",  # blue
             "#D55E00",  # vermillion
             "#009E73",  # bluish green
             "#CC79A7",  # reddish purple
             "#56B4E9",  # sky blue
             "#E69F00",  # orange
             "#7F3C8D",  # deep purple
             "#8C564B",  # brown
             "#17BECF",  # cyan
             "#BCBD22",  # olive
             "#4C4C9D",  # indigo
             "#F781BF"]  # light pink
BASELINE_COLOUR = "#222222"

BAND = "sem"          # module-level default for every seed band
N_GRID = 400          # points on the common env-step grid of a seed band


# --------------------------------------------------------------------------
# style
# --------------------------------------------------------------------------
def set_pub_style(base_size=10):
    """Matplotlib rcParams for ICLR/NeurIPS-style figures: no top/right spines,
    a faint dashed grid, unboxed legends, 300-dpi vector-friendly output."""
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
        "pdf.fonttype": 42,          # embed TrueType, not Type-3: ICLR requires it
        "ps.fonttype": 42,
        "font.size": base_size,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.titlesize": base_size + 1,
        "axes.titleweight": "regular",
        "axes.titlepad": 8,
        "axes.labelsize": base_size,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#B0B0B0",
        "grid.linestyle": (0, (3, 3)),
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "legend.fontsize": base_size - 1.5,
        "legend.handlelength": 1.9,
        "legend.borderaxespad": 0.4,
        "lines.linewidth": 1.9,
        "lines.solid_capstyle": "round",
        "figure.constrained_layout.use": True,
    })


def _steps_formatter():
    def fmt(v, _pos):
        if v >= 1e6:
            return f"{v / 1e6:g}M"
        if v >= 1e3:
            return f"{v / 1e3:g}k"
        return f"{v:g}"
    return FuncFormatter(fmt)


def _fmt_thr(t):
    return f"{t:.0f}" if abs(t) >= 10 else f"{t:.2g}"


# --------------------------------------------------------------------------
# thresholds
# --------------------------------------------------------------------------
def nice_thresholds(top, n_target=6, step=None, start=None):
    """A round ladder of return thresholds up to `top`, e.g. Ant's
    1000 / 1500 / ... / 3500. `step` and `start` override the automatic choice
    (which snaps the step to a 1-2-2.5-5 x 10^k grid)."""
    if not np.isfinite(top) or top <= 0:
        return []
    if step is None:
        raw = top / (n_target + 1)
        mag = 10.0 ** np.floor(np.log10(raw))
        step = min((m * mag for m in (1, 2, 2.5, 5, 10)),
                   key=lambda s: abs(s - raw))
    if start is None:
        start = 2 * step
    out, t = [], float(start)
    while t <= top + 1e-9:
        out.append(round(t, 6))
        t += step
    return out


def first_cross(x, y, thr):
    """Env step at which y first reaches thr (nan when it never does)."""
    hit = np.asarray(y) >= thr
    return float(np.asarray(x)[np.argmax(hit)]) if hit.any() else np.nan


# --------------------------------------------------------------------------
# seed aggregation
# --------------------------------------------------------------------------
def _aggregate(curves, n_grid=N_GRID, band=None):
    """[(seed, x, y), ...] -> (grid, centre, lo, hi, n_seeds).

    Every seed is interpolated onto the common env-step grid spanned by all of
    them, so seeds of different length (Ant terminates early, so episode counts
    differ) still compare pointwise instead of by index.
    """
    band = band or BAND
    curves = [(s, np.asarray(x, float), np.asarray(y, float))
              for s, x, y in curves if len(x) > 1]
    if not curves:
        return None
    lo_x = max(x[0] for _, x, _ in curves)
    hi_x = min(x[-1] for _, x, _ in curves)
    if not np.isfinite(lo_x) or hi_x <= lo_x:
        return None
    grid = np.linspace(lo_x, hi_x, n_grid)
    Y = np.stack([np.interp(grid, x, y) for _, x, y in curves])
    n = Y.shape[0]
    if n == 1 or band == "none":
        return grid, Y[0], Y[0], Y[0], n
    if band == "minmax":
        return grid, Y.mean(0), Y.min(0), Y.max(0), n
    if band == "iqr":
        return (grid, np.median(Y, 0), np.percentile(Y, 25, 0),
                np.percentile(Y, 75, 0), n)
    sem = Y.std(0, ddof=1) / np.sqrt(n)
    k = 1.96 if band == "ci95" else 1.0
    m = Y.mean(0)
    return grid, m, m - k * sem, m + k * sem, n


def _empty_fig(msg, figsize=(6.2, 2.2)):
    """A placeholder figure, so an unlaunched family renders a readable note
    instead of raising halfway down a notebook."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=10,
            color="#666666", transform=ax.transAxes, wrap=True)
    ax.set_axis_off()
    return fig


def _band_caption(band=None):
    return {"sem": "mean $\\pm$ 1 s.e.m.", "ci95": "mean $\\pm$ 95% CI",
            "iqr": "median, IQR band", "minmax": "mean, min-max band",
            "none": "single seed"}[band or BAND]


def _smooth(y, window):
    """Centred rolling mean that keeps the array length (edges shrink the
    window instead of padding), so bands stay aligned with the mean line."""
    y = np.asarray(y, float)
    w = int(max(1, min(window, len(y))))
    if w <= 1:
        return y
    kern = np.ones(w) / w
    num = np.convolve(y, kern, mode="same")
    den = np.convolve(np.ones_like(y), kern, mode="same")
    return num / den


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------
class Arm:
    """One launched arm of a family: its manifest key, plotting identity and
    the FHR hyper-parameters that distinguish it.

    `kind` is what separates arms that a lambda/order pair cannot:
    baseline | global (learned global c) | frozen (c_learning_rate 0) |
    ar (state-conditioned c(s,a) via c_predictor) | per (prioritized replay).
    Colour identifies the arm; kind picks the line style.
    """

    KIND_RANK = {"baseline": 0, "global": 1, "frozen": 2, "ar": 3, "per": 4}
    KIND_LS = {"baseline": "-", "global": "-", "frozen": (0, (5, 2)),
               "ar": (0, (1, 1.3)), "per": (0, (6, 1.6, 1, 1.6))}

    def __init__(self, key, kind, lam, order, frozen, c_init, pred, colour, ls,
                 algo_label="SAC"):
        self.key, self.kind = key, kind
        self.lam, self.order = lam, order
        self.frozen, self.c_init, self.pred = frozen, c_init, pred
        self.colour, self.ls = colour, ls
        self.algo_label = algo_label

    @property
    def is_baseline(self):
        return self.kind == "baseline"

    def _tag(self, long=False):
        if self.kind == "frozen":
            return " (frozen c)" if long else " frozen-c"
        if self.kind == "ar":
            return (f" (c(s,a), {self.pred})" if long
                    else f" c(s,a) {self.pred}")
        if self.kind == "per":
            return " (learned c, + PER)" if long else " +PER"
        return " (learned c)" if long else ""

    @property
    def short(self):
        if self.is_baseline:
            return "baseline"
        return f"$\\lambda${self.lam:g}$\\cdot$r{self.order}{self._tag()}"

    @property
    def label(self):
        if self.is_baseline:
            return f"baseline (stock SB3 {self.algo_label})"
        return (f"FHR  $\\lambda$={self.lam:g}, r={self.order}"
                f"{self._tag(long=True)}")

    @property
    def plain(self):
        """ASCII label for printed tables (the mathtext ones do not align)."""
        if self.is_baseline:
            return f"baseline (stock SB3 {self.algo_label})"
        return f"FHR lam={self.lam:g} r={self.order}{self._tag()}"

    def __repr__(self):
        return f"<Arm {self.key}: {self.plain}>"


def load_family(config, manifest, root=None, band=None):
    """Build a Family from a config path and its manifest, both relative to
    `root` (default: the current working directory, i.e. the experiment dir
    a notebook runs in)."""
    return Family(config, manifest, root=root, band=band)


class Family:
    def __init__(self, config, manifest, root=None, band=None):
        import yaml
        self.root = pathlib.Path(root or pathlib.Path.cwd())
        self.config_path = self.root / config
        self.cfg = yaml.safe_load(open(self.config_path))
        self.band = band or BAND
        mpath = self.root / manifest
        self.manifest = (json.load(open(mpath)) if mpath.exists()
                         else {"runs": {}})
        self.env = self.cfg["environment"]["name"]
        self.name = self.cfg["experiment"]["name"]
        self.algo = str(self.cfg.get("algo", {}).get("type", "sac")).lower()
        self.algo_label = {"sac": "SAC", "sacd": "SAC-Discrete", "dqn": "DQN",
                           "qrdqn": "QR-DQN"}.get(self.algo, self.algo.upper())
        # continuous-action algos have a policy trace to Hankel per action
        # dim; the discrete ones have a categorical actor and use the greedy
        # value trajectory instead (fig_value_hankel).
        self.is_continuous = self.algo in ("sac",)
        exp = self.cfg["experiment"]
        self.seeds = [str(s) for s in (exp.get("seeds") or [exp["seed"]])]
        self.defaults = self.cfg["agent"]
        self.figdir = self.root / "figures" / self.name
        self._diag_cache = {}
        self.arms, self.unrun = self._build_arms()
        self.baseline = next((a for a in self.arms.values() if a.is_baseline),
                             None)
        self.fhr_arms = [a for a in self.arms.values() if not a.is_baseline]
        self.orders = sorted({a.order for a in self.fhr_arms})
        self.lambdas = sorted({a.lam for a in self.fhr_arms})

    # -- construction ------------------------------------------------------
    def _build_arms(self):
        runs = self.manifest.get("runs", {})
        specs, unrun = [], []
        if runs.get("baseline"):
            specs.append(dict(key="baseline", kind="baseline", lam=0.0,
                              order=0, frozen=False, c_init=None, pred=None))
        for n, ov in sorted((int(k), v) for k, v in
                            (self.cfg["experiment"].get("fhr_experiments")
                             or {}).items()):
            d = dict(self.defaults) | dict(ov)
            frozen = float(d.get("c_learning_rate", 0.0)) == 0.0
            pred = d.get("c_predictor")
            if d.get("prioritized_replay"):
                kind = "per"
            elif pred:
                kind = "ar"
            elif frozen:
                kind = "frozen"
            else:
                kind = "global"
            spec = dict(key=f"exp{n}", kind=kind,
                        lam=float(d.get("fhr_weight", 0.0)),
                        order=int(d.get("fhr_order", 2)),
                        frozen=frozen, c_init=d.get("c_init"), pred=pred)
            (specs if runs.get(spec["key"]) else unrun).append(spec)
        # Plot order: baseline, learned-global arms, frozen-c controls, then
        # the c(s,a) and PER variants - so each kind reads as a legend block.
        body = sorted([s for s in specs if s["kind"] != "baseline"],
                      key=lambda s: (Arm.KIND_RANK[s["kind"]], s["order"],
                                     s["lam"], str(s["pred"] or "")))
        arms, ci = {}, 0
        for s in [s for s in specs if s["kind"] == "baseline"] + body:
            if s["kind"] == "baseline":
                colour = BASELINE_COLOUR
            else:
                colour = OKABE_ITO[ci % len(OKABE_ITO)]
                ci += 1
            arms[s["key"]] = Arm(colour=colour, ls=Arm.KIND_LS[s["kind"]],
                                 algo_label=self.algo_label, **s)
        return arms, [s["key"] for s in unrun]

    def __repr__(self):
        return (f"<Family {self.name}: {len(self.arms)} arms, "
                f"{len(self.seeds)} seeds>")

    def summary(self):
        print(f"{self.name}  ({self.env})   seeds {', '.join(self.seeds)}")
        for a in self.arms.values():
            print(f"  {a.key:9s} {a.short:26s} {len(self.run_dirs(a.key))} runs"
                  f"   {a.colour}")
        if self.unrun:
            print("  defined but not run:", ", ".join(self.unrun))

    # -- data --------------------------------------------------------------
    def run_dirs(self, arm):
        out = []
        for s in self.seeds:
            rel = self.manifest.get("runs", {}).get(arm, {}).get(s)
            if rel and (self.root / rel / "rewards.csv").exists():
                out.append((s, self.root / rel))
        return out

    def _csv(self, arm, fname, x, y):
        cs = []
        for s, d in self.run_dirs(arm):
            p = d / fname
            if not p.exists():
                continue
            rows = list(csv.DictReader(open(p)))
            if rows:
                cs.append((s, np.array([float(r[x]) for r in rows]),
                           np.array([float(r[y]) for r in rows])))
        return cs

    def eval_curves(self, arm):
        """Greedy evaluation (eval.csv): deterministic policy on fixed reset
        seeds -> paired across arms."""
        return self._csv(arm, "eval.csv", "env_steps", "mean_reward")

    def train_curves(self, arm, window=50):
        """Training-episode returns (rewards.csv) vs cumulative env steps,
        rolling-mean smoothed. This is the stream the sample-efficiency claim
        is read from."""
        cs = []
        for s, d in self.run_dirs(arm):
            rows = list(csv.DictReader(open(d / "rewards.csv")))
            if not rows:
                continue
            x = np.cumsum([float(r["steps"]) for r in rows])
            y = np.array([float(r["reward"]) for r in rows])
            w = max(1, min(int(window), len(y)))
            y = np.convolve(y, np.ones(w) / w, mode="valid")
            cs.append((s, x[w - 1:], y))
        return cs

    def finals(self, arm, mode="last", tail=5):
        """Per-seed final greedy-eval return: the last eval point (`last`) or
        the mean of the final `tail` eval points (`tail`, less noisy)."""
        cs = self.eval_curves(arm)
        if not cs:
            return np.array([])
        if mode == "tail":
            return np.array([float(np.mean(y[-tail:])) for _, _, y in cs])
        return np.array([float(y[-1]) for _, _, y in cs])

    # -- plumbing ----------------------------------------------------------
    def _plot_band(self, ax, arm, curves, *, band=None, smooth=1, lw=None,
                   label=None, zorder=2, alpha_band=0.18):
        a = self.arms[arm]
        agg = _aggregate(curves, band=band or self.band)
        if agg is None:
            return None
        grid, m, lo, hi, n = agg
        if smooth > 1:
            m, lo, hi = _smooth(m, smooth), _smooth(lo, smooth), _smooth(hi, smooth)
        if n > 1:
            ax.fill_between(grid, lo, hi, color=a.colour, alpha=alpha_band,
                            lw=0, zorder=zorder - 1, rasterized=True)
        ax.plot(grid, m, color=a.colour, ls=a.ls,
                lw=lw or (2.4 if a.is_baseline else 1.9), zorder=zorder,
                label=(a.label if label is None else label) + f"  ({n} seeds)")
        return grid, m, n

    def _finish(self, ax, xlabel="environment steps", ylabel=None, title=None,
                legend="best"):
        ax.xaxis.set_major_formatter(_steps_formatter())
        ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if legend:
            ax.legend(loc=legend)

    def legacy_namespace(self):
        """The names the older hand-written cells expect (ARMS, INFO, BASE,
        GLOBALS_, VARIANTS, PER_ARMS, labels_of, run_dirs, curves, diag),
        rebuilt from this Family.

        Notebooks that keep bespoke analysis cells from before the toolkit
        (the DQN comparison and c-conditioning studies) do
        `globals().update(F.legacy_namespace())` and keep working - and pick
        up the per-arm colours for free, since ARMS carries them.
        """
        def by_kind(*kinds):
            return [a.label for a in self.arms.values() if a.kind in kinds]
        return {
            "ARMS": {a.label: (a.key, a.colour) for a in self.arms.values()},
            "INFO": {a.label: {"arm": a.key, "kind": a.kind, "lam": a.lam,
                               "order": a.order, "pred": a.pred,
                               "frozen": a.frozen, "colour": a.colour,
                               "ls": a.ls, "label": a.label}
                     for a in self.arms.values()},
            "BASE": self.baseline.label if self.baseline else None,
            "GLOBALS_": by_kind("global", "frozen"),
            "VARIANTS": by_kind("ar"),
            "PRED_ARMS": by_kind("ar"),      # the c-conditioning notebooks' name
            "PER_ARMS": by_kind("per"),
            "labels_of": by_kind,
            "SEEDS": self.seeds,
            "CFG": self.cfg,
            "CMAN": self.manifest,
            "run_dirs": self.run_dirs,
            "curves": (lambda arm, fname="eval.csv", x="env_steps",
                       y="mean_reward": self._csv(arm, fname, x, y)),
            "diag": (lambda arm, col:
                     self._csv(arm, "train_diagnostics.csv", "episode", col)),
        }

    def save(self, fig, name, formats=("pdf", "png")):
        """Write one figure to <exp dir>/figures/<family>/<name>.{pdf,png}."""
        self.figdir.mkdir(parents=True, exist_ok=True)
        paths = []
        for ext in formats:
            p = self.figdir / f"{name}.{ext}"
            fig.savefig(p)
            paths.append(p)
        print("saved:", "  ".join(str(p.relative_to(self.root)) for p in paths))
        return paths

    # ----------------------------------------------------------------------
    # 0 / 1 - training and greedy-eval curves, one figure per arm
    # ----------------------------------------------------------------------
    def fig_training(self, arm, thresholds=None, window=50, figsize=(6.2, 3.9),
                     ax=None, title=None):
        """Training-episode returns for one arm against the baseline.

        `thresholds` draws the sample-efficiency ladder: a faint horizontal
        rule per threshold and a dot where each curve first reaches it, so the
        sample-efficiency gap is legible on the curve itself.
        """
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize)
        keys = ([self.baseline.key] if self.baseline and arm != self.baseline.key
                else []) + [arm]
        drawn = {}
        for k in keys:
            got = self._plot_band(ax, k, self.train_curves(k, window),
                                  zorder=2 if k == arm else 1)
            if got:
                drawn[k] = got
        for t in (thresholds or []):
            ax.axhline(t, color="#9A9A9A", lw=0.6, ls=(0, (1, 3)), zorder=0)
            for k, (grid, m, _) in drawn.items():
                xc = first_cross(grid, m, t)
                if np.isfinite(xc):
                    ax.plot(xc, t, "o", ms=5.5, color=self.arms[k].colour,
                            mec="white", mew=0.9, zorder=6)
        if not drawn:      # threshold rules are lines too, so test `drawn`
            plt.close(fig)
            return _empty_fig(f"no completed runs for {self.arms[arm].plain}")
        self._finish(ax, ylabel=f"training return (rolling {window} eps)",
                     title=title or f"{self.env} - {self.arms[arm].short} "
                                    f"vs baseline (training stream)",
                     legend="lower right")
        return fig

    def fig_eval(self, arm, figsize=(6.2, 3.9), ax=None, title=None, smooth=1):
        """Greedy-eval curve for one arm against the baseline."""
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize)
        keys = ([self.baseline.key] if self.baseline and arm != self.baseline.key
                else []) + [arm]
        for k in keys:
            self._plot_band(ax, k, self.eval_curves(k), smooth=smooth,
                            zorder=2 if k == arm else 1)
        if not ax.lines:
            plt.close(fig)
            return _empty_fig(f"no completed runs for {self.arms[arm].plain}")
        self._finish(ax, ylabel="greedy evaluation return",
                     title=title or f"{self.env} - {self.arms[arm].short} "
                                    f"vs baseline (greedy eval)",
                     legend="lower right")
        return fig

    def fig_overlay(self, which="eval", arms=None, window=50, smooth=1,
                    figsize=(6.6, 4.1), title=None):
        """Every arm on one axes - the summary panel, not the per-arm ones."""
        fig, ax = plt.subplots(figsize=figsize)
        keys = arms or list(self.arms)
        for k in keys:
            cs = (self.eval_curves(k) if which == "eval"
                  else self.train_curves(k, window))
            self._plot_band(ax, k, cs, smooth=smooth,
                            zorder=1 if self.arms[k].is_baseline else 2)
        if not ax.lines:
            plt.close(fig)
            return _empty_fig("no completed runs in this family yet")
        ylab = ("greedy evaluation return" if which == "eval"
                else f"training return (rolling {window} eps)")
        self._finish(ax, ylabel=ylab,
                     title=title or f"{self.env} - all arms ({which})",
                     legend="lower right")
        return fig

    # ----------------------------------------------------------------------
    # 2 - sample efficiency, measured on the TRAINING stream
    # ----------------------------------------------------------------------
    def auto_thresholds(self, window=50, n_target=6, step=None, start=None):
        tops = []
        for k in self.arms:
            agg = _aggregate(self.train_curves(k, window), band="none")
            if agg:
                tops.append(np.nanmax(agg[1]))
        return nice_thresholds(max(tops) if tops else np.nan,
                               n_target=n_target, step=step, start=start)

    def steps_to_thresholds(self, thresholds, window=50):
        """{arm key: {threshold: array of per-seed crossing steps}} on the
        rolling-mean training curve; nan where a seed never gets there."""
        out = {}
        for k in self.arms:
            cs = self.train_curves(k, window)
            out[k] = {t: np.array([first_cross(x, y, t) for _, x, y in cs])
                      for t in thresholds}
        return out

    def table_sample_efficiency(self, thresholds=None, window=50):
        """Printed table: env steps to first reach each threshold on the
        training stream (seed mean, and how many seeds ever got there)."""
        thresholds = thresholds if thresholds is not None else self.auto_thresholds(window)
        xs = self.steps_to_thresholds(thresholds, window)
        hdr = "  ".join(f"{_fmt_thr(t):>10s}" for t in thresholds)
        print(f"steps to reach, training stream (rolling {window} eps) - "
              f"{self.env}, seeds {', '.join(self.seeds)}")
        print(f"{'arm':30s} {'final eval':>11s}   {hdr}")
        print("-" * (44 + 12 * len(thresholds)))
        for k, a in self.arms.items():
            fin = self.finals(k)
            cells = []
            for t in thresholds:
                v = xs[k][t]
                if v.size == 0 or np.all(np.isnan(v)):
                    cells.append(f"{'-':>10s}")
                else:
                    tag = "" if np.isfinite(v).all() else f"*{int(np.isfinite(v).sum())}"
                    cells.append(f"{np.nanmean(v) / 1000:8.0f}k{tag:>2s}")
            fin_s = f"{fin.mean():11.1f}" if fin.size else f"{'-':>11s}"
            print(f"{a.plain:30s} {fin_s}   {'  '.join(cells)}")
        print("* = only that many seeds reached the threshold; "
              "the mean is over those seeds only.")
        return xs

    def fig_steps_to_threshold(self, thresholds=None, window=50,
                               figsize=(6.6, 4.1), logy=False, jitter=0.07,
                               seed_dots=True):
        """The sample-efficiency claim as one figure: env steps needed to
        first reach each return threshold, per arm, from the training stream.

        A filled marker on the solid line means EVERY seed reached that
        threshold. An open marker means only some did - the mean there is over
        the seeds that made it, which biases it downwards, so those points are
        deliberately left off the line rather than allowed to bend it.
        """
        thresholds = thresholds if thresholds is not None else self.auto_thresholds(window)
        xs = self.steps_to_thresholds(thresholds, window)
        fig, ax = plt.subplots(figsize=figsize)
        pos = np.arange(len(thresholds), dtype=float)
        rng = np.random.default_rng(0)
        partial_seen = False
        for i, (k, a) in enumerate(self.arms.items()):
            vals = [xs[k][t] for t in thresholds]
            # dtype=bool matters: an empty family gives an empty float array,
            # and ~float raises "ufunc 'invert' not supported"
            full = np.array([v.size > 0 and np.isfinite(v).all() for v in vals],
                            dtype=bool)
            mu = np.array([np.nanmean(v) if np.isfinite(v).any() else np.nan
                           for v in vals], dtype=float)
            se = np.array([np.nanstd(v, ddof=1) / np.sqrt(np.isfinite(v).sum())
                           if np.isfinite(v).sum() > 1 else 0.0 for v in vals],
                          dtype=float)
            ax.errorbar(pos, np.where(full, mu, np.nan),
                        yerr=np.where(full, se, np.nan),
                        color=a.colour, ls=a.ls,
                        lw=2.3 if a.is_baseline else 1.8,
                        marker="s" if a.is_baseline else "o", ms=5.2,
                        mec="white", mew=0.8, capsize=2.5, elinewidth=1.0,
                        label=a.label, zorder=4 if a.is_baseline else 3)
            part = (~full) & np.isfinite(mu)
            if part.any():
                partial_seen = True
                ax.plot(pos[part], mu[part], ls="none", marker="o", ms=5.2,
                        mfc="white", mec=a.colour, mew=1.4, zorder=3)
            if seed_dots:
                for jx, v in enumerate(vals):
                    v = v[np.isfinite(v)]
                    if v.size:
                        ax.plot(pos[jx] + rng.uniform(-jitter, jitter, v.size),
                                v, ".", ms=3.0, color=a.colour, alpha=0.4,
                                zorder=1)
        ax.set_xticks(pos, [_fmt_thr(t) for t in thresholds])
        ax.yaxis.set_major_formatter(_steps_formatter())
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("training-return threshold")
        ax.set_ylabel("environment steps to first reach")
        sub = "\nopen marker: not all seeds reached it" if partial_seen else ""
        ax.set_title(f"{self.env} - sample efficiency, training stream "
                     f"({_band_caption('sem')}){sub}", fontsize=10)
        ax.legend(loc="upper left")
        return fig

    def fig_speedup(self, thresholds=None, window=50, figsize=(6.6, 3.9)):
        """Sample-efficiency ratio: baseline steps / arm steps at each
        threshold. > 1 means the arm got there in fewer environment steps."""
        thresholds = thresholds if thresholds is not None else self.auto_thresholds(window)
        xs = self.steps_to_thresholds(thresholds, window)
        if not self.baseline or not self.fhr_arms:
            return _empty_fig("sample-efficiency ratio needs a baseline and at "
                              "least one FHR arm with completed runs")
        def _full(key, t):
            v = xs[key][t]
            return np.nanmean(v) if v.size and np.isfinite(v).all() else np.nan
        bkey = self.baseline.key
        # Only thresholds every baseline seed reached: a mean over the subset
        # that made it is biased low and would inflate the ratio.
        thresholds = [t for t in thresholds if np.isfinite(_full(bkey, t))]
        if not thresholds:
            return _empty_fig("no return threshold was reached by every "
                              "baseline seed - nothing to take a ratio of")
        base = np.array([_full(bkey, t) for t in thresholds])
        fig, ax = plt.subplots(figsize=figsize)
        pos = np.arange(len(thresholds))
        arms = self.fhr_arms
        w = 0.8 / max(1, len(arms))
        span = []
        for i, a in enumerate(arms):
            mu = np.array([_full(a.key, t) for t in thresholds])
            ratio = base / mu
            ax.bar(pos + (i - (len(arms) - 1) / 2) * w, ratio - 1.0, bottom=1.0,
                   width=w * 0.92, color=a.colour, alpha=0.9, label=a.label,
                   hatch="//" if a.frozen else None, edgecolor="white",
                   linewidth=0.4)
            span.extend(ratio[np.isfinite(ratio)].tolist())
        ax.axhline(1.0, color=BASELINE_COLOUR, lw=1.2, zorder=3)
        if span:
            dev = max(0.05, max(abs(np.array(span) - 1.0)))
            ax.set_ylim(1 - dev * 1.35, 1 + dev * 1.9)
        ax.set_xticks(pos, [_fmt_thr(t) for t in thresholds])
        ax.set_xlabel("training-return threshold")
        ax.set_ylabel("speed-up vs baseline\n(baseline steps / arm steps)")
        ax.set_title(f"{self.env} - sample-efficiency ratio "
                     "(thresholds all baseline seeds reached)")
        ax.legend(loc="best", ncol=2)
        return fig

    # ----------------------------------------------------------------------
    # 3 - final performance, one figure per lambda
    # ----------------------------------------------------------------------
    def fig_final(self, lam=None, mode="last", figsize=(6.2, 3.4), arms=None):
        """Final greedy-eval return for the arms at one lambda, against the
        baseline. Bar = seed mean, whisker = +- 1 s.e.m., dots = seeds."""
        if arms is None:
            arms = [a for a in self.fhr_arms
                    if lam is None or np.isclose(a.lam, lam)]
        rows = ([self.baseline] if self.baseline else []) + list(arms)
        rows = [(a, self.finals(a.key, mode)) for a in rows]
        rows = [(a, v) for a, v in rows if v.size]
        if not rows:
            return _empty_fig("no completed runs for this selection")
        fig, ax = plt.subplots(figsize=figsize)
        ypos = np.arange(len(rows))[::-1]
        base = next((v.mean() for a, v in rows if a.is_baseline), np.nan)
        xmax = max(max(v.max(), v.mean()) for _, v in rows)
        for y, (a, v) in zip(ypos, rows):
            sem = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0
            ax.barh(y, v.mean(), color=a.colour, alpha=0.88, height=0.6,
                    hatch="//" if a.frozen else None, edgecolor="white",
                    linewidth=0.5, zorder=2)
            ax.errorbar(v.mean(), y, xerr=sem, color="#111111", lw=1.2,
                        capsize=3, zorder=4)
            ax.plot(v, np.full(v.size, y), "o", ms=3.2, mfc="white",
                    mec="#111111", mew=0.7, ls="none", zorder=5)
            delta = "" if a.is_baseline or not np.isfinite(base) else \
                f"  ({v.mean() / base - 1:+.1%})"
            # one aligned label column past the widest seed dot, so the
            # numbers never collide with the bars or the whiskers
            ax.text(xmax * 1.04, y, f"{v.mean():.0f}{delta}", va="center",
                    fontsize=8.5)
        ax.set_yticks(ypos, [a.short for a, _ in rows])
        ax.set_xlabel("final greedy-eval return")
        ax.set_xlim(0, xmax * 1.38)
        ttl = (f"{self.env} - final return" if lam is None else
               f"{self.env} - final return, $\\lambda$ = {lam:g} vs baseline")
        ax.set_title(ttl)
        ax.grid(axis="y", visible=False)
        return fig

    def fig_grid(self, mode="last", figsize=(4.6, 3.6)):
        """lambda x order heat map of final return relative to the baseline."""
        lams, orders = self.lambdas, self.orders
        grid = np.full((len(lams), len(orders)), np.nan)
        for a in self.fhr_arms:
            if a.frozen:
                continue
            v = self.finals(a.key, mode)
            if v.size:
                grid[lams.index(a.lam), orders.index(a.order)] = v.mean()
        base = self.finals(self.baseline.key, mode).mean() if self.baseline else np.nan
        rel = grid / base if np.isfinite(base) and base else grid
        if not np.isfinite(grid).any():
            return _empty_fig("no learned-c arm has completed runs")
        fig, ax = plt.subplots(figsize=figsize)
        span = np.nanmax(np.abs(rel - 1)) if np.isfinite(rel).any() else 0.5
        im = ax.imshow(rel, cmap="RdYlGn", vmin=1 - span, vmax=1 + span,
                       aspect="auto")
        ax.set_xticks(range(len(orders)), [f"r = {o}" for o in orders])
        ax.set_yticks(range(len(lams)), [f"$\\lambda$ = {l:g}" for l in lams])
        for i in range(len(lams)):
            for j in range(len(orders)):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.0f}\n({rel[i, j]:.2f}x)",
                            ha="center", va="center", fontsize=8.5)
        ax.set_title(f"final return vs baseline ({base:.0f})")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="x baseline")
        return fig

    # ----------------------------------------------------------------------
    # 4 - FHR + SAC internals, from a cached down-sampling of the diagnostics
    # ----------------------------------------------------------------------
    def load_diag(self, run_dir, n_bins=600, refresh=False):
        """train_diagnostics.csv, binned to `n_bins` points and cached.

        The raw file is one row per gradient step - ~1e6 rows / ~180 MB per
        run, and the old notebooks parsed all of it with csv.DictReader for
        every panel. Here it is read once with pandas, reduced to a per-bin
        median of every column, given an env-step axis (via the episode ->
        cumulative-steps map in rewards.csv) and cached to
        <run>/diag_binned_<n_bins>.npz, which reloads in milliseconds.
        """
        run_dir = pathlib.Path(run_dir)
        key = (str(run_dir), n_bins)
        if key in self._diag_cache and not refresh:
            return self._diag_cache[key]
        src = run_dir / "train_diagnostics.csv"
        if not src.exists():
            return None
        cache = run_dir / f"diag_binned_{n_bins}.npz"
        if cache.exists() and not refresh and \
                cache.stat().st_mtime >= src.stat().st_mtime:
            with np.load(cache) as z:
                out = {k: z[k] for k in z.files}
            self._diag_cache[key] = out
            return out
        import pandas as pd
        df = pd.read_csv(src)
        df = df.select_dtypes(include=[np.number])
        n = len(df)
        if n == 0:
            return None
        k = max(1, n // n_bins)
        trim = (n // k) * k
        out = {}
        import warnings
        for col in df.columns:
            v = df[col].to_numpy(dtype=np.float64)[:trim].reshape(-1, k)
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                # a column that is all-NaN in a bin (the rampdown columns are,
                # whenever the rampdown is off) is a valid empty bin, not a bug
                warnings.simplefilter("ignore", RuntimeWarning)
                out[col] = np.nanmedian(v, axis=1).astype(np.float32)
        # env-step axis: episode index -> cumulative env steps from rewards.csv
        rw = run_dir / "rewards.csv"
        if "episode" in out and rw.exists():
            rows = list(csv.DictReader(open(rw)))
            ep = np.array([float(r["episode"]) for r in rows])
            cum = np.cumsum([float(r["steps"]) for r in rows])
            out["env_steps"] = np.interp(out["episode"], ep, cum).astype(np.float32)
        else:
            out["env_steps"] = (np.arange(len(next(iter(out.values()))),
                                          dtype=np.float32) * k)
        try:
            np.savez_compressed(cache, **out)
        except OSError:
            pass
        self._diag_cache[key] = out
        return out

    def diag_curves(self, arm, col, n_bins=600):
        """[(seed, env_steps, value)] for one diagnostics column."""
        cs = []
        for s, d in self.run_dirs(arm):
            data = self.load_diag(d, n_bins)
            if data is not None and col in data:
                y = np.asarray(data[col], float)
                if np.isfinite(y).any():
                    cs.append((s, np.asarray(data["env_steps"], float), y))
        return cs

    def prime_diag_cache(self, n_bins=600, refresh=False):
        """Read every run's diagnostics once, up front. Slow on the first
        call (the raw CSVs are large), instant on every call after."""
        import time
        t0 = time.time()
        for k in self.arms:
            for s, d in self.run_dirs(k):
                self.load_diag(d, n_bins, refresh=refresh)
        print(f"diagnostics cache ready ({time.time() - t0:.1f}s)")

    # Candidate panels, in priority order. Columns come from the agent's own
    # diagnostics dict (agents.sb3_fhr._FHR_base_diag + the SAC/SACD extras),
    # so a family only ever plots what its algorithm actually logged -
    # ent_coef/actor_loss are absent for DQN, c_spread only appears on the
    # state-conditioned c(s,a) arms.
    DIAG_CANDIDATES = (("penalty_weighted", "$\\lambda\\cdot$penalty (weighted)", "log", "fhr"),
                       ("td_loss", "critic TD loss", "log", "all"),
                       ("rho", "$\\rho=\\lambda\\cdot$penalty / TD", "log", "fhr"),
                       ("residual_rms", "recurrence residual RMS", "log", "fhr"),
                       ("companion_radius", "companion spectral radius", None, "fhr"),
                       ("sum_c", "$\\sum_i c_i$", None, "fhr"),
                       ("c_spread", "$c(s,a)$ spread across the batch", None, "fhr"),
                       ("ent_coef", "temperature $\\alpha$", "log", "all"),
                       ("actor_loss", "actor loss", None, "all"),
                       ("b_h", "penalty batch size", None, "fhr"),
                       ("nan_skips", "NaN skips", None, "fhr"))

    def available_diag_cols(self, n_bins=600):
        """Union of the diagnostics columns actually present, over one run per
        arm (plus the derived `rho` when its two inputs are there)."""
        cols = set()
        for k in self.arms:
            dirs = self.run_dirs(k)
            if dirs:
                d = self.load_diag(dirs[0][1], n_bins)
                if d:
                    cols |= set(d)
        if {"td_loss", "penalty_weighted"} <= cols:
            cols.add("rho")
        return cols

    def diag_panels(self, n_bins=600, limit=9):
        """DIAG_CANDIDATES filtered to what this family logged."""
        avail = self.available_diag_cols(n_bins)
        out = [pn for pn in self.DIAG_CANDIDATES
               if pn[0] in avail and (pn[3] != "fhr" or self.fhr_arms)]
        return out[:limit]

    def _diag_series(self, arm, col, n_bins):
        """Diagnostics column, with the derived `rho` handled here."""
        if col != "rho":
            return self.diag_curves(arm, col, n_bins)
        out = []
        for s, d in self.run_dirs(arm):
            data = self.load_diag(d, n_bins)
            if data is None or "td_loss" not in data or "penalty_weighted" not in data:
                continue
            td = np.asarray(data["td_loss"], float)
            pen = np.asarray(data["penalty_weighted"], float)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(td > 0, pen / td, np.nan)
            out.append((s, np.asarray(data["env_steps"], float), r))
        return out

    def fig_internal(self, col, title=None, scale=None, arms=None,
                     figsize=(6.2, 3.6), n_bins=600, ax=None, smooth=5):
        """One diagnostics column as its own figure - seed mean + band."""
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize)
        keys = arms or ([a.key for a in self.fhr_arms] +
                        ([self.baseline.key] if self.baseline else []))
        for k in keys:
            cs = self._diag_series(k, col, n_bins)
            if cs:
                self._plot_band(ax, k, cs, smooth=smooth, alpha_band=0.15,
                                zorder=1 if self.arms[k].is_baseline else 2)
        if scale:
            ax.set_yscale(scale)
        self._finish(ax, ylabel=title or col, title=title or col, legend="best")
        return fig

    def fig_internals(self, panels=None, n_bins=600, ncols=3,
                      panel_size=(4.3, 3.0), smooth=5):
        """The internals grid. Every panel is a seed-aggregated line + band
        over the cached down-sampling, so this is a few hundred points per
        curve instead of ~1e6 - it renders and re-renders instantly."""
        panels = panels if panels is not None else self.diag_panels(n_bins)
        if not panels:
            return _empty_fig("no train_diagnostics.csv columns available "
                              "for this family")
        ncols = min(ncols, len(panels))
        nrows = -(-len(panels) // ncols)
        fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                                 figsize=(panel_size[0] * ncols,
                                          panel_size[1] * nrows))
        for ax, (col, title, scale, who) in zip(axes.ravel(), panels):
            keys = ([a.key for a in self.fhr_arms] if who == "fhr"
                    else list(self.arms))
            drew = False
            for k in keys:
                cs = self._diag_series(k, col, n_bins)
                if cs:
                    self._plot_band(ax, k, cs, smooth=smooth, alpha_band=0.14,
                                    zorder=1 if self.arms[k].is_baseline else 2)
                    drew = True
            if not drew:
                ax.text(0.5, 0.5, f"{col}\nnot logged", ha="center",
                        va="center", transform=ax.transAxes, color="#888888")
                ax.set_axis_off()
                continue
            if scale:
                ax.set_yscale(scale)
            ax.set_title(title)
            ax.xaxis.set_major_formatter(_steps_formatter())
            ax.set_xlabel("environment steps")
        for ax in axes.ravel()[len(panels):]:
            ax.set_axis_off()
        # one figure-level legend over every arm, not just the arms that
        # happen to appear in the first panel (the baseline is absent from the
        # FHR-only panels but present in the TD-loss one)
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=a.colour, ls=a.ls,
                          lw=2.4 if a.is_baseline else 1.9, label=a.label)
                   for a in self.arms.values()]
        if handles:
            fig.legend(handles=handles, loc="outside lower center",
                       ncol=min(3, len(handles)))
        fig.suptitle(f"{self.env} - FHR + SAC internals "
                     f"({_band_caption(self.band)}, {n_bins}-point binning)")
        return fig

    def table_rho(self, n_bins=600, tail=0.5):
        """rho = lambda*penalty / TD over the training tail, per arm - the
        quantity the fetch_reach calibration pipeline solves for."""
        print(f"{'arm':30s} {'median TD':>12s} {'median L*pen':>14s} {'rho':>10s}")
        print("-" * 70)
        for k, a in self.arms.items():
            tds, pens = [], []
            for s, d in self.run_dirs(k):
                data = self.load_diag(d, n_bins)
                if data is None:
                    continue
                cut = int(len(data["env_steps"]) * (1 - tail))
                if "td_loss" in data:
                    tds.append(np.nanmedian(np.asarray(data["td_loss"], float)[cut:]))
                if "penalty_weighted" in data:
                    pens.append(np.nanmedian(np.asarray(data["penalty_weighted"], float)[cut:]))
            td = np.nanmean(tds) if tds else np.nan
            pen = np.nanmean(pens) if pens else np.nan
            rho = pen / td if td else np.nan
            print(f"{a.plain:30s} {td:12.4g} {pen:14.4g} {rho:10.3g}")

    # ----------------------------------------------------------------------
    # 5a - rollout Hankel rank of the CONVERGED policy
    # ----------------------------------------------------------------------
    def fig_rollout_hankel(self, runner, n_rollouts=3, base_seed=52, seed_idx=0,
                           n_show=30, figsize=(7.2, 3.4), arms=None):
        """Stacked per-rollout Hankels of the min-twin critic trace
        Q(s_t, pi(s_t)) and of the first action dimension of pi(s_t), for the
        converged policy of one seed per arm.

        `runner` is the experiment dir's run_sb3_seeds module (it owns
        load_run_model / _make_env). Returns (fig, table).
        """
        from analysis.low_rank.continuous_rollout import hankel_rollout_continuous
        from analysis.low_rank.rank import energy_rank
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        table = []
        for k in (arms or list(self.arms)):
            a = self.arms[k]
            dirs = self.run_dirs(k)
            if len(dirs) <= seed_idx:
                continue
            _, adapter = runner.load_run_model(dirs[seed_idx][1], device="cpu")
            env = runner._make_env(self.cfg)
            try:
                mats = hankel_rollout_continuous(adapter, env,
                                                 n_rollouts=n_rollouts,
                                                 base_seed=base_seed)
            finally:
                env.close()
            h_q, h_acts = mats[0], mats[1:]
            sv = np.linalg.svd(h_q, compute_uv=False)
            sv = sv / sv[0]
            rq = energy_rank(sv, 0.999)
            axes[0].semilogy(np.arange(1, min(len(sv), n_show) + 1), sv[:n_show],
                             color=a.colour, ls=a.ls,
                             lw=2.3 if a.is_baseline else 1.8,
                             label=f"{a.label} (rank {rq})")
            pi_ranks = []
            for j, h_a in enumerate(h_acts):
                sva = np.linalg.svd(h_a, compute_uv=False)
                sva = sva / sva[0]
                pi_ranks.append(energy_rank(sva, 0.999))
                if j == 0:
                    axes[1].semilogy(np.arange(1, min(len(sva), n_show) + 1),
                                     sva[:n_show], color=a.colour, ls=a.ls,
                                     lw=2.3 if a.is_baseline else 1.8,
                                     label=f"{a.label} (rank {pi_ranks[0]})")
            table.append((a.plain, rq, pi_ranks))
        axes[0].set(title="Hankel$(Q(s_t,\\pi(s_t)))$ - critic trace",
                    xlabel="singular-value index", ylabel="$\\sigma_i/\\sigma_1$")
        axes[1].set(title="Hankel$(\\pi(s_t)_0)$ - first action dim",
                    xlabel="singular-value index")
        for ax in axes:
            ax.legend(fontsize=7)
        fig.suptitle(f"{self.env} - rollout Hankel spectra of the final policy "
                     f"(seed {self.seeds[seed_idx]}, {n_rollouts} rollouts)")
        return fig, table

    def fig_value_hankel(self, runner, n_rollouts=3, base_seed=52, seed_idx=0,
                         n_show=30, figsize=(6.6, 4.0), arms=None):
        """Discrete-action counterpart of `fig_rollout_hankel`: Hankel of the
        min-twin Q(s_t, a_t) sequence along the greedy rollout.

        The claim under test is that the greedy rollout's Q sequence obeys a
        low-order linear recurrence, so FHR arms should concentrate the Hankel
        spectrum faster than the baseline. Returns (fig, table).
        """
        from analysis.low_rank.hankel_policy import (_hankel_from_sequence,
                                                     collect_hankel_sequences)
        from analysis.low_rank.rank import energy_rank
        fig, ax = plt.subplots(figsize=figsize)
        table = []
        for k in (arms or list(self.arms)):
            a = self.arms[k]
            dirs = self.run_dirs(k)
            if len(dirs) <= seed_idx:
                continue
            _, adapter = runner.load_run_model(dirs[seed_idx][1], device="cpu")
            adapter.epsilon = 0.0          # greedy rollout
            env = runner._make_env(self.cfg)
            spectra = []
            try:
                for j in range(n_rollouts):
                    seqs = collect_hankel_sequences(adapter, env,
                                                    seed=base_seed + j)
                    H = _hankel_from_sequence(seqs["Hankel Q"])
                    sv = np.linalg.svd(H, compute_uv=False)
                    spectra.append(sv / sv[0])
            finally:
                env.close()
            if not spectra:
                continue
            n = min(len(x) for x in spectra)
            S = np.stack([x[:n] for x in spectra]).mean(0)
            r = energy_rank(S, 0.999)
            ax.semilogy(np.arange(1, min(len(S), n_show) + 1), S[:n_show],
                        color=a.colour, ls=a.ls,
                        lw=2.3 if a.is_baseline else 1.8,
                        label=f"{a.label} (rank {r})")
            table.append((a.plain, r))
        if not table:
            plt.close(fig)
            return _empty_fig("no completed runs in this family yet"), []
        ax.set(title=f"{self.env} - Hankel(Q along the greedy rollout)",
               xlabel="singular-value index",
               ylabel="$\\sigma_i/\\sigma_1$")
        ax.legend(fontsize=7.5)
        return fig, table

    # ----------------------------------------------------------------------
    # 5 - penalised-window Hankel rank (the in-training probe)
    # ----------------------------------------------------------------------
    def window_metrics(self, arm):
        return arm_tick_metrics(self.run_dirs(arm))

    @property
    def has_window_probe(self):
        return any(self.window_metrics(k) for k in self.arms)

    def _window_curves(self, arm, key):
        return [(s, m["env_steps"], m[key]) for s, m in self.window_metrics(arm)
                if m["env_steps"].size]

    WINDOW_KEYS = (("rank999", "energy rank @ 99.9%", None),
                   ("pen_tail_ratio",
                    "penalty-block tail  $\\sigma_{r+1}/\\sigma_1$", "log"))

    def fig_window_rank(self, arm, keys=None, figsize=(6.8, 3.4), smooth=3):
        """Penalised-replay-window Hankel spectrum for one arm vs the
        baseline: rank measured where the penalty is applied, so the baseline
        curve is the control on identical windows."""
        keys = keys or self.WINDOW_KEYS
        fig, axes = plt.subplots(1, len(keys), figsize=figsize, squeeze=False)
        plot_keys = ([self.baseline.key]
                     if self.baseline and arm != self.baseline.key else []) + [arm]
        for ax, (mkey, title, scale) in zip(axes.ravel(), keys):
            for k in plot_keys:
                cs = self._window_curves(k, mkey)
                if cs:
                    self._plot_band(ax, k, cs, smooth=smooth, alpha_band=0.16,
                                    zorder=1 if self.arms[k].is_baseline else 2)
            if scale:
                ax.set_yscale(scale)
            ax.set_title(title)
            ax.xaxis.set_major_formatter(_steps_formatter())
            ax.set_xlabel("environment steps")
        axes[0, 0].legend(loc="best")
        fig.suptitle(f"{self.env} - {self.arms[arm].short} vs baseline, "
                     f"penalised-window Hankel spectrum")
        return fig

    def fig_window_rank_overlay(self, keys=None, figsize=(7.2, 3.6), smooth=3):
        keys = keys or self.WINDOW_KEYS
        fig, axes = plt.subplots(1, len(keys), figsize=figsize, squeeze=False)
        for ax, (mkey, title, scale) in zip(axes.ravel(), keys):
            for k in self.arms:
                cs = self._window_curves(k, mkey)
                if cs:
                    self._plot_band(ax, k, cs, smooth=smooth, alpha_band=0.12,
                                    zorder=1 if self.arms[k].is_baseline else 2)
            if scale:
                ax.set_yscale(scale)
            ax.set_title(title)
            ax.xaxis.set_major_formatter(_steps_formatter())
            ax.set_xlabel("environment steps")
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=a.colour, ls=a.ls,
                          lw=2.4 if a.is_baseline else 1.9, label=a.label)
                   for a in self.arms.values()]
        if handles:
            fig.legend(handles=handles, loc="outside lower center",
                       ncol=min(3, len(handles)))
        fig.suptitle(f"{self.env} - penalised-window Hankel spectrum, all arms")
        return fig

    def table_window_rank(self):
        """Final-quarter means of the window-spectrum metrics, per arm."""
        from analysis.low_rank.window_rank import final_quarter_summary
        print(f"{'arm':30s} {'rank@99.9%':>11s} {'rank@99%':>10s} "
              f"{'s2/s1':>8s} {'pen tail':>10s}   (final-quarter means)")
        print("-" * 76)
        any_row = False
        for k, a in self.arms.items():
            s = final_quarter_summary(self.window_metrics(k))
            if not s:
                continue
            any_row = True
            print(f"{a.plain:30s} {s['rank999']:11.2f} {s['rank99']:10.2f} "
                  f"{s['s2_s1']:8.4f} {s['pen_tail_ratio']:10.4f}")
        if not any_row:
            print("(no instrumented runs - see Family.window_probe_status())")

    def window_probe_status(self):
        """Why the window-rank figures are empty, when they are: the probe is
        an IN-TRAINING measurement, so it cannot be recovered from finished
        runs - the family has to be (re)launched with agent.window_rank_every
        > 0."""
        cfg_on = float(self.defaults.get("window_rank_every") or 0) > 0
        have = {k: len(self.window_metrics(k)) for k in self.arms}
        print(f"{self.config_path.name}: window_rank_every = "
              f"{self.defaults.get('window_rank_every', 'unset')}"
              f"  ({'ON' if cfg_on else 'OFF'})")
        for k, a in self.arms.items():
            print(f"  {a.plain:30s} {have[k]}/{len(self.run_dirs(k))} "
                  f"runs carry window_hankel.csv")
        if not any(have.values()):
            print("\nNo instrumented runs. The probe writes window_hankel.csv "
                  "during training only, so this cannot be back-filled from "
                  "the finished runs: set agent.window_rank_every (e.g. 5000) "
                  "and agent.window_rank_lags (16) in the config and relaunch "
                  "the family.")
        return have
