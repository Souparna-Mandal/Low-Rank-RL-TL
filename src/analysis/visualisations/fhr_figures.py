"""Publication-quality figures for the SB3 FHR comparison families (SAC, TD3
and the discrete DQN / QR-DQN / SAC-Discrete stacks).

One module behind every `exp*_results*.ipynb` in experiments/stable_baselines_3,
so every notebook plots its family the same way and a figure only has to be
fixed once. `experiments/stable_baselines_3/src/make_results_notebook.py`
generates the notebooks from one template; each saves EVERY panel as its own
file (fig_internals_individual, fig_rollout_hankel_single, single-key
fig_window_rank) so figures can be copy-pasted one at a time.

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
from matplotlib.ticker import FuncFormatter, MaxNLocator

from analysis.low_rank.window_rank import arm_tick_metrics

__all__ = ["set_pub_style", "load_family", "Family", "OKABE_ITO", "BAND",
           "PAPER", "cross_env_final", "iqm", "iqm_ci"]

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
PAPER = False         # True: Family.save strips titles from the SAVED files
                      # (the caption carries them), legends use the short
                      # paper labels without seed counts
BOOT_SEED = 0         # bootstrap RNG seed for the IQM confidence intervals
CLIP_WARMUP = True    # diagnostics panels start at algo.learning_starts, not 0
# One colour per (kind, order, lambda, predictor) across EVERY family loaded
# in a session, so the same arm has the same colour in every environment.
ARM_COLOUR_MAP = {}


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
        "grid.alpha": 0.35,
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
    step = float(step)
    if start is None:
        start = 2 * step
    out, t = [], float(start)
    while t <= top + 1e-9:
        out.append(float(round(t, 6)))
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
    if band == "iqm":
        # interquartile mean per grid point with a percentile bootstrap over
        # seeds (vectorised: B x n x G)
        rng = np.random.default_rng(BOOT_SEED)
        B = 300
        idx = rng.integers(0, n, (B, n))
        S = np.sort(Y[idx], axis=1)                     # (B, n, G)
        lo_i, hi_i = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
        mids = S[:, lo_i:hi_i, :].mean(axis=1) if hi_i > lo_i else S.mean(axis=1)
        centre = np.sort(Y, axis=0)[lo_i:hi_i].mean(axis=0) if hi_i > lo_i else Y.mean(0)
        return (grid, centre, np.percentile(mids, 2.5, axis=0),
                np.percentile(mids, 97.5, axis=0), n)
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
            "iqm": "IQM, 95% bootstrap CI",
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


def iqm(v):
    """Interquartile mean: the mean of the middle 50% of the sorted values
    (rliable's aggregate; with 5 seeds it drops the best and worst)."""
    v = np.sort(np.asarray(v, float))
    n = v.size
    if n == 0:
        return np.nan
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(v[lo:hi].mean()) if hi > lo else float(v.mean())


def iqm_ci(v, n_boot=2000, level=0.95, seed=None):
    """Percentile-bootstrap CI of the IQM over seeds -> (lo, hi)."""
    v = np.asarray(v, float)
    if v.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(BOOT_SEED if seed is None else seed)
    boots = np.array([iqm(rng.choice(v, v.size, replace=True))
                      for _ in range(n_boot)])
    a = (1 - level) / 2
    return (float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a)))


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
        if PAPER:
            return self.paper_label
        if self.is_baseline:
            return f"baseline (stock SB3 {self.algo_label})"
        return (f"FHR  $\\lambda$={self.lam:g}, r={self.order}"
                f"{self._tag(long=True)}")

    @property
    def paper_label(self):
        """The short label set for the paper: 'baseline', 'λ=1, r=8',
        'λ=1, r=2, theory c (frozen)', 'λ=0.1, r=2, c(s,a)', 'λ=0.1, r=2 + PER'."""
        if self.is_baseline:
            return "baseline"
        base = f"$\\lambda$={self.lam:g}, r={self.order}"
        if self.kind == "frozen":
            return base + ", theory c (frozen)"
        if self.kind == "ar":
            return base + ", c(s,a)"
        if self.kind == "per":
            return base + " + PER"
        return base

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
                           "qrdqn": "QR-DQN", "td3": "TD3"}.get(
                               self.algo, self.algo.upper())
        # continuous-action algos have a policy trace to Hankel per action
        # dim; the discrete ones have a categorical actor and use the greedy
        # value trajectory instead (fig_value_hankel).
        self.is_continuous = self.algo in ("sac", "td3")
        exp = self.cfg["experiment"]
        self.seeds = [str(s) for s in (exp.get("seeds") or [exp["seed"]])]
        self.defaults = self.cfg["agent"]
        self.figdir = self.root / "figures" / self.name
        self._diag_cache = {}
        self.arms, self.unrun = self._build_arms()
        self.baseline = next((a for a in self.arms.values() if a.is_baseline),
                             None)
        self.selected = None      # F.select([...]) restricts every figure/table
        self.gamma = float(self.cfg.get("algo", {}).get("gamma", 0.99))
        self.learning_starts = int(self.cfg.get("algo", {}).get("learning_starts", 0) or 0)
        solved = (self.cfg.get("training") or {}).get("solved_reward")
        self.solved = float(solved) if solved is not None else None

    # -- arm selection ------------------------------------------------------
    def select(self, arms=None):
        """Restrict EVERY figure and table to these arm keys (the baseline is
        always kept as the reference). None = all arms. Returns self."""
        if arms is None:
            self.selected = None
            return self
        unknown = [k for k in arms if k not in self.arms]
        if unknown:
            raise KeyError(f"unknown arm(s) {unknown}; have {list(self.arms)}")
        self.selected = list(arms)
        return self

    def _arms(self, arms=None, fhr_only=False):
        """Arm keys in plot order, honouring an explicit `arms` argument, else
        the family selection, else every arm."""
        keys = list(arms) if arms is not None else (
            [k for k in self.arms if self.selected is None or k in self.selected
             or self.arms[k].is_baseline])
        keys = [k for k in self.arms if k in keys]          # canonical order
        if fhr_only:
            keys = [k for k in keys if not self.arms[k].is_baseline]
        return keys

    @property
    def fhr_arms(self):
        return [self.arms[k] for k in self._arms(fhr_only=True)]

    @property
    def orders(self):
        return sorted({a.order for a in self.fhr_arms})

    @property
    def lambdas(self):
        return sorted({a.lam for a in self.fhr_arms})

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
        arms = {}
        for s in [s for s in specs if s["kind"] == "baseline"] + body:
            if s["kind"] == "baseline":
                colour = BASELINE_COLOUR
            else:
                # session-wide map: the same (kind, order, lambda, predictor)
                # gets the same colour in every family, in first-seen order
                ck = (s["kind"], s["order"], s["lam"], str(s["pred"] or ""))
                if ck not in ARM_COLOUR_MAP:
                    ARM_COLOUR_MAP[ck] = OKABE_ITO[len(ARM_COLOUR_MAP) % len(OKABE_ITO)]
                colour = ARM_COLOUR_MAP[ck]
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
            # same-length rolling mean (edges shrink the window) so the curve
            # starts at the first episode instead of `window` episodes in -
            # the start of training is where the sample-efficiency gap lives
            cs.append((s, x, _smooth(y, window)))
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
        lab = a.label if label is None else label
        if not PAPER:
            lab += f"  ({n} seeds)"
        ax.plot(grid, m, color=a.colour, ls=a.ls,
                lw=lw or (2.4 if a.is_baseline else 1.9), zorder=zorder,
                label=lab)
        return grid, m, n

    def _finish(self, ax, xlabel="environment steps", ylabel=None, title=None,
                legend="best"):
        ax.xaxis.set_major_formatter(_steps_formatter())
        ax.xaxis.set_major_locator(MaxNLocator(5))
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
        """Write one figure to <exp dir>/figures/<family>/<name>.{pdf,png}.
        With fhr_figures.PAPER = True the saved files carry no axes titles or
        suptitle (the caption does); the inline figure keeps them."""
        self.figdir.mkdir(parents=True, exist_ok=True)
        hidden = []
        if PAPER:
            for ax in fig.axes:
                if ax.get_title():
                    hidden.append((ax, ax.get_title()))
                    ax.set_title("")
            st = getattr(fig, "_suptitle", None)
            if st is not None and st.get_text():
                hidden.append((st, st.get_text()))
                st.set_text("")
        paths = []
        for ext in formats:
            p = self.figdir / f"{name}.{ext}"
            fig.savefig(p)
            paths.append(p)
        for obj, text in hidden:
            (obj.set_title if hasattr(obj, "set_title") else obj.set_text)(text)
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

    def _solved_line(self, ax, axis="y"):
        if self.solved is None:
            return
        (ax.axhline if axis == "y" else ax.axvline)(
            self.solved, color="#777777", lw=0.9, ls=(0, (1, 2)), zorder=0)

    def fig_eval(self, arm, figsize=(6.2, 3.9), ax=None, title=None, smooth=1,
                 thresholds=None):
        """Greedy-eval curve for one arm against the baseline. `thresholds`
        draws the ladder and first-crossing dots on the eval stream (the
        eval-stream sample-efficiency read of the older notebooks)."""
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize)
        keys = ([self.baseline.key] if self.baseline and arm != self.baseline.key
                else []) + [arm]
        drawn = {}
        for k in keys:
            got = self._plot_band(ax, k, self.eval_curves(k), smooth=smooth,
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
        self._solved_line(ax)
        if not drawn:
            plt.close(fig)
            return _empty_fig(f"no completed runs for {self.arms[arm].plain}")
        self._finish(ax, ylabel="greedy evaluation return",
                     title=title or f"{self.env} - {self.arms[arm].short} "
                                    f"vs baseline (greedy eval)",
                     legend="lower right")
        return fig

    def fig_overlay(self, which="eval", arms=None, window=50, smooth=1,
                    figsize=(6.6, 4.1), title=None, band=None):
        """Every arm on one axes - the summary panel, not the per-arm ones.
        `band` overrides the seed band ("sem" | "ci95" | "iqr" | "iqm" |
        "minmax" | "none") for this figure."""
        fig, ax = plt.subplots(figsize=figsize)
        keys = self._arms(arms)
        for k in keys:
            cs = (self.eval_curves(k) if which == "eval"
                  else self.train_curves(k, window))
            self._plot_band(ax, k, cs, smooth=smooth, band=band,
                            zorder=1 if self.arms[k].is_baseline else 2)
        if not ax.lines:
            plt.close(fig)
            return _empty_fig("no completed runs in this family yet")
        if which == "eval":
            self._solved_line(ax)
        ylab = ("greedy evaluation return" if which == "eval"
                else f"training return (rolling {window} eps)")
        self._finish(ax, ylabel=ylab,
                     title=title or f"{self.env} - all arms ({which})",
                     legend="lower right")
        return fig

    # ----------------------------------------------------------------------
    # 2 - sample efficiency, measured on the TRAINING stream
    # ----------------------------------------------------------------------
    def auto_thresholds(self, window=50, n_target=6, step=None, start=None,
                        arms=None, stream="train"):
        """A round ladder of training-return thresholds for the family.
        Positive-return envs get nice_thresholds(top); negative-return envs
        (Pendulum: -1400 -> -140) get the same round ladder laid over the span
        from the worst seed-mean level to the top, so the sample-efficiency
        section is not empty there."""
        tops, lows = [], []
        for k in self._arms(arms):
            agg = _aggregate(self._curves(k, stream, window), band="none")
            if agg:
                tops.append(np.nanmax(agg[1]))
                lows.append(np.nanmin(agg[1]))
        if not tops:
            return []
        top = max(tops)
        if top > 0:
            return nice_thresholds(top, n_target=n_target, step=step, start=start)
        lo, span = float(min(lows)), float(top - min(lows))
        if not np.isfinite(span) or span <= 0:
            return []
        if step is None:
            raw = span / (n_target + 1)
            mag = 10.0 ** np.floor(np.log10(raw))
            step = min((m * mag for m in (1, 2, 2.5, 5, 10)),
                       key=lambda s_: abs(s_ - raw))
        # multiples of the step, starting two steps above the worst level
        first = (np.ceil((lo + 2 * step) / step) * step if start is None
                 else float(start))
        out, t = [], float(first)
        while t <= top + 1e-9:
            out.append(float(round(t, 6)))
            t += step
        return out

    def steps_to_thresholds(self, thresholds, window=50, arms=None,
                            stream="train"):
        """{arm key: {threshold: array of per-seed crossing steps}} on the
        rolling-mean training curve (stream="train", the toolkit's rule) or on
        the greedy-eval curve (stream="eval", the older notebooks' read); nan
        where a seed never gets there."""
        out = {}
        for k in self._arms(arms):
            cs = self._curves(k, stream, window)
            out[k] = {t: np.array([first_cross(x, y, t) for _, x, y in cs])
                      for t in thresholds}
        return out

    def table_sample_efficiency(self, thresholds=None, window=50, arms=None,
                                stream="train"):
        """Printed table: env steps to first reach each threshold on the
        training stream (seed mean, and how many seeds ever got there);
        stream="eval" reads the greedy-eval curve instead."""
        thresholds = thresholds if thresholds is not None else self.auto_thresholds(window, stream=stream)
        xs = self.steps_to_thresholds(thresholds, window, arms, stream)
        hdr = "  ".join(f"{_fmt_thr(t):>10s}" for t in thresholds)
        what = (f"training stream (rolling {window} eps)" if stream == "train"
                else "greedy-eval stream")
        print(f"steps to reach, {what} - "
              f"{self.env}, seeds {', '.join(self.seeds)}")
        print(f"{'arm':30s} {'final eval':>11s}   {hdr}")
        print("-" * (44 + 12 * len(thresholds)))
        for k in xs:
            a = self.arms[k]
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
                               seed_dots=True, arms=None, stream="train"):
        """The sample-efficiency claim as one figure: env steps needed to
        first reach each return threshold, per arm, from the training stream.

        A filled marker on the solid line means EVERY seed reached that
        threshold. An open marker means only some did - the mean there is over
        the seeds that made it, which biases it downwards, so those points are
        deliberately left off the line rather than allowed to bend it.
        """
        thresholds = thresholds if thresholds is not None else self.auto_thresholds(window, stream=stream)
        xs = self.steps_to_thresholds(thresholds, window, arms, stream)
        fig, ax = plt.subplots(figsize=figsize)
        pos = np.arange(len(thresholds), dtype=float)
        rng = np.random.default_rng(0)
        partial_seen = False
        for i, k in enumerate(xs):
            a = self.arms[k]
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
        what = "training stream" if stream == "train" else "greedy-eval stream"
        ax.set_title(f"{self.env} - sample efficiency, {what} "
                     f"({_band_caption('sem')}){sub}", fontsize=10)
        ax.legend(loc="upper left")
        return fig

    def fig_speedup(self, thresholds=None, window=50, figsize=(6.6, 3.9),
                    arms=None, stream="train"):
        """Sample-efficiency ratio: baseline steps / arm steps at each
        threshold. > 1 means the arm got there in fewer environment steps."""
        thresholds = thresholds if thresholds is not None else self.auto_thresholds(window, stream=stream)
        keys = self._arms(arms)
        if self.baseline and self.baseline.key not in keys:
            keys = [self.baseline.key] + keys
        xs = self.steps_to_thresholds(thresholds, window, keys, stream)
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
        arms = [self.arms[k] for k in keys if not self.arms[k].is_baseline]
        w = 0.8 / max(1, len(arms))
        span = []
        for i, a in enumerate(arms):
            mu = np.array([_full(a.key, t) for t in thresholds])
            ratio = base / mu
            ax.bar(pos + (i - (len(arms) - 1) / 2) * w, ratio - 1.0, bottom=1.0,
                   width=w * 0.92, color=a.colour, alpha=0.55 if a.frozen else 0.9,
                   label=a.label, edgecolor=a.colour if a.frozen else "white",
                   linestyle="--" if a.frozen else "-", linewidth=0.9 if a.frozen else 0.4)
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
        else:
            arms = [self.arms[k] if isinstance(k, str) else k for k in arms]
            arms = [a for a in arms if not a.is_baseline]
        rows = ([self.baseline] if self.baseline else []) + list(arms)
        rows = [(a, self.finals(a.key, mode)) for a in rows]
        rows = [(a, v) for a, v in rows if v.size]
        if not rows:
            return _empty_fig("no completed runs for this selection")
        fig, ax = plt.subplots(figsize=figsize)
        ypos = np.arange(len(rows))[::-1]
        base = next((v.mean() for a, v in rows if a.is_baseline), np.nan)
        hi = max(max(v.max(), v.mean()) for _, v in rows)
        lo = min(min(v.min(), v.mean()) for _, v in rows)
        span = max(hi, 0) - min(lo, 0)
        positive = hi >= 0
        for y, (a, v) in zip(ypos, rows):
            sem = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0
            ax.barh(y, v.mean(), color=a.colour, alpha=0.55 if a.frozen else 0.88,
                    height=0.6, edgecolor=a.colour if a.frozen else "white",
                    linestyle="--" if a.frozen else "-",
                    linewidth=0.9 if a.frozen else 0.5, zorder=2)
            ax.errorbar(v.mean(), y, xerr=sem, color="#111111", lw=1.2,
                        capsize=3, zorder=4)
            ax.plot(v, np.full(v.size, y), "o", ms=3.2, mfc="white",
                    mec="#111111", mew=0.7, ls="none", zorder=5)
            delta = "" if a.is_baseline or not np.isfinite(base) else \
                f"  ({v.mean() / base - 1:+.1%})"
            # one aligned label column past the widest seed dot (to the right
            # of the bars for positive returns, of zero for negative ones)
            xlab = (max(hi, 0) + 0.04 * span)
            ax.text(xlab, y, f"{v.mean():.0f}{delta}", va="center", fontsize=8.5)
        ax.set_yticks(ypos, [a.short for a, _ in rows])
        ax.set_xlabel("final greedy-eval return")
        self._solved_line(ax, axis="x")
        ax.set_xlim(min(lo, 0) - 0.05 * span if not positive else 0,
                    max(hi, 0) + 0.38 * span)
        ttl = (f"{self.env} - final return" if lam is None else
               f"{self.env} - final return, $\\lambda$ = {lam:g} vs baseline")
        ax.set_title(ttl)
        ax.grid(axis="y", visible=False)
        return fig

    def fig_grid(self, mode="last", figsize=(4.6, 3.6), arms=None):
        """lambda x order heat map of final return relative to the baseline."""
        fhr = [self.arms[k] for k in self._arms(arms, fhr_only=True)]
        lams = sorted({a.lam for a in fhr})
        orders = sorted({a.order for a in fhr})
        grid = np.full((len(lams), len(orders)), np.nan)
        for a in fhr:
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
    # diagnostics dict (agents.sb3_fhr._fhr_base_diag + the SAC/SACD/TD3
    # extras), so a family only ever plots what its algorithm actually logged:
    # ent_coef is absent for DQN and TD3, c_spread only appears on the
    # state-conditioned c(s,a) arms, and the stream ratios (loss_ratio /
    # rho_loss from the penalty, grad_* from the gradient probe,
    # agent.grad_probe_every > 0) only on families launched with them. The
    # unweighted ratios are plotted for the baseline too - on it they are the
    # calibration signal (lambda* = target / ratio) measured for free.
    DIAG_CANDIDATES = (("penalty_raw", "recurrence penalty (raw, unweighted)", "log", "all"),
                       ("penalty_weighted", "$\\lambda\\cdot$penalty (weighted)", "log", "fhr"),
                       ("td_loss", "critic TD loss", "log", "all"),
                       ("rho", "$\\rho_{loss}=\\lambda\\cdot$penalty / TD", "log", "fhr"),
                       ("grad_rho", "$\\rho_{grad}=\\lambda\\,\\|\\nabla$pen$\\|/\\|\\nabla$TD$\\|$", "log", "fhr"),
                       ("loss_ratio", "penalty / TD (unweighted)", "log", "all"),
                       ("grad_ratio", "$\\|\\nabla$pen$\\|/\\|\\nabla$TD$\\|$ (unweighted)", "log", "all"),
                       ("grad_cos", "cos$(\\nabla$TD$,\\nabla$pen$)$ - stream alignment", None, "all"),
                       ("grad_norm_td", "$\\|\\nabla_\\theta$TD$\\|$", "log", "all"),
                       ("grad_norm_pen", "$\\|\\nabla_\\theta$penalty$\\|$", "log", "all"),
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
        for k in self._arms():
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
        """Diagnostics column, with `rho` handled here: the logged rho_loss
        when the family carries it, else derived from its two inputs."""
        if col != "rho":
            return self.diag_curves(arm, col, n_bins)
        logged = self.diag_curves(arm, "rho_loss", n_bins)
        if logged:
            return logged
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

    # reference levels drawn on a diagnostics panel: (label, value-callable)
    def diag_refs(self, col):
        g = self.gamma
        return {"companion_radius": [("$1/\\gamma$ (theory)", 1.0 / g),
                                     ("1 (contractive boundary)", 1.0)],
                "sum_c": [("theory $\\sum c$ = 1", 1.0)],
                "grad_cos": [("0", 0.0)]}.get(col, [])

    # columns whose per-step noise needs a wider rolling mean
    NOISY_DIAG = ("grad_cos", "grad_ratio", "grad_rho", "grad_norm_td",
                  "grad_norm_pen", "loss_ratio", "rho", "rho_loss")

    def fig_internal(self, col, title=None, scale=None, arms=None,
                     figsize=(6.2, 3.6), n_bins=600, ax=None, smooth=None,
                     ylabel=None, refs=True):
        """One diagnostics column as its own figure - seed mean + band, with
        the theory / boundary reference levels (dotted) where they exist and
        the x-axis clipped to after algo.learning_starts (CLIP_WARMUP)."""
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize)
        keys = self._arms(arms)
        smooth = (25 if col in self.NOISY_DIAG else 5) if smooth is None else smooth
        for k in keys:
            cs = self._diag_series(k, col, n_bins)
            if cs:
                self._plot_band(ax, k, cs, smooth=smooth, alpha_band=0.12,
                                zorder=1 if self.arms[k].is_baseline else 2)
        if not ax.lines:
            plt.close(fig)
            return _empty_fig(f"{col}: not logged by any completed run")
        if scale:
            ax.set_yscale(scale)
        if refs:
            for lab, val in self.diag_refs(col):
                ax.axhline(val, color="#555555", lw=0.9, ls=(0, (2, 2)), zorder=0)
                ax.annotate(lab, (1.0, val), xycoords=("axes fraction", "data"),
                            xytext=(-4, 3), textcoords="offset points",
                            ha="right", fontsize=7, color="#555555")
        if CLIP_WARMUP and self.learning_starts > 0:
            ax.set_xlim(left=self.learning_starts)
        self._finish(ax, ylabel=ylabel or title or col, title=title or col,
                     legend="best")
        return fig

    def fig_internals_individual(self, n_bins=600, smooth=None, limit=None,
                                 figsize=(6.2, 3.6), arms=None):
        """The internals as ONE FIGURE PER PANEL - the copy-pastable form of
        fig_internals. Yields (column, fig) over diag_panels(limit=None), so
        a family plots exactly the diagnostics it logged, FHR-only panels
        over the FHR arms and the rest over every arm including the
        baseline."""
        for col, title, scale, who in self.diag_panels(n_bins, limit=limit):
            keys = self._arms(arms, fhr_only=(who == "fhr"))
            yield col, self.fig_internal(col, title=f"{self.env} - {title}",
                                         ylabel=title, scale=scale, arms=keys,
                                         n_bins=n_bins, smooth=smooth,
                                         figsize=figsize)

    def fig_internals(self, panels=None, n_bins=600, ncols=3,
                      panel_size=(4.3, 3.0), smooth=5, arms=None):
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
            keys = self._arms(arms, fhr_only=(who == "fhr"))
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
                   for a in (self.arms[k] for k in self._arms(arms))]
        if handles:
            fig.legend(handles=handles, loc="outside lower center",
                       ncol=min(3, len(handles)))
        fig.suptitle(f"{self.env} - FHR + {self.algo_label} internals "
                     f"({_band_caption(self.band)}, {n_bins}-point binning)")
        return fig

    def table_rho(self, n_bins=600, tail=0.5, arms=None):
        """rho = lambda*penalty / TD over the training tail, per arm - the
        quantity the fetch_reach calibration pipeline solves for."""
        print(f"{'arm':30s} {'median TD':>12s} {'median pen':>12s} "
              f"{'median L*pen':>14s} {'rho':>10s}")
        print("-" * 84)
        for k in self._arms(arms):
            a = self.arms[k]
            tds, raws, pens = [], [], []
            for s, d in self.run_dirs(k):
                data = self.load_diag(d, n_bins)
                if data is None:
                    continue
                cut = int(len(data["env_steps"]) * (1 - tail))
                if "td_loss" in data:
                    tds.append(np.nanmedian(np.asarray(data["td_loss"], float)[cut:]))
                if "penalty_raw" in data:
                    v = np.asarray(data["penalty_raw"], float)[cut:]
                    if np.isfinite(v).any():
                        raws.append(np.nanmedian(v))
                if "penalty_weighted" in data:
                    pens.append(np.nanmedian(np.asarray(data["penalty_weighted"], float)[cut:]))
            td = np.nanmean(tds) if tds else np.nan
            raw = np.nanmean(raws) if raws else np.nan
            pen = np.nanmean(pens) if pens else np.nan
            rho = pen / td if td else np.nan
            print(f"{a.plain:30s} {td:12.4g} {raw:12.4g} {pen:14.4g} {rho:10.3g}")

    # the two streams' ratios: loss-side (from the penalty) and gradient-side
    # (from the probe), unweighted and lambda-weighted, plus their alignment
    STREAM_COLS = (("loss_ratio", "pen/TD"), ("rho_loss", "rho_loss"),
                   ("grad_ratio", "|g_pen|/|g_TD|"), ("grad_rho", "rho_grad"),
                   ("grad_cos", "cos"))

    def stream_medians(self, arm, tail=0.5, n_bins=600):
        """Seed-mean of the per-seed tail medians of every stream column
        (nan where the family did not log it)."""
        vals = {c: [] for c, _ in self.STREAM_COLS}
        for _, d in self.run_dirs(arm):
            data = self.load_diag(d, n_bins)
            if data is None:
                continue
            cut = int(len(data["env_steps"]) * (1 - tail))
            for c, _ in self.STREAM_COLS:
                if c in data:
                    v = np.asarray(data[c], float)[cut:]
                    if np.isfinite(v).any():
                        vals[c].append(float(np.nanmedian(v)))
        return {c: (float(np.mean(v)) if v else np.nan) for c, v in vals.items()}

    def table_rho_streams(self, tail=0.5, targets=(0.1, 0.5, 1.0), n_bins=600,
                          arms=None):
        """The hyper-parameter-selection table: loss- and gradient-stream
        ratios over the training tail per arm, and - from the BASELINE's
        unweighted measurements, where the penalty never entered the loss -
        the lambda that would put each target ratio on the critic. Returns
        {arm: medians} (+ "lambda_for" when a baseline is present)."""
        hdr = "  ".join(f"{lab:>14s}" for _, lab in self.STREAM_COLS)
        print(f"stream ratios over the last {tail:.0%} of training - "
              f"{self.env}, seeds {', '.join(self.seeds)}")
        print(f"{'arm':30s}  {hdr}")
        print("-" * (32 + 16 * len(self.STREAM_COLS)))
        out = {}
        for k in self._arms(arms):
            a = self.arms[k]
            m = self.stream_medians(k, tail, n_bins)
            out[k] = m
            cells = "  ".join(f"{m[c]:14.4g}" if np.isfinite(m[c]) else f"{'-':>14s}"
                              for c, _ in self.STREAM_COLS)
            print(f"{a.plain:30s}  {cells}")
        if self.baseline:
            base = out[self.baseline.key]
            print(f"\nlambda for a target ratio, from the baseline's unweighted "
                  f"streams (lambda* = target / ratio):")
            print(f"{'target':>8s} {'by gradients':>14s} {'by losses':>12s}")
            out["lambda_for"] = {}
            for t in targets:
                lg = t / base["grad_ratio"] if np.isfinite(base["grad_ratio"]) and base["grad_ratio"] > 0 else np.nan
                ll = t / base["loss_ratio"] if np.isfinite(base["loss_ratio"]) and base["loss_ratio"] > 0 else np.nan
                out["lambda_for"][t] = {"grad": lg, "loss": ll}
                print(f"{t:8.3g} {lg:14.4g} {ll:12.4g}")
        return out

    # ----------------------------------------------------------------------
    # 5a - rollout Hankel rank of the CONVERGED policy
    # ----------------------------------------------------------------------
    def _rollout_spectra(self, runner, n_rollouts=3, base_seed=52, seed_idx=0,
                         arms=None):
        """[(arm key, sv_q, rank_q, [sv per action dim], [rank per dim])] for
        the converged policy of one seed per arm - cached, since it reloads
        checkpoints and steps the env."""
        from analysis.low_rank.continuous_rollout import hankel_rollout_continuous
        from analysis.low_rank.rank import energy_rank
        keys = self._arms(arms)
        ck = (n_rollouts, base_seed, seed_idx, tuple(keys))
        cache = self.__dict__.setdefault("_rollout_cache", {})
        if ck in cache:
            return cache[ck]
        out = []
        for k in keys:
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
            svas = []
            for h_a in h_acts:
                sva = np.linalg.svd(h_a, compute_uv=False)
                svas.append(sva / sva[0])
            out.append((k, sv, energy_rank(sv, 0.999), svas,
                        [energy_rank(x, 0.999) for x in svas]))
        cache[ck] = out
        return out

    def _plot_spectra(self, ax, spectra, which, n_show, dim=0, floor=1e-3):
        orders = set()
        for k, sv, rq, svas, pr in spectra:
            a = self.arms[k]
            if which == "q":
                s_, r_ = sv, rq
            elif dim < len(svas):
                s_, r_ = svas[dim], pr[dim]
            else:
                continue
            ax.semilogy(np.arange(1, min(len(s_), n_show) + 1),
                        np.maximum(s_[:n_show], floor), color=a.colour, ls=a.ls,
                        lw=2.3 if a.is_baseline else 1.8,
                        label=f"{a.label} (rank {r_})")
            if not a.is_baseline:
                orders.add(a.order)
        for r in sorted(orders):      # the penalty order(s) present
            ax.axvline(r, color="#888888", lw=0.7, ls=(0, (2, 2)), zorder=0)
            ax.annotate(f"r={r}", (r, 1.0), xytext=(2, -8), textcoords="offset points",
                        fontsize=7, color="#666666")
        ax.set_xlim(0.5, n_show + 0.5)
        ax.set_ylim(bottom=floor)

    def fig_rollout_hankel(self, runner, n_rollouts=3, base_seed=52, seed_idx=0,
                           n_show=30, figsize=(7.2, 3.4), arms=None):
        """Stacked per-rollout Hankels of the min-twin critic trace
        Q(s_t, pi(s_t)) and of the first action dimension of pi(s_t), for the
        converged policy of one seed per arm, side by side. Returns
        (fig, table); fig_rollout_hankel_single gives each panel on its own.

        `runner` is the experiment dir's run_sb3_seeds module (it owns
        load_run_model / _make_env).
        """
        spectra = self._rollout_spectra(runner, n_rollouts, base_seed,
                                        seed_idx, arms)
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        self._plot_spectra(axes[0], spectra, "q", n_show)
        self._plot_spectra(axes[1], spectra, "pi", n_show, 0)
        axes[0].set(title="Hankel$(Q(s_t,\\pi(s_t)))$ - critic trace",
                    xlabel="singular-value index", ylabel="$\\sigma_i/\\sigma_1$")
        axes[1].set(title="Hankel$(\\pi(s_t)_0)$ - first action dim",
                    xlabel="singular-value index")
        for ax in axes:
            if ax.lines:
                ax.legend(fontsize=7)
        fig.suptitle(f"{self.env} - rollout Hankel spectra of the final policy "
                     f"(seed {self.seeds[seed_idx]}, {n_rollouts} rollouts)")
        table = [(self.arms[k].plain, rq, pr) for k, _, rq, _, pr in spectra]
        return fig, table

    def table_rollout_hankel(self, runner, n_rollouts=3, base_seed=52,
                             seed_idx=0, arms=None):
        """Printed energy ranks (99.9%) of the final policy's rollout Hankels:
        the critic trace and every action dimension, per arm."""
        spectra = self._rollout_spectra(runner, n_rollouts, base_seed,
                                        seed_idx, arms)
        print(f"{'arm':30s} rank(Q)  ranks(pi dims)   (energy rank @ 99.9%, "
              f"seed {self.seeds[seed_idx]}, {n_rollouts} rollouts)")
        print("-" * 72)
        for k, _, rq, _, pr in spectra:
            print(f"{self.arms[k].plain:30s} {rq:7d}  {pr}")
        if not spectra:
            print("(no completed runs in this family yet)")
        return [(self.arms[k].plain, rq, pr) for k, _, rq, _, pr in spectra]

    def fig_rollout_hankel_single(self, runner, which="q", dim=0, n_rollouts=3,
                                  base_seed=52, seed_idx=0, n_show=30,
                                  figsize=(6.2, 3.6), arms=None):
        """One rollout-Hankel spectrum as its own figure: which="q" for the
        critic trace Q(s_t, pi(s_t)), which="pi" (+ dim) for one action
        dimension of pi(s_t)."""
        spectra = self._rollout_spectra(runner, n_rollouts, base_seed,
                                        seed_idx, arms)
        fig, ax = plt.subplots(figsize=figsize)
        self._plot_spectra(ax, spectra, which, n_show, dim)
        if not ax.lines:
            plt.close(fig)
            return _empty_fig("no completed runs in this family yet")
        if which == "q":
            title = (f"{self.env} - Hankel$(Q(s_t,\\pi(s_t)))$ of the final "
                     f"policy (critic trace)")
        else:
            title = (f"{self.env} - Hankel$(\\pi(s_t)_{{{dim}}})$ of the final "
                     f"policy (action dim {dim})")
        ax.set(title=title, xlabel="singular-value index",
               ylabel="$\\sigma_i/\\sigma_1$")
        ax.legend(fontsize=7.5)
        return fig

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
        """Penalised-replay-window Hankel spectrum for one arm (or a list of
        arms, e.g. one lambda's frozen + learned pair) vs the baseline: rank
        measured where the penalty is applied, so the baseline curve is the
        control on identical windows."""
        keys = keys or self.WINDOW_KEYS
        arm_keys = [arm] if isinstance(arm, str) else list(arm)
        fig, axes = plt.subplots(1, len(keys), figsize=figsize, squeeze=False)
        plot_keys = ([self.baseline.key] if self.baseline
                     and self.baseline.key not in arm_keys else []) + arm_keys
        for ax, (mkey, title, scale) in zip(axes.ravel(), keys):
            for k in plot_keys:
                cs = self._window_curves(k, mkey)
                if cs:
                    self._plot_band(ax, k, cs, smooth=smooth, alpha_band=0.16,
                                    zorder=1 if self.arms[k].is_baseline else 2)
            if scale:
                ax.set_yscale(scale)
            ax.set_title(title)
            ax.set_ylabel(title)
            ax.xaxis.set_major_formatter(_steps_formatter())
            ax.set_xlabel("environment steps")
        if axes[0, 0].lines:
            axes[0, 0].legend(loc="best")
        who = ", ".join(self.arms[k].short for k in arm_keys)
        if len(keys) == 1:      # single panel: one title, no suptitle
            axes[0, 0].set_title(f"{self.env} - {who} vs baseline, "
                                 f"penalised windows\n{keys[0][1]}")
        else:
            fig.suptitle(f"{self.env} - {who} vs baseline, "
                         f"penalised-window Hankel spectrum")
        return fig

    def fig_window_rank_overlay(self, keys=None, figsize=(7.2, 3.6), smooth=3,
                                arms=None):
        keys = keys or self.WINDOW_KEYS
        sel = self._arms(arms)
        fig, axes = plt.subplots(1, len(keys), figsize=figsize, squeeze=False)
        for ax, (mkey, title, scale) in zip(axes.ravel(), keys):
            for k in sel:
                cs = self._window_curves(k, mkey)
                if cs:
                    self._plot_band(ax, k, cs, smooth=smooth, alpha_band=0.12,
                                    zorder=1 if self.arms[k].is_baseline else 2)
            if scale:
                ax.set_yscale(scale)
            ax.set_title(title)
            ax.set_ylabel(title)
            ax.xaxis.set_major_formatter(_steps_formatter())
            ax.set_xlabel("environment steps")
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=a.colour, ls=a.ls,
                          lw=2.4 if a.is_baseline else 1.9, label=a.label)
                   for a in (self.arms[k] for k in sel)]
        if handles:
            fig.legend(handles=handles, loc="outside lower center",
                       ncol=min(3, len(handles)))
        if len(keys) == 1:
            axes[0, 0].set_title(f"{self.env} - penalised windows, all arms"
                                 f"\n{keys[0][1]}")
        else:
            fig.suptitle(f"{self.env} - penalised-window Hankel spectrum, all arms")
        return fig

    def table_window_rank(self, arms=None):
        """Final-quarter means of the window-spectrum metrics, per arm."""
        from analysis.low_rank.window_rank import final_quarter_summary
        print(f"{'arm':30s} {'rank@99.9%':>11s} {'rank@99%':>10s} "
              f"{'s2/s1':>8s} {'pen tail':>10s}   (final-quarter means)")
        print("-" * 76)
        any_row = False
        for k in self._arms(arms):
            a = self.arms[k]
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

    # ----------------------------------------------------------------------
    # 8 - ICLR figures: per-seed detail, escape rate, IQM strip, rho vs gain,
    #     the learned recurrence, compression vs return
    # ----------------------------------------------------------------------
    def _curves(self, arm, which="eval", window=50):
        return (self.eval_curves(arm) if which == "eval"
                else self.train_curves(arm, window))

    def fig_seed_curves_arm(self, arm, which="eval", window=50, smooth=1,
                            figsize=(6.2, 3.6), ax=None, title=None):
        """One arm's individual seeds (thin) + seed mean (bold), with the
        baseline seed mean as a grey reference - the view that shows collapse
        and escape events the mean +- band hides."""
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize)
        a = self.arms[arm]
        if self.baseline and not a.is_baseline:
            agg = _aggregate(self._curves(self.baseline.key, which, window), band="none")
            if agg:
                ax.plot(agg[0], _smooth(agg[1], smooth), color=BASELINE_COLOUR,
                        lw=1.6, ls=(0, (4, 2)), alpha=0.8, zorder=1,
                        label="baseline (seed mean)")
        cs = self._curves(arm, which, window)
        for s, x, y in cs:
            ax.plot(x, _smooth(y, smooth), color=a.colour, lw=0.9, alpha=0.5,
                    zorder=2, label=f"seed {s}")
        agg = _aggregate(cs, band="none")
        if agg:
            ax.plot(agg[0], _smooth(agg[1], smooth), color=a.colour, ls=a.ls,
                    lw=2.4, zorder=3, label=f"{a.label} (mean)")
        if not ax.lines:
            plt.close(fig)
            return _empty_fig(f"no completed runs for {a.plain}")
        ylab = ("greedy evaluation return" if which == "eval"
                else f"training return (rolling {window} eps)")
        self._finish(ax, ylabel=ylab,
                     title=title or f"{self.env} - {a.short}, every seed ({which})",
                     legend="lower right")
        return fig

    def fig_seed_curves(self, which="eval", arms=None, window=50, smooth=1,
                        ncols=3, panel_size=(3.5, 2.5), sharey=True):
        """Small multiples: one panel per arm, thin per-seed lines + bold seed
        mean, baseline mean dashed in grey on every panel, shared y-axis."""
        keys = self._arms(arms)
        if not keys:
            return _empty_fig("no arms selected")
        ncols = min(ncols, len(keys))
        nrows = -(-len(keys) // ncols)
        fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharey=sharey,
                                 figsize=(panel_size[0] * ncols,
                                          panel_size[1] * nrows))
        base_agg = (_aggregate(self._curves(self.baseline.key, which, window),
                               band="none") if self.baseline else None)
        for ax, k in zip(axes.ravel(), keys):
            a = self.arms[k]
            if base_agg and not a.is_baseline:
                ax.plot(base_agg[0], _smooth(base_agg[1], smooth),
                        color=BASELINE_COLOUR, lw=1.3, ls=(0, (4, 2)),
                        alpha=0.75, zorder=1)
            cs = self._curves(k, which, window)
            for _, x, y in cs:
                ax.plot(x, _smooth(y, smooth), color=a.colour, lw=0.8,
                        alpha=0.5, zorder=2)
            agg = _aggregate(cs, band="none")
            if agg:
                ax.plot(agg[0], _smooth(agg[1], smooth), color=a.colour,
                        ls=a.ls, lw=2.2, zorder=3)
            ax.set_title(a.short, fontsize=9)
            ax.xaxis.set_major_formatter(_steps_formatter())
        for ax in axes.ravel()[len(keys):]:
            ax.set_axis_off()
        for ax in axes[-1, :]:
            ax.set_xlabel("environment steps")
        ylab = ("greedy eval return" if which == "eval"
                else f"training return (rolling {window})")
        for ax in axes[:, 0]:
            ax.set_ylabel(ylab)
        fig.suptitle(f"{self.env} - every seed per arm ({which}); grey dashed = "
                     f"baseline seed mean", fontsize=10)
        return fig

    def escape_curves(self, threshold, which="eval", arms=None, window=50,
                      n_grid=N_GRID):
        """{arm: (grid, fraction of seeds whose curve is >= threshold)}."""
        out = {}
        for k in self._arms(arms):
            cs = [(s, np.asarray(x, float), np.asarray(y, float))
                  for s, x, y in self._curves(k, which, window) if len(x) > 1]
            if not cs:
                continue
            lo = max(x[0] for _, x, _ in cs)
            hi = min(x[-1] for _, x, _ in cs)
            if hi <= lo:
                continue
            grid = np.linspace(lo, hi, n_grid)
            Y = np.stack([np.interp(grid, x, y) for _, x, y in cs])
            out[k] = (grid, (Y >= threshold).mean(axis=0), Y.shape[0])
        return out

    def auto_escape_threshold(self, which="eval", frac=0.5, arms=None,
                              window=50):
        """Half-way between the lowest starting level and the best arm's final
        seed mean: a bar that separates seeds that made it from seeds that
        stayed at the starting level (the ~50 local optimum on Swimmer, the
        collapsed seeds on Ant), and a plain 'solved' bar elsewhere."""
        starts, fins = [], []
        for k in self._arms(arms):
            agg = _aggregate(self._curves(k, which, window), band="none")
            if agg:
                starts.append(float(agg[1][0]))
                fins.append(float(np.nanmax(agg[1])))
        if not fins:
            return np.nan
        lo, hi = min(starts), max(fins)
        return float(lo + frac * (hi - lo))

    def fig_escape_rate(self, threshold=None, which="eval", arms=None,
                        window=50, figsize=(6.2, 3.6)):
        """Fraction of seeds at or above a return threshold vs env steps, per
        arm: the escape / collapse rate, i.e. how many seeds have made it at
        each point in training (the claim '4/5 seeds escape' as a curve)."""
        thr = (self.auto_escape_threshold(which, arms=arms)
               if threshold is None else float(threshold))
        curves = self.escape_curves(thr, which, arms, window)
        if not curves or not np.isfinite(thr):
            return _empty_fig("no completed runs in this family yet")
        fig, ax = plt.subplots(figsize=figsize)
        for k, (grid, frac, n) in curves.items():
            a = self.arms[k]
            ax.step(grid, frac, where="post", color=a.colour, ls=a.ls,
                    lw=2.4 if a.is_baseline else 1.9,
                    zorder=1 if a.is_baseline else 2,
                    label=f"{a.label}  ({n} seeds)")
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        self._finish(ax, ylabel=f"fraction of seeds with return $\\geq$ {_fmt_thr(thr)}",
                     title=f"{self.env} - seeds above {_fmt_thr(thr)} ({which})",
                     legend="best")
        return fig

    def table_escape(self, threshold=None, which="eval", arms=None, window=50):
        thr = (self.auto_escape_threshold(which, arms=arms)
               if threshold is None else float(threshold))
        curves = self.escape_curves(thr, which, arms, window)
        print(f"seeds at or above {_fmt_thr(thr)} ({which}) - {self.env}")
        print(f"{'arm':30s} {'at end':>8s} {'at half':>8s} {'first all':>10s}")
        print("-" * 60)
        for k, (grid, frac, n) in curves.items():
            first_all = grid[np.argmax(frac >= 1.0)] if (frac >= 1.0).any() else np.nan
            fa = f"{first_all / 1000:7.0f}k" if np.isfinite(first_all) else f"{'-':>8s}"
            print(f"{self.arms[k].plain:30s} {frac[-1] * n:4.0f}/{n:<3d} "
                  f"{frac[len(frac) // 2] * n:4.0f}/{n:<3d} {fa}")
        return thr

    def fig_final_strip(self, mode="last", arms=None, figsize=(6.6, 3.6),
                        n_boot=2000, jitter=0.08):
        """Every arm on one axis: per-seed final returns (dots), the IQM
        (interquartile mean, rliable's aggregate) with a 95% bootstrap CI,
        and the baseline IQM as a reference line."""
        keys = [k for k in self._arms(arms) if self.finals(k, mode).size]
        if not keys:
            return _empty_fig("no completed runs in this family yet")
        fig, ax = plt.subplots(figsize=figsize)
        rng = np.random.default_rng(BOOT_SEED)
        base_iqm = (iqm(self.finals(self.baseline.key, mode))
                    if self.baseline and self.baseline.key in keys else np.nan)
        if np.isfinite(base_iqm):
            ax.axhline(base_iqm, color=BASELINE_COLOUR, lw=1.1, ls=(0, (4, 2)),
                       zorder=1)
        for i, k in enumerate(keys):
            a = self.arms[k]
            v = self.finals(k, mode)
            ax.plot(i + rng.uniform(-jitter, jitter, v.size), v, "o", ms=4.2,
                    mfc="white", mec=a.colour, mew=1.2, ls="none", zorder=3)
            m = iqm(v)
            lo, hi = iqm_ci(v, n_boot=n_boot)
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]] if np.isfinite(lo) else None,
                        fmt="s" if a.is_baseline else "D", ms=7, color=a.colour,
                        mec="#111111", mew=0.6, capsize=4, elinewidth=1.4,
                        zorder=4)
        ax.set_xticks(range(len(keys)), [self.arms[k].short for k in keys],
                      rotation=25, ha="right", fontsize=8.5)
        ax.set_ylabel(f"final greedy-eval return ({mode})")
        ax.set_title(f"{self.env} - final return per arm: seeds (o), IQM with "
                     f"95% bootstrap CI; dashed = baseline IQM", fontsize=9.5)
        ax.grid(axis="x", visible=False)
        return fig

    def fig_rho_vs_gain(self, stream="grad", arms=None, tail=0.5, mode="last",
                        figsize=(6.2, 3.8), n_bins=600):
        """The hyper-parameter-selection figure: each FHR arm's tail-median
        stream ratio (x: rho_grad = lambda |grad pen| / |grad TD|, or rho_loss)
        against its final-return gain over the baseline (y, %), +- 1 s.e.m.
        Arms whose penalty barely moves the critic sit at the left; the
        over-penalised ones at the right."""
        col = {"grad": "grad_rho", "loss": "rho_loss"}[stream]
        if not self.baseline:
            return _empty_fig("rho-vs-gain needs a baseline")
        base = self.finals(self.baseline.key, mode)
        if not base.size:
            return _empty_fig("no completed baseline runs yet")
        bmean = base.mean()
        fig, ax = plt.subplots(figsize=figsize)
        ax.axhline(0.0, color=BASELINE_COLOUR, lw=1.1, ls=(0, (4, 2)), zorder=1)
        drew = False
        for k in self._arms(arms, fhr_only=True):
            a = self.arms[k]
            v = self.finals(k, mode)
            x = self.stream_medians(k, tail, n_bins).get(col, np.nan)
            if not v.size or not np.isfinite(x) or x <= 0:
                continue
            y = 100.0 * (v.mean() / bmean - 1.0)
            se = 100.0 * (v.std(ddof=1) / np.sqrt(v.size) / abs(bmean)) if v.size > 1 else 0.0
            ax.errorbar(x, y, yerr=se, fmt="D" if a.frozen else "o", ms=7,
                        color=a.colour, mec="#111111", mew=0.5, capsize=3,
                        elinewidth=1.1, zorder=3, label=a.label)
            ax.annotate(a.short, (x, y), xytext=(5, 4), textcoords="offset points",
                        fontsize=7.5, color=a.colour)
            drew = True
        if not drew:
            plt.close(fig)
            return _empty_fig(f"no arm logged {col} - launch with grad_probe_every > 0")
        ax.set_xscale("log")
        lab = ("$\\rho_{grad}=\\lambda\\,\\|\\nabla$pen$\\|/\\|\\nabla$TD$\\|$"
               if stream == "grad" else "$\\rho_{loss}=\\lambda\\cdot$penalty / TD")
        ax.set_xlabel(f"{lab}  (tail median over the last {tail:.0%} of training)")
        ax.set_ylabel("final return vs baseline (%)")
        ax.set_title(f"{self.env} - how hard the penalty pushes vs what it buys",
                     fontsize=10)
        ax.legend(loc="best", fontsize=7)
        return fig

    def theory_c(self, order):
        """The Bellman-informed coefficient vector for this family's gamma:
        (1 + 1/gamma, -1/gamma, 0, ...) - the annihilator of a
        Bellman-consistent sequence under constant reward."""
        c = np.zeros(order)
        if order >= 2:
            c[0], c[1] = 1.0 + 1.0 / self.gamma, -1.0 / self.gamma
        else:
            c[0] = 1.0 / self.gamma
        return c

    def fig_coefficient(self, j, arms=None, n_bins=600, smooth=5,
                        figsize=(6.2, 3.4)):
        """c_j over training for every arm of order >= j (learned arms move,
        frozen arms sit still), with the Bellman-theory value as a reference
        line: does the learned recurrence stay near the Bellman poles?"""
        keys = [k for k in self._arms(arms, fhr_only=True)
                if self.arms[k].order >= j]
        if not keys:
            return _empty_fig(f"no arm of order >= {j}")
        fig, ax = plt.subplots(figsize=figsize)
        for k in keys:
            cs = self.diag_curves(k, f"c_{j}", n_bins)
            if cs:
                self._plot_band(ax, k, cs, smooth=smooth, alpha_band=0.15)
        if not ax.lines:
            plt.close(fig)
            return _empty_fig(f"c_{j} not logged")
        for order in sorted({self.arms[k].order for k in keys}):
            ref = self.theory_c(order)[j - 1]
            ax.axhline(ref, color="#555555", lw=1.0, ls=(0, (2, 2)), zorder=0)
            ax.annotate(f"theory r={order}: {ref:.3f}", (1.0, ref),
                        xycoords=("axes fraction", "data"), xytext=(-4, 3),
                        textcoords="offset points", ha="right", fontsize=7,
                        color="#555555")
        self._finish(ax, ylabel=f"$c_{{{j}}}$",
                     title=f"{self.env} - recurrence coefficient $c_{{{j}}}$ vs "
                           f"the Bellman theory value", legend="best")
        return fig

    def final_coefficients(self, arm, tail=0.05, n_bins=600):
        """[(seed, c vector)] from the last `tail` fraction of the binned
        diagnostics (nan-median), for the learned or frozen global c."""
        a = self.arms[arm]
        out = []
        for s, d in self.run_dirs(arm):
            data = self.load_diag(d, n_bins)
            if data is None or "c_1" not in data:
                continue
            n = len(data["env_steps"])
            cut = max(0, n - max(1, int(n * tail)))
            c = np.array([np.nanmedian(np.asarray(data[f"c_{j}"], float)[cut:])
                          for j in range(1, a.order + 1) if f"c_{j}" in data])
            if c.size == a.order and np.isfinite(c).all():
                out.append((s, c))
        return out

    def fig_companion_roots(self, arms=None, figsize=(4.8, 5.4), tail=0.05):
        """End-of-training roots of z^r - sum_j c_j z^(r-j) per arm and seed
        in the complex plane, with the unit circle and the Bellman-theory
        roots {1, 1/gamma}: where the learned recurrence put its poles."""
        keys = self._arms(arms, fhr_only=True)
        fig, ax = plt.subplots(figsize=figsize)
        t = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(t), np.sin(t), color="#999999", lw=0.8, zorder=0)
        drew = False
        for k in keys:
            a = self.arms[k]
            for s, c in self.final_coefficients(k, tail):
                roots = np.roots(np.concatenate(([1.0], -c)))
                ax.plot(roots.real, roots.imag, "o" if not a.frozen else "D",
                        ms=5.5 if a.order <= 2 else 4.2, mfc=a.colour if not a.frozen else "white",
                        mec=a.colour, mew=1.1, alpha=0.85, ls="none",
                        label=a.label if s == self.run_dirs(k)[0][0] else None,
                        zorder=3)
                drew = True
        if not drew:
            plt.close(fig)
            return _empty_fig("no arm has logged coefficients yet")
        ax.plot([1.0, 1.0 / self.gamma], [0, 0], "x", ms=9, mew=2.0,
                color="#111111", ls="none", zorder=4,
                label=f"theory roots $\\{{1, 1/\\gamma\\}}$ = {{1, {1 / self.gamma:.3f}}}")
        ax.axhline(0, color="#BBBBBB", lw=0.6, zorder=0)
        ax.axvline(0, color="#BBBBBB", lw=0.6, zorder=0)
        ax.set_aspect("equal")
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.set_title(f"{self.env} - companion roots of the learned recurrence "
                     f"(final, per seed)", fontsize=9.5)
        # the legend goes below the axes: inside, it always lands on the roots
        fig.legend(loc="outside lower center", ncol=2, fontsize=6.5)
        return fig

    def fig_window_vs_return(self, key="pen_tail_ratio", arms=None, mode="last",
                             figsize=(6.2, 3.8), quarter=0.25):
        """Compression vs performance: per seed, the final-quarter mean of a
        penalised-window metric (x) against the final return (y); arm means
        as large markers with +- 1 s.e.m. both ways."""
        title = {k: t for k, t, _ in self.WINDOW_KEYS}.get(key, key)
        fig, ax = plt.subplots(figsize=figsize)
        drew = False
        for k in self._arms(arms):
            a = self.arms[k]
            xs, ys = [], []
            fins = dict(zip([s for s, _ in self.run_dirs(k)], self.finals(k, mode)))
            for s, m in self.window_metrics(k):
                v = np.asarray(m.get(key, []), float)
                if v.size == 0 or s not in fins:
                    continue
                cut = max(0, int(len(v) * (1 - quarter)))
                xs.append(float(np.nanmean(v[cut:])))
                ys.append(float(fins[s]))
            if not xs:
                continue
            xs, ys = np.array(xs), np.array(ys)
            ax.plot(xs, ys, "o", ms=3.6, mfc="white", mec=a.colour, mew=1.0,
                    ls="none", alpha=0.9, zorder=2)
            sx = xs.std(ddof=1) / np.sqrt(xs.size) if xs.size > 1 else 0
            sy = ys.std(ddof=1) / np.sqrt(ys.size) if ys.size > 1 else 0
            ax.errorbar(xs.mean(), ys.mean(), xerr=sx, yerr=sy,
                        fmt="s" if a.is_baseline else ("D" if a.frozen else "o"),
                        ms=8, color=a.colour, mec="#111111", mew=0.6, capsize=3,
                        elinewidth=1.1, zorder=3, label=a.label)
            drew = True
        if not drew:
            plt.close(fig)
            return _empty_fig("no instrumented runs (window_rank_every) yet")
        if key == "pen_tail_ratio":
            ax.set_xscale("log")
        ax.set_xlabel(f"{title} (final quarter of training)")
        ax.set_ylabel(f"final greedy-eval return ({mode})")
        ax.set_title(f"{self.env} - penalised-window compression vs return",
                     fontsize=10)
        ax.legend(loc="best", fontsize=7)
        return fig

    def fig_normalised_overlay(self, which="eval", arms=None, window=50,
                               smooth=1, figsize=(6.6, 4.1)):
        """fig_overlay with returns divided by the baseline's final seed-mean
        return, so families of different envs share one y-axis (1.0 = the
        baseline's final level)."""
        if not self.baseline or not self.finals(self.baseline.key).size:
            return _empty_fig("normalised curves need a completed baseline")
        scale = float(self.finals(self.baseline.key).mean())
        if not np.isfinite(scale) or scale == 0:
            return _empty_fig("baseline final return is zero - cannot normalise")
        fig, ax = plt.subplots(figsize=figsize)
        for k in self._arms(arms):
            cs = [(s, x, np.asarray(y, float) / scale)
                  for s, x, y in self._curves(k, which, window)]
            self._plot_band(ax, k, cs, smooth=smooth,
                            zorder=1 if self.arms[k].is_baseline else 2)
        if not ax.lines:
            plt.close(fig)
            return _empty_fig("no completed runs in this family yet")
        ax.axhline(1.0, color="#999999", lw=0.8, ls=(0, (2, 2)), zorder=0)
        self._finish(ax, ylabel="return / baseline final return",
                     title=f"{self.env} - all arms, baseline-normalised ({which})",
                     legend="lower right")
        return fig

    # ----------------------------------------------------------------------
    # 9 - analyses ported from the hand-written notebooks on main: window
    #     spectrum profiles, sigma_{r+1}/sigma_r, compression vs baseline,
    #     in-training rollout Hankel sweep, per-seed solve table, frozen-c
    #     drift check
    # ----------------------------------------------------------------------
    @staticmethod
    def _read_rows(path):
        with open(path) as f:
            return [{k: (float(v) if v not in ("", None) else np.nan)
                     if k not in ("matrix",) else v
                     for k, v in r.items()} for r in csv.DictReader(f)]

    def window_rows(self, arm, quarter=None):
        """Raw window_hankel.csv rows (all seeds x critics) of one arm; with
        `quarter` only the rows from the last fraction of gradient steps."""
        out = []
        for _, d in self.run_dirs(arm):
            f = d / "window_hankel.csv"
            if not f.exists():
                continue
            rows = [r for r in self._read_rows(f) if r.get("n_windows", 0) > 0]
            if not rows:
                continue
            if quarter:
                gmax = max(r["grad_step"] for r in rows)
                rows = [r for r in rows if r["grad_step"] >= (1 - quarter) * gmax]
            out.extend(rows)
        return out

    @staticmethod
    def _spectrum_profile(rows, prefix="sv_"):
        keys = sorted(k for k in rows[0] if k.startswith(prefix))
        mat = np.array([[r[k] for k in keys] for r in rows], dtype=float)
        med = np.nanmedian(mat, axis=0)
        return med[np.isfinite(med) & (med > 0)]

    def fig_window_spectrum_profile(self, arms=None, which="sv", quarter=0.25,
                                    ratio="normalised", figsize=(6.2, 3.6)):
        """Shape of the penalised-window spectrum late in training: the
        nan-median singular values over every raw probe row (all seeds x
        critics) in the final `quarter` of gradient steps, per arm, as
        sigma_i/sigma_1 (ratio="normalised") or the consecutive decay
        sigma_{i+1}/sigma_i (ratio="consecutive"), vs index. which="pen_sv"
        gives the trailing (r+1)-column penalty block instead of the full
        window."""
        prefix = "pen_sv_" if which == "pen_sv" else "sv_"
        fig, ax = plt.subplots(figsize=figsize)
        orders = set()
        for k in self._arms(arms):
            rows = self.window_rows(k, quarter)
            if not rows:
                continue
            prof = self._spectrum_profile(rows, prefix)
            if prof.size < 2:
                continue
            a = self.arms[k]
            if ratio == "consecutive":
                y = prof[1:] / prof[:-1]
                x = np.arange(1, y.size + 1)
            else:
                y = prof / prof[0]
                x = np.arange(1, y.size + 1)
            ax.semilogy(x, y, marker="o", ms=3.2, color=a.colour, ls=a.ls,
                        lw=2.3 if a.is_baseline else 1.8, label=a.label,
                        zorder=1 if a.is_baseline else 2)
            if not a.is_baseline:
                orders.add(a.order)
        if not ax.lines:
            plt.close(fig)
            return _empty_fig("no instrumented runs (window_rank_every) yet")
        for r in sorted(orders):
            ax.axvline(r, color="#888888", lw=0.7, ls=(0, (2, 2)), zorder=0)
        what = "penalty block" if which == "pen_sv" else "replay window"
        ylab = ("$\\sigma_{i+1}/\\sigma_i$" if ratio == "consecutive"
                else "$\\sigma_i/\\sigma_1$")
        ax.set(xlabel="singular-value index $i$", ylabel=ylab,
               title=f"{self.env} - {what} spectrum, final {quarter:.0%} of "
                     f"training (median over seeds and critics)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc="best", fontsize=7)
        return fig

    def window_ratio_curves(self, arm, num=-1, den=-2, prefix="pen_sv_"):
        """[(seed, env_steps, sigma_num/sigma_den)] per run from the raw
        probe rows, averaged over critics per tick. Defaults give the
        penalty block's sigma_{r+1}/sigma_r (last / second-last column)."""
        cs = []
        for s, d in self.run_dirs(arm):
            f = d / "window_hankel.csv"
            if not f.exists():
                continue
            rows = [r for r in self._read_rows(f) if r.get("n_windows", 0) > 0]
            if not rows:
                continue
            keys = sorted(k for k in rows[0] if k.startswith(prefix))
            by_tick = {}
            for r in rows:
                vals = np.array([r[k] for k in keys], float)
                vals = vals[np.isfinite(vals)]
                if vals.size < 2:
                    continue
                by_tick.setdefault(r["env_steps"], []).append(vals[num] / vals[den])
            if by_tick:
                xs = np.array(sorted(by_tick))
                ys = np.array([np.mean(by_tick[x]) for x in xs])
                cs.append((s, xs, ys))
        return cs

    def fig_window_pen_ratio(self, arms=None, smooth=3, figsize=(6.2, 3.6)):
        """The penalty block's sigma_{r+1}/sigma_r vs env steps per arm - the
        order-relative tail the older notebooks tracked (a different
        denominator from pen_tail_ratio = sigma_{r+1}/sigma_1)."""
        fig, ax = plt.subplots(figsize=figsize)
        for k in self._arms(arms):
            cs = self.window_ratio_curves(k)
            if cs:
                self._plot_band(ax, k, cs, smooth=smooth, alpha_band=0.12,
                                zorder=1 if self.arms[k].is_baseline else 2)
        if not ax.lines:
            plt.close(fig)
            return _empty_fig("no instrumented runs (window_rank_every) yet")
        ax.set_yscale("log")
        self._finish(ax, ylabel="penalty block $\\sigma_{r+1}/\\sigma_r$",
                     title=f"{self.env} - penalised windows: "
                           f"$\\sigma_{{r+1}}/\\sigma_r$ of the penalty block",
                     legend="best")
        return fig

    def fig_window_compression(self, key="pen_tail_ratio", arms=None, smooth=3,
                               figsize=(6.2, 3.6)):
        """Each FHR arm's seed-median window metric divided by the baseline's
        seed-median on a common env-step grid (identical windows): the
        compression FHR achieves relative to the control, with the rule at 1.
        The baseline is the rule, so it is not drawn."""
        if not self.baseline:
            return _empty_fig("compression needs a baseline")
        base = _aggregate(self._window_curves(self.baseline.key, key), band="iqr")
        if base is None:
            return _empty_fig("no instrumented baseline runs (window_rank_every) yet")
        bgrid, bmed = base[0], base[1]
        fig, ax = plt.subplots(figsize=figsize)
        title = {k: t for k, t, _ in self.WINDOW_KEYS}.get(key, key)
        for k in self._arms(arms, fhr_only=True):
            cs = self._window_curves(k, key)
            agg = _aggregate(cs, band="iqr")
            if agg is None:
                continue
            grid, med, lo, hi, n = agg
            ref = np.interp(grid, bgrid, bmed)
            ok = ref > 0
            a = self.arms[k]
            ax.plot(grid[ok], _smooth(med[ok] / ref[ok], smooth), color=a.colour,
                    ls=a.ls, lw=1.9, label=a.label)
            if n > 1:
                ax.fill_between(grid[ok], _smooth(lo[ok] / ref[ok], smooth),
                                _smooth(hi[ok] / ref[ok], smooth), color=a.colour,
                                alpha=0.10, lw=0, rasterized=True)
        if not ax.lines:
            plt.close(fig)
            return _empty_fig("no instrumented FHR runs yet")
        ax.axhline(1.0, color=BASELINE_COLOUR, lw=1.1, ls=(0, (4, 2)), zorder=0)
        ax.set_yscale("log")
        self._finish(ax, ylabel=f"{title}\n(arm / baseline, same windows)",
                     title=f"{self.env} - compression relative to the baseline",
                     legend="best")
        return fig

    # -- in-training rollout Hankel sweep (hankel_sweep.csv with sv columns)
    def sweep_rows(self, arm, matrix="Hankel Q"):
        out = []
        for s, d in self.run_dirs(arm):
            f = d / "hankel_sweep.csv"
            if not f.exists():
                continue
            rows = [r for r in self._read_rows(f)
                    if r.get("matrix") == matrix and np.isfinite(r.get("sv_01", np.nan))]
            if rows:
                out.append((s, rows))
        return out

    def fig_rollout_hankel_sweep(self, arms=None, ratio=(3, 2), smooth=1,
                                 figsize=(6.2, 3.6)):
        """In-training greedy-rollout Hankel(Q) sigma_a/sigma_b vs episode from
        the periodic analysis sweep (median over the rollouts of each tick),
        per arm - needs hankel_sweep.csv with sv_NN columns (families launched
        after the sweep started logging singular values)."""
        a_, b_ = ratio
        fig, ax = plt.subplots(figsize=figsize)
        for k in self._arms(arms):
            cs = []
            for s, rows in self.sweep_rows(k):
                eps = np.array([r["episode"] for r in rows])
                val = np.array([r.get(f"sv_{a_:02d}", np.nan) / r.get(f"sv_{b_:02d}", np.nan)
                                for r in rows])
                uniq = np.unique(eps)
                med = np.array([np.nanmedian(val[eps == e]) for e in uniq])
                ok = np.isfinite(med)
                if ok.sum() > 1:
                    cs.append((s, uniq[ok], med[ok]))
            if cs:
                self._plot_band(ax, k, cs, smooth=smooth, alpha_band=0.12,
                                zorder=1 if self.arms[k].is_baseline else 2)
        if not ax.lines:
            plt.close(fig)
            return _empty_fig("hankel_sweep.csv carries no singular values for "
                              "this family (logged for families launched after "
                              "the sweep started recording sv_NN)")
        ax.axhline(1.0, color="#999999", lw=0.8, ls=(0, (2, 2)), zorder=0)
        ax.set_yscale("log")
        ax.set_xlabel("training episode")
        ax.set_ylabel(f"rollout Hankel(Q) $\\sigma_{a_}/\\sigma_{b_}$")
        ax.set_title(f"{self.env} - in-training rollout Hankel spectrum ratio")
        ax.legend(loc="best", fontsize=7)
        return fig

    def fig_rollout_spectrum_late(self, arms=None, last_eps=200, n_show=12,
                                  figsize=(6.2, 3.6)):
        """Late-training greedy-rollout Hankel(Q) spectrum sigma_i/sigma_1
        (median over every sweep row of the last `last_eps` episodes, all
        seeds and rollouts), per arm."""
        fig, ax = plt.subplots(figsize=figsize)
        for k in self._arms(arms):
            prof_rows = []
            for s, rows in self.sweep_rows(k):
                eps = np.array([r["episode"] for r in rows])
                emax = eps.max()
                prof_rows += [r for r, e in zip(rows, eps) if e >= emax - last_eps]
            if not prof_rows:
                continue
            keys = sorted(k_ for k_ in prof_rows[0] if k_.startswith("sv_"))[:n_show]
            svs = np.array([[r.get(k_, np.nan) for k_ in keys] for r in prof_rows], float)
            prof = np.nanmedian(svs, axis=0)
            prof = prof[np.isfinite(prof) & (prof > 0)]
            if prof.size < 2:
                continue
            a = self.arms[k]
            ax.semilogy(np.arange(1, prof.size + 1), prof / prof[0], "o-", ms=3.2,
                        color=a.colour, ls=a.ls, lw=2.3 if a.is_baseline else 1.8,
                        label=a.label)
        if not ax.lines:
            plt.close(fig)
            return _empty_fig("hankel_sweep.csv carries no singular values for this family")
        ax.set(xlabel="singular-value index $i$", ylabel="$\\sigma_i/\\sigma_1$",
               title=f"{self.env} - late-training rollout Hankel(Q) spectrum "
                     f"(last {last_eps} episodes)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc="best", fontsize=7)
        return fig

    # -- per-seed solve table + frozen-c drift check
    def table_per_seed(self, threshold=None, arms=None, mode="last"):
        """One row per seed: first env step at which the greedy-eval return
        reached the solved level (training.solved_reward, else the escape
        threshold), the final and the best eval return."""
        thr = (self.solved if self.solved is not None else
               (self.auto_escape_threshold("eval") if threshold is None else threshold))
        print(f"per-seed greedy-eval summary - {self.env} (bar {_fmt_thr(thr)})")
        print(f"{'arm':30s} {'seed':>5s} {'solve@step':>11s} {'final':>9s} {'best':>9s}")
        print("-" * 70)
        out = []
        for k in self._arms(arms):
            for s, x, y in self.eval_curves(k):
                hit = np.flatnonzero(np.asarray(y) >= thr)
                step = float(x[hit[0]]) if hit.size else np.nan
                fin, best = float(y[-1]), float(np.nanmax(y))
                out.append((k, s, step, fin, best))
                st = f"{step / 1000:9.0f}k" if np.isfinite(step) else f"{'never':>10s}"
                print(f"{self.arms[k].plain:30s} {s:>5s} {st:>11s} {fin:9.1f} {best:9.1f}")
        return out

    def table_coefficient_drift(self, arms=None, n_bins=600, tol=1e-4):
        """Frozen-c control check: the largest |c_j - theory_j| any seed of a
        frozen arm ever logged (must stay ~0 - the constant never leaves its
        theory init); learned arms are listed for contrast."""
        print(f"{'arm':30s} {'seed':>5s} {'max |c - theory|':>17s}")
        print("-" * 58)
        out = {}
        for k in self._arms(arms, fhr_only=True):
            a = self.arms[k]
            theory = self.theory_c(a.order)
            if a.c_init is not None and len(a.c_init) == a.order:
                theory = np.asarray(a.c_init, float)
            for s, d in self.run_dirs(k):
                data = self.load_diag(d, n_bins)
                if data is None or "c_1" not in data:
                    continue
                c = np.array([np.asarray(data[f"c_{j}"], float)
                              for j in range(1, a.order + 1) if f"c_{j}" in data])
                if c.shape[0] != a.order:
                    continue
                drift = float(np.nanmax(np.abs(c - theory[:, None])))
                out[(k, s)] = drift
                flag = ("OK" if drift < tol else "MOVED") if a.frozen else "(learned)"
                print(f"{a.plain:30s} {s:>5s} {drift:17.2e}  {flag}")
        return out

    # ----------------------------------------------------------------------
    # 10 - paper figures: stream summary dot plot, paired speed-up, IQM
    #      learning curves, seed-banded spectra
    # ----------------------------------------------------------------------
    def stream_medians_by_seed(self, arm, tail=0.5, n_bins=600):
        vals = {c: [] for c, _ in self.STREAM_COLS}
        for _, d in self.run_dirs(arm):
            data = self.load_diag(d, n_bins)
            if data is None:
                continue
            cut = int(len(data["env_steps"]) * (1 - tail))
            for c, _ in self.STREAM_COLS:
                if c in data:
                    v = np.asarray(data[c], float)[cut:]
                    if np.isfinite(v).any():
                        vals[c].append(float(np.nanmedian(v)))
        return {c: np.array(v) for c, v in vals.items()}

    def fig_stream_summary(self, tail=0.5, arms=None, targets=(0.1, 0.5, 1.0),
                           n_bins=600, figsize=(7.4, 3.4)):
        """The seven stream time series as one readable summary: per arm the
        tail medians of rho_grad, rho_loss (log x) and cos(grad TD, grad pen)
        as seed dots + median diamond, arms as rows; the baseline row shows
        its UNWEIGHTED ratios (its penalty never enters the loss). The target
        rho interval is shaded."""
        keys = self._arms(arms)
        rows = [(k, self.stream_medians_by_seed(k, tail, n_bins)) for k in keys]
        rows = [(k, m) for k, m in rows if any(v.size for v in m.values())]
        if not rows:
            return _empty_fig("no stream ratios logged (grad_probe_every)")
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True,
                                 gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
        panels = (("grad_rho", "grad_ratio", "$\\rho_{grad}$"),
                  ("rho_loss", "loss_ratio", "$\\rho_{loss}$"),
                  ("grad_cos", "grad_cos", "cos$(\\nabla$TD$,\\nabla$pen$)$"))
        ypos = np.arange(len(rows))[::-1]
        for ax, (col, base_col, lab) in zip(axes, panels):
            for y, (k, m) in zip(ypos, rows):
                a = self.arms[k]
                v = m.get(base_col if a.is_baseline else col, np.array([]))
                v = v[np.isfinite(v)]
                if col != "grad_cos":
                    v = v[v > 0]
                if not v.size:
                    continue
                colour = "#777777" if a.is_baseline else a.colour
                ax.plot(v, np.full(v.size, y), "o", ms=3.6, mfc="white",
                        mec=colour, mew=1.0, ls="none", zorder=2)
                ax.plot(np.median(v), y, "D", ms=6.5, color=colour,
                        mec="#111111", mew=0.5, zorder=3)
            if col != "grad_cos":
                ax.set_xscale("log")
                if targets:
                    ax.axvspan(min(targets), max(targets), color="#DDDDDD",
                               alpha=0.5, zorder=0)
            else:
                ax.axvline(0, color="#777777", lw=0.8, ls=(0, (2, 2)), zorder=0)
            ax.set_xlabel(lab)
            ax.grid(axis="y", visible=False)
        axes[0].set_yticks(ypos, [self.arms[k].short for k, _ in rows], fontsize=8)
        fig.suptitle(f"{self.env} - stream ratios, tail medians over the last "
                     f"{tail:.0%} (baseline: unweighted); shaded = target range",
                     fontsize=9.5)
        return fig

    def fig_speedup_paired(self, thresholds=None, arms=None, window=50,
                           n_boot=2000, ci=0.95, stream="train",
                           figsize=(6.6, 3.9)):
        """Sample-efficiency speed-up per threshold as a ratio of IQM steps
        (baseline / arm) on a log2 axis, with a seed-PAIRED bootstrap CI
        (seed ids are shared across arms) and explicit censoring: a hollow
        marker with 'k/n' where only k seeds of the arm reached the bar."""
        thresholds = thresholds if thresholds is not None else self.auto_thresholds(window, stream=stream)
        if not self.baseline or not thresholds:
            return _empty_fig("paired speed-up needs a baseline and a threshold ladder")
        keys = self._arms(arms)
        if self.baseline.key not in keys:
            keys = [self.baseline.key] + keys
        xs = self.steps_to_thresholds(thresholds, window, keys, stream)
        bkey = self.baseline.key
        rng = np.random.default_rng(BOOT_SEED)
        fig, ax = plt.subplots(figsize=figsize)
        pos = np.arange(len(thresholds), dtype=float)
        fhr = [k for k in keys if k != bkey]
        for i, k in enumerate(fhr):
            a = self.arms[k]
            ys, los, his, cens = [], [], [], []
            for t in thresholds:
                b, v = xs[bkey][t], xs[k][t]
                n = min(b.size, v.size)
                if n == 0:
                    ys.append(np.nan); los.append(np.nan); his.append(np.nan); cens.append("")
                    continue
                b, v = b[:n], v[:n]
                okb, okv = np.isfinite(b), np.isfinite(v)
                if not okb.all() or not okv.any():
                    ys.append(np.nan); los.append(np.nan); his.append(np.nan)
                    cens.append(f"{int(okv.sum())}/{n}")
                    continue
                ratio = iqm(b[okv]) / iqm(v[okv])
                boots = []
                ids_all = np.flatnonzero(okv)
                for _ in range(n_boot):
                    ids = rng.choice(ids_all, ids_all.size, replace=True)
                    boots.append(iqm(b[ids]) / iqm(v[ids]))
                alpha = (1 - ci) / 2
                ys.append(ratio)
                los.append(ratio - np.quantile(boots, alpha))
                his.append(np.quantile(boots, 1 - alpha) - ratio)
                cens.append("" if okv.all() else f"{int(okv.sum())}/{n}")
            off = (i - (len(fhr) - 1) / 2) * 0.12
            ys, los, his = np.array(ys), np.array(los), np.array(his)
            ax.errorbar(pos + off, ys, yerr=[np.nan_to_num(los), np.nan_to_num(his)],
                        fmt="o" if not a.frozen else "D", ms=5.5, color=a.colour,
                        mec="#111111", mew=0.5, capsize=2.5, elinewidth=1.0,
                        ls=a.ls, lw=1.4, label=a.label, zorder=3)
            for x_, c_ in zip(pos + off, cens):
                if c_:
                    ax.annotate(c_, (x_, 1.0), xytext=(0, 4), textcoords="offset points",
                                ha="center", fontsize=6.5, color=a.colour)
        ax.axhline(1.0, color=BASELINE_COLOUR, lw=1.2, zorder=2)
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}x"))
        ax.set_xticks(pos, [_fmt_thr(t) for t in thresholds])
        ax.set_xlabel(f"{'training' if stream == 'train' else 'greedy-eval'}-return threshold")
        ax.set_ylabel("speed-up vs baseline (IQM steps ratio)")
        ax.set_title(f"{self.env} - paired-bootstrap speed-up ({ci:.0%} CI); "
                     f"'k/n' = seeds that reached the bar", fontsize=9.5)
        ax.legend(loc="best", fontsize=7, ncol=2)
        return fig

    def fig_learning_curves_paper(self, which="eval", arms=None, window=50,
                                  smooth=1, ax=None, figsize=(5.5, 2.4),
                                  letter=None):
        """The headline learning-curve panel: IQM with a 95% bootstrap CI band,
        baseline black, selected arms in their colours, short labels, no
        title (a panel letter + env name inside the axes instead). Draws into
        `ax` when given so several envs can share one row."""
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize)
        for k in self._arms(arms):
            a = self.arms[k]
            self._plot_band(ax, k, self._curves(k, which, window), band="iqm",
                            smooth=smooth, alpha_band=0.12, label=a.paper_label,
                            lw=1.8 if a.is_baseline else 1.4,
                            zorder=1 if a.is_baseline else 2)
        if not ax.lines:
            plt.close(fig)
            return _empty_fig("no completed runs in this family yet")
        ax.xaxis.set_major_formatter(_steps_formatter())
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel("environment steps")
        ax.set_ylabel("eval return" if which == "eval" else "training return")
        tag = f"{letter}  " if letter else ""
        ax.text(0.02, 0.96, f"{tag}{self.env.split('-')[0]}", transform=ax.transAxes,
                ha="left", va="top", fontsize=8, fontweight="bold")
        ax.legend(loc="lower right", fontsize=6.5, handlelength=1.5)
        return fig

    def fig_spectra_paper(self, runner, arms=None, n_show=12, floor=1e-3,
                          n_rollouts=3, base_seed=52, figsize=(7.0, 3.0)):
        """Low-rank structure on both objects the paper talks about: (a) the
        converged policy's rollout Hankel(Q) spectrum sigma_i/sigma_1 as the
        median over ALL seeds with a min-max band, (b) the final penalised
        replay-window spectrum (median over seeds and critics in the final
        quarter). Vertical rules at the penalty orders present."""
        keys = self._arms(arms)
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        orders = {self.arms[k].order for k in keys if not self.arms[k].is_baseline}
        drew = False
        for k in keys:
            a = self.arms[k]
            specs = []
            for si in range(len(self.run_dirs(k))):
                sp = self._rollout_spectra(runner, n_rollouts, base_seed, si, [k])
                if sp:
                    specs.append(sp[0][1][:n_show])
            if specs:
                m = min(len(x) for x in specs)
                S = np.stack([x[:m] for x in specs])
                x = np.arange(1, m + 1)
                axes[0].semilogy(x, np.maximum(np.median(S, 0), floor), color=a.colour,
                                 ls=a.ls, lw=1.8 if a.is_baseline else 1.4,
                                 label=a.paper_label)
                if S.shape[0] > 1:
                    axes[0].fill_between(x, np.maximum(S.min(0), floor),
                                         np.maximum(S.max(0), floor), color=a.colour,
                                         alpha=0.10, lw=0, rasterized=True)
                drew = True
            rows = self.window_rows(k, 0.25)
            if rows:
                prof = self._spectrum_profile(rows, "sv_")[:n_show]
                if prof.size > 1:
                    axes[1].semilogy(np.arange(1, prof.size + 1),
                                     np.maximum(prof / prof[0], floor), color=a.colour,
                                     ls=a.ls, lw=1.8 if a.is_baseline else 1.4,
                                     label=a.paper_label)
                    drew = True
        if not drew:
            plt.close(fig)
            return _empty_fig("no completed runs in this family yet")
        for ax, ttl in zip(axes, ("rollout Hankel$(Q(s_t,\\pi(s_t)))$, final policy",
                                  "penalised replay windows, final quarter")):
            for r in sorted(orders):
                ax.axvline(r, color="#888888", lw=0.7, ls=(0, (2, 2)), zorder=0)
            ax.set_xlim(0.5, n_show + 0.5)
            ax.set_ylim(bottom=floor)
            ax.set_xlabel("singular-value index $i$")
            ax.set_title(ttl, fontsize=8.5)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0].set_ylabel("$\\sigma_i/\\sigma_1$")
        axes[0].legend(loc="lower left", fontsize=6.5)
        return fig


# --------------------------------------------------------------------------
# cross-environment summary
# --------------------------------------------------------------------------
def cross_env_final(families, arms=None, mode="last", figsize=(7.2, 3.8),
                    n_boot=2000):
    """One figure over several families (envs): each arm's final return
    relative to that env's baseline (IQM ratio, 95% bootstrap CI of the arm's
    IQM scaled by the baseline IQM), grouped by env. Arms are matched across
    families by key (exp1, exp2, ...); labels come from the first family."""
    fams = list(families)
    if not fams:
        return _empty_fig("no families")
    keys = [k for k in fams[0]._arms(arms, fhr_only=True)]
    fig, ax = plt.subplots(figsize=figsize)
    w = 0.8 / max(1, len(keys))
    for j, k in enumerate(keys):
        a = fams[0].arms[k]
        xs, ys, los, his = [], [], [], []
        for i, F in enumerate(fams):
            if k not in F.arms or not F.baseline:
                continue
            v, b = F.finals(k, mode), F.finals(F.baseline.key, mode)
            if not v.size or not b.size:
                continue
            bi = iqm(b)
            if not np.isfinite(bi) or bi == 0:
                continue
            m = iqm(v) / bi
            lo, hi = iqm_ci(v, n_boot=n_boot)
            xs.append(i + (j - (len(keys) - 1) / 2) * w)
            ys.append(m)
            los.append(m - lo / bi)
            his.append(hi / bi - m)
        if xs:
            ax.bar(xs, np.array(ys) - 1.0, bottom=1.0, width=w * 0.92,
                   color=a.colour, alpha=0.9, hatch="//" if a.frozen else None,
                   edgecolor="white", linewidth=0.4, label=a.label)
            ax.errorbar(xs, ys, yerr=[los, his], fmt="none", ecolor="#111111",
                        elinewidth=0.9, capsize=2.5, zorder=4)
    ax.axhline(1.0, color=BASELINE_COLOUR, lw=1.2, zorder=3)
    ax.set_xticks(range(len(fams)), [F.env.split("-")[0] for F in fams])
    ax.set_ylabel("final return / baseline (IQM ratio)")
    ax.set_title("FHR arms across environments (bars: IQM ratio; whiskers: "
                 "95% bootstrap CI)", fontsize=10)
    ax.legend(loc="best", ncol=2, fontsize=7)
    return fig
