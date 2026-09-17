"""Export the Atari-100k 5-seed results as pgfplots data files + LaTeX figures.

Mirrors exp_atari100k_effrainbow_5seed.ipynb exactly: same manifests, the notebook's
own SEED0_FALLBACK table (read from the .ipynb), the same seeded stratified bootstrap
(numpy default_rng(0), 2000 replicates, cell order preserved) and the same rolling-20
learning curves on a 200-point env-step grid.  Nothing here recomputes a published
number: the published means/medians are the rows the notebook's PUBLISHED cell
reproduces from the papers' per-game tables.

    python experiments/src/export_atari100k_pgfplots.py              # -> docs/figures/atari100k/
    python experiments/src/export_atari100k_pgfplots.py --out DIR

Writes (all paths relative to --out):
    data/model_free_bars.dat        FHR vs published model-free methods (mean, median, CI)
    data/delta_hns.dat              per-game dHNS with the 95 % seed-bootstrap CI
    data/curves/<Game>_<arm>.dat    learning curves: step, mean, std, sem, n
    atari100k_pgfplots_preamble.tex packages, colours and the shared plot style
    fig_*_body.tex                  the tikzpictures -- REGENERATED on every run
    fig_*.tex                       figure wrappers (caption + label + \\input of the body),
                                    written ONCE and never overwritten: edit captions there
                                    (--rewrite-wrappers re-creates them)
    standalone_test.tex             compiles the three figures on their own
"""
import argparse
import ast
import json
import pathlib
import re
import sys
import warnings

import numpy as np
import yaml

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")

ROOT = pathlib.Path(__file__).resolve().parents[2]
ATARI = ROOT / "experiments" / "atari"
NOTEBOOK = ATARI / "exp_atari100k_effrainbow_5seed.ipynb"
sys.path.insert(0, str(ROOT / "src"))
from analysis.atari100k import (ATARI100K_ENV_STEPS, BENCHMARK_GAMES,  # noqa: E402
                                REFERENCE, game_key, hns)

REF_CONFIG = "config_effrainbow_100k_ref.yaml"
REF_MANIFEST = "cached/effrainbow100kref_runs_manifest.json"
SEED0_MANIFEST = "cached/effrainbow100k_runs_manifest.json"
ARMS = ["baseline", "exp3"]
N_SEEDS_FULL = 5
SHOWCASE_DEFAULT = ["Pong", "BankHeist", "Freeway", "UpNDown"]   # user's pick (2026-09-16)
N_BOOT = 2000
ROLL_W, GRID_N = 20, 200

# Published model-free rows exactly as the notebook's PUBLISHED cell prints them
# (mean/median HNS recomputed from each paper's per-game table and asserted against
# the paper's own aggregate row).  Parameter counts: notebook PARAMS cell.
PUBLISHED_MODEL_FREE = [
    # name,        params, mean,  median, source
    ("OTRainbow",  "2.2M", 0.264, 0.204, "EfficientZero Table 1"),
    ("DER",        "1.1M", 0.350, 0.189, "BBF Table A.1"),
    ("DrQ",        "3.3M", 0.357, 0.268, "EfficientZero Table 1"),
    ("CURL",       "1.5M", 0.381, 0.175, "EfficientZero Table 1"),
    ("DrQ(eps)",   "3.3M", 0.465, 0.312, "BBF Table A.1"),
    ("SPR",        "4M",   0.703, 0.415, "EfficientZero Table 1"),
    ("SR-SPR",     "4M",   1.271, 0.684, "BBF Table A.1"),
    ("BBF",        "40M*", 2.247, 0.917, "BBF Table A.1"),
]
OUR_PARAMS = "3.5M"
TEX_NAME = {"DrQ(eps)": r"DrQ($\epsilon$)"}          # LaTeX spelling of a method name
OUR_TEX = {"baseline": r"Baseline\\{\tiny EffRainbow}",
           "exp3": r"+\,FHR\\{\tiny (ours)}"}
LEGEND = {"baseline": "EfficientRainbow (baseline)",
          "exp3": r"EfficientRainbow + FHR ($\lambda=2$, $r=8$)"}


# ----------------------------------------------------------------------------- data
def seed0_fallback():
    """The SEED0_FALLBACK dict literal of the notebook (single source of truth)."""
    for c in json.load(open(NOTEBOOK))["cells"]:
        m = re.search(r"SEED0_FALLBACK = (\{.*?\n\})", "".join(c["source"]), re.S)
        if m:
            return ast.literal_eval(m.group(1))
    raise RuntimeError(f"SEED0_FALLBACK not found in {NOTEBOOK}")


def eval_mean(run_dir):
    p = run_dir / "eval_summary.json"
    return json.load(open(p))["mean"] if p.exists() else None


def load_scores():
    """SCORES[game][arm] = {seed: final eval mean}, exactly as notebook cell 2."""
    fallback = seed0_fallback()

    def seed0(gd, arm, key):
        p = gd / SEED0_MANIFEST
        if p.exists():
            vals = [eval_mean(gd / rel) for rel in json.load(open(p))["runs"].get(arm, {}).values()]
            vals = [v for v in vals if v is not None]
            if vals:
                return float(np.mean(vals))
        return fallback[key][ARMS.index(arm)]

    scores, overrides, run_dirs = {}, {}, {}
    for gd in sorted(ATARI.glob("dqn_*")):
        if not (gd / REF_CONFIG).exists():
            continue
        key = game_key(yaml.safe_load(open(gd / REF_CONFIG))["environment"]["name"])
        ref = (json.load(open(gd / REF_MANIFEST)) if (gd / REF_MANIFEST).exists()
               else {"runs": {}, "overrides": {}})
        if "exp3" in ref.get("overrides", {}):
            overrides[key] = ref["overrides"]["exp3"]
        scores[key], run_dirs[key] = {}, {}
        for arm in ARMS:
            d = {"0": seed0(gd, arm, key)}
            run_dirs[key][arm] = []
            for seed, rel in sorted(ref["runs"].get(arm, {}).items()):
                v = eval_mean(gd / rel)
                if v is not None:
                    d[seed] = v
                    run_dirs[key][arm].append(("ref", seed, gd / rel))
            scores[key][arm] = d
        p = gd / SEED0_MANIFEST                       # seed-0 curves, when on this machine
        if p.exists():
            for arm in ARMS:
                for seed, rel in sorted(json.load(open(p))["runs"].get(arm, {}).items()):
                    run_dirs[key][arm].insert(0, ("seed0", seed, gd / rel))
    ov = next(iter(overrides.values()))
    assert all(o == ov for o in overrides.values()), overrides
    return scores, run_dirs, ov


# ------------------------------------------------------------------------ statistics
def iqm(values, weights):
    v, w = np.asarray(values, float), np.asarray(weights, float)
    o = np.argsort(v); v, w = v[o], w[o]
    cw = np.cumsum(w); lo, hi = 0.25 * cw[-1], 0.75 * cw[-1]
    inside = np.clip(np.minimum(cw, hi) - np.maximum(cw - w, lo), 0, None)
    return float((v * inside).sum() / inside.sum())


STATS = {
    "mean": lambda s: float(np.mean([v.mean() for v in s])),
    "median": lambda s: float(np.median([v.mean() for v in s])),
    "IQM": lambda s: iqm(np.concatenate(s),
                         np.concatenate([np.full(len(v), 1.0 / len(v)) for v in s])),
}


def analyse(scores):
    games = list(scores)
    agg = [k for k in games if k in BENCHMARK_GAMES]
    hns_runs = {(k, a): np.array([hns(v, k) for v in scores[k][a].values()])
                for k in games for a in ARMS}
    n_seeds = {(k, a): len(scores[k][a]) for k in games for a in ARMS}
    short = [k for k in games if min(n_seeds[k, a] for a in ARMS) < N_SEEDS_FULL]

    rng = np.random.default_rng(0)                    # notebook cell 10, same draw order

    def bootstrap(arm):
        per_game = [hns_runs[k, arm] for k in agg]
        point = {s: f(per_game) for s, f in STATS.items()}
        reps = {s: [] for s in STATS}
        for _ in range(N_BOOT):
            sample = [rng.choice(v, size=len(v), replace=True) for v in per_game]
            for s, f in STATS.items():
                reps[s].append(f(sample))
        ci = {s: tuple(np.percentile(reps[s], [2.5, 97.5])) for s in STATS}
        return point, ci, reps

    ours = {a: bootstrap(a) for a in ARMS}
    diff = {s: (ours["exp3"][0][s] - ours["baseline"][0][s],
                tuple(np.percentile(np.array(ours["exp3"][2][s]) - np.array(ours["baseline"][2][s]),
                                    [2.5, 97.5])))
            for s in STATS}

    # notebook cell 4: per-game deltas on the mean score
    dhns, wins, losses = {}, [], []
    for k in agg:
        b = np.array(list(scores[k]["baseline"].values()))
        f = np.array(list(scores[k]["exp3"].values()))
        dhns[k] = hns(f.mean(), k) - hns(b.mean(), k)
        (wins if f.mean() > b.mean() else losses if f.mean() < b.mean() else []).append(k)

    # notebook cell 16: per-game seed bootstrap of the difference (continues the RNG)
    order = sorted(agg, key=lambda k: dhns[k])
    ci = {}
    for k in order:
        f, b = hns_runs[k, "exp3"], hns_runs[k, "baseline"]
        reps = [rng.choice(f, len(f)).mean() - rng.choice(b, len(b)).mean() for _ in range(N_BOOT)]
        ci[k] = tuple(np.percentile(reps, [2.5, 97.5]))
    return dict(games=games, agg=agg, n_seeds=n_seeds, short=short, ours=ours, diff=diff,
                dhns=dhns, order=order, ci=ci, wins=wins, losses=losses)


# ---------------------------------------------------------------------------- curves
def load_curve(run_dir):
    p = run_dir / "rewards.csv"
    if not p.exists():
        return None
    t = np.loadtxt(p, delimiter=",", skiprows=1, ndmin=2)
    return (t[:, 1], np.cumsum(t[:, 2])) if t.shape[1] > 2 else None


def curve_on_grid(curves, w=ROLL_W, n=GRID_N):
    grid = np.linspace(0, ATARI100K_ENV_STEPS, n)
    stack = np.full((len(curves), n), np.nan)
    for i, (rewards, steps) in enumerate(curves):
        if len(rewards) < w:
            continue
        y = np.convolve(rewards, np.ones(w) / w, mode="valid")
        xs = steps[len(steps) - len(y):]
        inside = (grid >= xs[0]) & (grid <= xs[-1])
        stack[i, inside] = np.interp(grid[inside], xs, y)
    return grid, stack


# ----------------------------------------------------------------------------- LaTeX
def tex_escape(s):
    return s.replace("*", r"$^{*}$").replace("_", r"\_")


BODY_HEADER = ("% Generated by experiments/src/export_atari100k_pgfplots.py -- do not edit by hand.\n"
               "% Needs atari100k_pgfplots_preamble.tex in the preamble.\n")
WRAPPER_HEADER = ("% Written ONCE by experiments/src/export_atari100k_pgfplots.py and never overwritten:\n"
                  "% edit caption / label / placement freely (--rewrite-wrappers re-creates it).\n"
                  "% The picture itself is \\input from the regenerated *_body.tex file.\n")


def write_wrapper(out, name, figures, force):
    """fig_<name>.tex = figure environment(s) around \\input{\\figtex/<body>}; written once."""
    path = out / f"fig_{name}.tex"
    if path.exists() and not force:
        return False
    blocks = [rf"""\begin{{figure}}[{placement}]
  \centering
  \input{{\figtex/{body}}}
  \caption{{{caption}}}
  \label{{{label}}}
\end{{figure}}
""" for body, caption, label, placement in figures]
    path.write_text(WRAPPER_HEADER + "\n".join(blocks))
    return True


def write_bars(out, A, force):
    ours = A["ours"]
    rows = sorted(PUBLISHED_MODEL_FREE, key=lambda r: r[2])     # ascending mean, ours last
    lines = ["idx mean median meanlo meanhi meanlab medlab"]
    labels = []
    for i, (name, params, mean, med, _) in enumerate(rows):
        lines.append(f"{i} {mean:.3f} {med:.3f} nan nan {mean:.3f} {med:.3f}")
        labels.append(rf"{TEX_NAME.get(name, name)}\\{{\tiny$\approx${tex_escape(params)}}}")
    n_pub = len(rows)
    for j, a in enumerate(ARMS):
        i = n_pub + j
        pt, ci, _ = ours[a]
        lines.append(f"{i} {pt['mean']:.3f} {pt['median']:.3f} {ci['mean'][0]:.3f} {ci['mean'][1]:.3f} "
                     f"{pt['mean']:.3f} {pt['median']:.3f}")
        labels.append(rf"{OUR_TEX[a]}\\{{\tiny$\approx${OUR_PARAMS}}}")
    (out / "data" / "model_free_bars.dat").write_text("\n".join(lines) + "\n")

    n = n_pub + len(ARMS)
    lift = 100 * A["diff"]["mean"][0] / ours["baseline"][0]["mean"]
    ymax = max(max(r[2] for r in rows), ours["exp3"][0]["mean"]) * 1.24
    tex = r"""% FHR vs published model-free Atari-100k methods (grouped bars, landscape).
% The 95 % bootstrap CI of our mean bars is kept in data/model_free_bars.dat (meanlo/meanhi)
% but not drawn.  Numbers above each group: mean (black) over median (grey).
  \begin{tikzpicture}
    \begin{axis}[
      atari style,
      width=\textwidth, height=5.2cm,
      xmin=-0.6, xmax=@XMAX@, ymin=0, ymax=@YMAX@,
      xtick={@XTICKS@},
      xticklabels={@XLABELS@},
      xticklabel style={align=center, font=\scriptsize},
      ylabel={Human Normalised Score (HNS)},
      ymajorgrids, xmajorgrids=false,
      bar width=7pt, area legend,
      legend style={at={(0.03,0.97)}, anchor=north west, legend columns=1},
      legend cell align=left,
      axis on top,
    ]
      % grey band behind our two groups
      \fill[gray!14] (axis cs:@BANDLO@,0) rectangle (axis cs:@XMAX@,@YMAX@);
      % published methods
      \addplot[ybar, bar shift=-4pt, fill=gray!65, draw=none]
        table[x=idx, y=mean, restrict expr to domain={\thisrow{idx}}{0:@NPUBM1@}] {\figdata/model_free_bars.dat};
      \addlegendentry{mean HNS (upper number)}
      \addplot[ybar, bar shift=4pt, fill=gray!30, draw=none]
        table[x=idx, y=median, restrict expr to domain={\thisrow{idx}}{0:@NPUBM1@}] {\figdata/model_free_bars.dat};
      \addlegendentry{median HNS (lower number)}
      % ours: baseline (group @IBASE@) and FHR (group @IFHR@)
      \addplot[ybar, bar shift=-4pt, fill=basecolor, draw=none, forget plot]
        table[x=idx, y=mean, restrict expr to domain={\thisrow{idx}}{@IBASE@:@IBASE@}] {\figdata/model_free_bars.dat};
      \addplot[ybar, bar shift=4pt, fill=basecolor!45, draw=none, forget plot]
        table[x=idx, y=median, restrict expr to domain={\thisrow{idx}}{@IBASE@:@IBASE@}] {\figdata/model_free_bars.dat};
      \addplot[ybar, bar shift=-4pt, fill=fhrcolor, draw=none, forget plot]
        table[x=idx, y=mean, restrict expr to domain={\thisrow{idx}}{@IFHR@:@IFHR@}] {\figdata/model_free_bars.dat};
      \addplot[ybar, bar shift=4pt, fill=fhrcolor!45, draw=none, forget plot]
        table[x=idx, y=median, restrict expr to domain={\thisrow{idx}}{@IFHR@:@IFHR@}] {\figdata/model_free_bars.dat};
      % value labels, horizontal, stacked above each group: mean (black) over median (grey)
      \addplot[only marks, mark=none, forget plot, nodes near coords, point meta=explicit symbolic,
               every node near coord/.style={anchor=south, yshift=10pt, font=\footnotesize, inner sep=1pt}]
        table[x=idx, y=mean, meta=meanlab] {\figdata/model_free_bars.dat};
      \addplot[only marks, mark=none, forget plot, nodes near coords, point meta=explicit symbolic,
               every node near coord/.style={anchor=south, yshift=1pt, font=\footnotesize, inner sep=1pt, text=gray!70!black}]
        table[x=idx, y=mean, meta=medlab] {\figdata/model_free_bars.dat};
      % reference lines: human level and the FHR mean
      \draw[dashed, gray!80, line width=0.7pt] (axis cs:-0.6,1) -- (axis cs:@XMAX@,1)
        node[pos=0, above right, font=\scriptsize, text=gray, inner sep=1.5pt] {human};
      \draw[dotted, fhrcolor, line width=1pt] (axis cs:-0.6,@FHRMEAN@) -- (axis cs:@XMAX@,@FHRMEAN@);
      \node[draw=fhrcolor, rounded corners=2pt, fill=white, text=fhrcolor, font=\small\bfseries,
            align=center, anchor=north west, inner sep=4pt]
        at (rel axis cs:0.36,0.96) {FHR: @LIFT@\,\% mean HNS\\over the baseline};
    \end{axis}
  \end{tikzpicture}
"""
    tex = (tex.replace("@XMAX@", f"{n - 0.4:.1f}").replace("@YMAX@", f"{ymax:.2f}")
              .replace("@XTICKS@", ",".join(str(i) for i in range(n)))
              .replace("@XLABELS@", ",".join("{" + l + "}" for l in labels))
              .replace("@BANDLO@", f"{n_pub - 0.5:.1f}").replace("@NPUBM1@", str(n_pub - 1))
              .replace("@IBASE@", str(n_pub)).replace("@IFHR@", str(n_pub + 1))
              .replace("@FHRMEAN@", f"{ours['exp3'][0]['mean']:.3f}")
              .replace("@LIFT@", f"{lift:+.0f}"))
    (out / "fig_model_free_bars_body.tex").write_text(BODY_HEADER + tex)
    caption = (rf"Atari-100k, 26 games, 100k environment steps: mean and median human-normalised "
               rf"score of published model-free agents (grey; the counts under the names are the "
               rf"trainable parameters of each paper's network) and of our EfficientRainbow baseline "
               rf"with and without the FHR loss ({sum(A['n_seeds'].values())} runs pooled, seeds 0--4). "
               rf"Published rows are quoted from one source each (EfficientZero Table~1; BBF Table~A.1).")
    write_wrapper(out, "model_free_bars",
                  [("fig_model_free_bars_body", caption, "fig:atari100k-model-free", "t")], force)


def write_delta(out, A, force):
    order, dhns, ci, short = A["order"], A["dhns"], A["ci"], A["short"]
    lines = ["idx game dhns lo hi pos neg hatch em ep"]
    for i, k in enumerate(order, 1):
        d, (lo, hi) = dhns[k], ci[k]
        pos = f"{d:.4f}" if d >= 0 else "nan"
        neg = f"{d:.4f}" if d < 0 else "nan"
        hatch = f"{d:.4f}" if k in short else "nan"
        lines.append(f"{i} {k} {d:.4f} {lo:.4f} {hi:.4f} {pos} {neg} {hatch} {d - lo:.4f} {hi - d:.4f}")
    (out / "data" / "delta_hns.dat").write_text("\n".join(lines) + "\n")
    labels = [k + ("$^{*}$" if k in short else "") for k in order]
    n = len(order)
    mean_d = float(np.mean([dhns[k] for k in order]))
    short_note = ", ".join(f"{k} ({A['n_seeds'][k, 'exp3']} of {N_SEEDS_FULL} FHR seeds)" for k in short if k in order)
    tex = r"""% Per-game dHNS (FHR - baseline) with the 95 % seed-bootstrap CI, sorted.
  \begin{tikzpicture}
    \begin{axis}[
      atari style,
      width=\textwidth, height=4.9cm,
      xmin=0.4, xmax=@NP@, ymin=-0.7, ymax=1.25,
      xtick={@XTICKS@},
      xticklabels={@XLABELS@},
      x tick label style={rotate=55, anchor=east, font=\scriptsize, yshift=-1pt},
      ylabel={$\Delta$HNS (FHR $-$ baseline)},
      ymajorgrids, xmajorgrids=false,
      ybar, bar width=8pt, area legend,
      legend style={at={(0.03,0.97)}, anchor=north west, legend columns=1},
      legend cell align=left,
      title={Per-game $\Delta$HNS, 5 seeds per arm --- mean $\Delta$HNS @MEAND@, @W@W/@L@L},
    ]
      \draw[black, line width=0.5pt] (axis cs:0.4,0) -- (axis cs:@NP@,0);
      \addplot[bar shift=0pt, fill=fhrcolor, draw=black!60, line width=0.3pt,
               error bars/.cd, y dir=both, y explicit, error bar style={black, line width=0.5pt},
               error mark options={mark size=1.2pt, black}]
        table[x=idx, y=pos, y error minus=em, y error plus=ep] {\figdata/delta_hns.dat};
      \addlegendentry{FHR better}
      \addplot[bar shift=0pt, fill=basecolor, draw=black!60, line width=0.3pt,
               error bars/.cd, y dir=both, y explicit, error bar style={black, line width=0.5pt},
               error mark options={mark size=1.2pt, black}]
        table[x=idx, y=neg, y error minus=em, y error plus=ep] {\figdata/delta_hns.dat};
      \addlegendentry{baseline better}
      % hatch = FHR arm has fewer than 5 seeds
      \addplot[bar shift=0pt, fill=none, draw=none, pattern=north east lines, pattern color=black!70, forget plot]
        table[x=idx, y=hatch] {\figdata/delta_hns.dat};
    \end{axis}
  \end{tikzpicture}
"""
    tex = (tex.replace("@NP@", f"{n + 0.6:.1f}").replace("@XTICKS@", ",".join(str(i) for i in range(1, n + 1)))
              .replace("@XLABELS@", ",".join(labels)).replace("@MEAND@", f"{mean_d:+.3f}")
              .replace("@W@", str(len(A["wins"]))).replace("@L@", str(len(A["losses"]))))
    (out / "fig_delta_hns_body.tex").write_text(BODY_HEADER + tex)
    caption = (rf"Per-game difference in human-normalised score between the FHR arm and the baseline "
               rf"(bar: difference of the per-game mean scores over the seeds; whisker: 95\,\% percentile "
               rf"interval of an unpaired seed bootstrap within the game, {N_BOOT} replicates). Positive "
               rf"bars favour FHR." + (f" Hatched: {short_note}." if short_note else ""))
    write_wrapper(out, "delta_hns", [("fig_delta_hns_body", caption, "fig:atari100k-delta-hns", "t")], force)


def write_curves(out, A, run_dirs, showcase, force, ncols=4, rows_per_fig=4):
    """Curve data for every game; the appendix grids cover the 26 suite games (Enduro is
    exported but not plotted); the showcase figure is one row of the chosen games."""
    games = A["games"]
    suite = [k for k in games if k in A["agg"]]
    (out / "data" / "curves").mkdir(parents=True, exist_ok=True)
    counts = {}
    for k in games:
        for arm in ARMS:
            curves = [c for c in (load_curve(rd) for _, _, rd in run_dirs[k][arm]) if c is not None]
            grid, stack = curve_on_grid(curves)
            n_ok = np.sum(~np.isnan(stack), axis=0)
            with np.errstate(all="ignore"):
                m = np.nanmean(stack, axis=0)
                sd = np.where(n_ok >= 2, np.nanstd(stack, axis=0, ddof=1), 0.0)
            sem = sd / np.sqrt(np.maximum(n_ok, 1))
            counts[k, arm] = len(curves)
            lines = ["step mean std sem n"]
            for g, mm, sdev, se, nn in zip(grid, m, sd, sem, n_ok):
                if np.isnan(mm):
                    continue
                lines.append(f"{g / 1e5:.4f} {mm:.3f} {sdev:.3f} {se:.3f} {nn}")
            (out / "data" / "curves" / f"{k}_{arm}.dat").write_text("\n".join(lines) + "\n")
    n_seed_txt = " or ".join(str(x) for x in sorted({counts[k, a] for k in suite for a in ARMS}))
    band_txt = (rf"Lines: mean over {n_seed_txt} seeds of the raw episode return smoothed with a "
                rf"moving average of {ROLL_W} episodes, on a common grid of {GRID_N} points over the "
                rf"100k environment steps; shaded: $\pm$ one standard deviation across seeds.")

    def panel(k, first, tag):
        nb, nf = counts[k, "baseline"], counts[k, "exp3"]
        title = k + ("" if nb == nf else rf" {{\scriptsize({nf} FHR seeds)}}")
        opts = f", legend to name=atari-lc-legend-{tag}, legend columns=-1" if first else ""
        body = [f"      \\nextgroupplot[title={{{title}}}{opts}]"]
        for arm, col, t in (("baseline", "basecolor", "B"), ("exp3", "fhrcolor", "F")):
            f = rf"\figdata/curves/{k}_{arm}.dat"
            body += [
                rf"      \addplot[name path=hi{t}, draw=none, forget plot] table[x=step, y expr=\thisrow{{mean}}+\thisrow{{std}}] {{{f}}};",
                rf"      \addplot[name path=lo{t}, draw=none, forget plot] table[x=step, y expr=\thisrow{{mean}}-\thisrow{{std}}] {{{f}}};",
                rf"      \addplot[{col}, fill opacity=0.22, draw=none, forget plot] fill between[of=hi{t} and lo{t}];",
                rf"      \addplot[{col}, line width=0.9pt] table[x=step, y=mean] {{{f}}};",
            ]
            if first:
                body.append(rf"      \addlegendentry{{{LEGEND[arm]}}}")
        return "\n".join(body)

    def group_body(chunk, tag, width, height, hsep, vsep):
        nrows = -(-len(chunk) // ncols)
        panels = "\n".join(panel(k, i == 0, tag) for i, k in enumerate(chunk))
        return (r"""  \begin{tikzpicture}
    \begin{groupplot}[
      group style={group size=@NC@ by @NR@, horizontal sep=@HSEP@, vertical sep=@VSEP@},
      atari style,
      width=@W@, height=@H@,
      xmin=0, xmax=1, xtick={0,0.25,0.5,0.75,1},
      xticklabels={0.00,0.25,0.50,0.75,1.00},
      xlabel={Environment Steps ($\times 10^{5}$)}, ylabel={Episode Return},
      title style={font=\small, yshift=-3pt},
      label style={font=\scriptsize}, tick label style={font=\tiny},
      scaled y ticks=false, y tick label style={/pgf/number format/fixed},
      legend style={draw=gray!60, fill=white, font=\small,
                    /tikz/every even column/.append style={column sep=0.6cm}},
    ]
@PANELS@
    \end{groupplot}
    \coordinate (lcx) at ($(group c1r1.west)!0.5!(group c@NC@r1.east)$);   % grid centre
    \node[anchor=north, yshift=-1.05cm] at (lcx |- group c1r@NR@.south)
      {\pgfplotslegendfromname{atari-lc-legend-@TAG@}};
  \end{tikzpicture}
""".replace("@NC@", str(min(ncols, len(chunk)))).replace("@NR@", str(nrows))
           .replace("@HSEP@", hsep).replace("@VSEP@", vsep).replace("@W@", width).replace("@H@", height)
           .replace("@PANELS@", panels).replace("@TAG@", tag))

    # appendix: the 26 suite games, page-sized grids
    per_fig = ncols * rows_per_fig
    chunks = [suite[i:i + per_fig] for i in range(0, len(suite), per_fig)]
    wrappers = []
    for ci, chunk in enumerate(chunks):
        tag = chr(ord("a") + ci)
        body = f"fig_learning_curves_{tag}_body"
        (out / f"{body}.tex").write_text(
            BODY_HEADER + f"% Appendix grid {tag}: games {chunk[0]}..{chunk[-1]} (26 suite games, Enduro excluded).\n"
            + group_body(chunk, tag, r"0.235\textwidth", r"0.2\textwidth", "1.35cm", "1.55cm"))
        cont = "" if ci == 0 else " (continued)"
        caption = (rf"Training curves on the 26 Atari-100k games{cont}: EfficientRainbow baseline and "
                   rf"the same agent with the FHR loss. {band_txt} Panels whose FHR arm has fewer seeds "
                   rf"say so in the title. Games {chunk[0]}--{chunk[-1]}.")
        wrappers.append((body, caption, f"fig:atari100k-learning-curves-{tag}", "p"))
    write_wrapper(out, "learning_curves", wrappers, force)

    # main text: one row of hand-picked games (--showcase)
    (out / "fig_learning_curves_showcase_body.tex").write_text(
        BODY_HEADER + f"% One row of games chosen with --showcase: {' '.join(showcase)}\n"
        + group_body(showcase, "s", r"0.255\textwidth", r"0.21\textwidth", "1.3cm", "1.5cm"))
    caption = (rf"Training curves on {', '.join(showcase[:-1])} and {showcase[-1]}: EfficientRainbow "
               rf"baseline and the same agent with the FHR loss. {band_txt} All 26 games are in "
               rf"Figures~\ref{{fig:atari100k-learning-curves-a}}--\ref{{fig:atari100k-learning-curves-{chr(ord('a') + len(chunks) - 1)}}}.")
    write_wrapper(out, "learning_curves_showcase",
                  [("fig_learning_curves_showcase_body", caption, "fig:atari100k-learning-curves-showcase", "t")], force)
    return counts


PREAMBLE = r"""% Shared preamble for the Atari-100k pgfplots figures (\input this in your preamble).
% Generated by experiments/src/export_atari100k_pgfplots.py.
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\usepgfplotslibrary{groupplots,fillbetween}
\usetikzlibrary{patterns,calc}
% where the .dat files live, relative to the main .tex file
\providecommand{\figdata}{figures/atari100k/data}
% where the fig_*_body.tex files live, relative to the main .tex file
\providecommand{\figtex}{figures/atari100k}
% the two arms: swap these two lines to recolour every figure at once
\definecolor{fhrcolor}{RGB}{31,119,180}    % proposed method (tab:blue, as SR-SAC in D'Oro et al.)
\definecolor{basecolor}{RGB}{255,127,14}   % baseline (tab:orange, as SAC in D'Oro et al.)
% ICLR-paper look: serif document fonts, boxed axes, dashed light grid, no tick marks
\pgfplotsset{
  atari style/.style={
    axis line style={gray!60, line width=0.4pt},
    grid style={dashed, gray!45, line width=0.3pt},
    grid=both,
    tick style={draw=none},
    tick label style={font=\scriptsize},
    label style={font=\small},
    title style={font=\small},
    legend style={font=\scriptsize, draw=gray!60, fill=white, fill opacity=0.9, text opacity=1},
    every axis plot/.append style={line join=round},
  },
}
"""

STANDALONE = r"""\documentclass[10pt]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{amsmath}
\renewcommand{\figdata}{data}
\newcommand{\figtex}{.}
\input{atari100k_pgfplots_preamble}
\begin{document}
\input{fig_model_free_bars}
\input{fig_delta_hns}
\input{fig_learning_curves_showcase}
\clearpage
\input{fig_learning_curves}
\end{document}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "docs" / "figures" / "atari100k")
    ap.add_argument("--rewrite-wrappers", action="store_true",
                    help="also overwrite the fig_*.tex wrappers (captions, labels, placement)")
    ap.add_argument("--showcase", nargs="+", default=None, metavar="GAME",
                    help=f"games for the one-row main-text curve figure (default: "
                         f"{' '.join(SHOWCASE_DEFAULT)}; 'auto' = the 4 largest FHR gains in dHNS)")
    args = ap.parse_args()
    out = args.out
    (out / "data").mkdir(parents=True, exist_ok=True)

    scores, run_dirs, ov = load_scores()
    A = analyse(scores)
    write_bars(out, A, args.rewrite_wrappers)
    write_delta(out, A, args.rewrite_wrappers)
    if args.showcase == ["auto"]:
        showcase = A["order"][-4:][::-1]              # largest FHR gains first
    elif args.showcase or True:
        wanted = args.showcase or SHOWCASE_DEFAULT
        by_lower = {k.lower(): k for k in A["games"]}
        unknown = [g for g in wanted if g.lower() not in by_lower]
        if unknown:
            sys.exit(f"unknown game(s) {unknown}; choose from {A['games']}")
        showcase = [by_lower[g.lower()] for g in wanted]
    counts = write_curves(out, A, run_dirs, showcase, args.rewrite_wrappers)
    (out / "atari100k_pgfplots_preamble.tex").write_text(PREAMBLE)
    (out / "standalone_test.tex").write_text(
        STANDALONE.replace(r"\renewcommand{\figdata}{data}", r"\newcommand{\figdata}{data}"))

    o = A["ours"]
    print(f"exp3 overrides {ov}; {len(A['games'])} games, {sum(A['n_seeds'].values())} runs pooled; "
          f"short of {N_SEEDS_FULL} seeds: {A['short'] or 'none'}")
    for a in ARMS:
        pt, ci, _ = o[a]
        print(f"  {a:9s} mean {pt['mean']:.3f} [{ci['mean'][0]:.3f}, {ci['mean'][1]:.3f}]  "
              f"median {pt['median']:.3f}  IQM {pt['IQM']:.3f} [{ci['IQM'][0]:.3f}, {ci['IQM'][1]:.3f}]")
    d = A["diff"]["mean"]
    print(f"  mean lift {d[0]:+.3f} [{d[1][0]:+.3f}, {d[1][1]:+.3f}] = {100 * d[0] / o['baseline'][0]['mean']:+.1f} %; "
          f"{len(A['wins'])}W/{len(A['losses'])}L")
    print("  per-game dHNS [95 % CI]:")
    for k in A["order"]:
        print(f"    {k:15s} {A['dhns'][k]:+.3f} [{A['ci'][k][0]:+.3f}, {A['ci'][k][1]:+.3f}]")
    print("  curves per arm:", {a: sorted({counts[k, a] for k in A['games']}) for a in ARMS})
    print("  showcase games:", showcase)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
