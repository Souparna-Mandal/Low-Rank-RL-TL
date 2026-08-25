"""Sample-efficiency summary across tuning variants.

    cd experiments/stable_baselines_3/mountaincar_tuning
    python summarize.py                      # all variants with results
    python summarize.py cad2_1 cad16_8

Per (variant, arm, seed) two views:
  * greedy eval curve (eval.csv, deterministic policy, paired reset seeds) —
    the primary sample-efficiency measure: env steps to first eval mean >=
    {-160, -140, -120, -110} and eval mean at fixed step budgets;
  * eps-greedy training curve (rewards.csv, rolling-10) as the secondary view.
Seed-aggregated rows print mean over seeds; a threshold never reached counts
as the run's total steps (censored — marked '>' when any seed is censored).
Full per-seed rows land in summary.csv next to this script.
"""
import csv
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
BARS = (-160, -140, -120, -110)
BUDGETS = (30000, 60000, 100000)


def eval_curve(run_dir):
    p = run_dir / "eval.csv"
    if not p.exists():
        return None
    t = np.genfromtxt(p, delimiter=",", names=True, ndmin=1)
    return t["env_steps"], t["mean_reward"]


def train_curve(run_dir):
    t = np.loadtxt(run_dir / "rewards.csv", delimiter=",", skiprows=1, ndmin=2)
    return np.cumsum(t[:, 2]), t[:, 1]


def steps_to(steps, values, bar):
    idx = np.flatnonzero(values >= bar)
    return float(steps[idx[0]]) if idx.size else None


def value_at(steps, values, budget):
    mask = steps <= budget
    return float(values[mask][-1]) if mask.any() else float("nan")


def collect(variants):
    rows = []
    for variant in variants:
        vdir = HERE / variant
        mpath = vdir / "cached" / "sb3_runs_manifest.json"
        if not mpath.exists():
            continue
        manifest = json.loads(mpath.read_text())
        for arm, per_seed in manifest["runs"].items():
            for seed, rel in per_seed.items():
                run_dir = vdir / rel
                if not (run_dir / "rewards.csv").exists():
                    continue
                ts, tr = train_curve(run_dir)
                row = {"variant": variant, "arm": arm, "seed": int(seed),
                       "total_steps": float(ts[-1]),
                       "train_final25": float(np.mean(tr[-25:]))}
                rm = np.convolve(tr, np.ones(10) / 10, "valid")
                for bar in BARS:
                    row[f"train_to{bar}"] = steps_to(ts[9:], rm, bar)
                ev = eval_curve(run_dir)
                if ev is not None:
                    es, er = ev
                    for bar in BARS:
                        row[f"eval_to{bar}"] = steps_to(es, er, bar)
                    for b in BUDGETS:
                        row[f"eval@{b // 1000}k"] = value_at(es, er, b)
                    row["eval_final"] = float(er[-1])
                rows.append(row)
    return rows


def print_table(rows):
    key = lambda r: (r["variant"], r["arm"])
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)

    def agg(g, col, censor=False):
        vals, censored = [], False
        for r in g:
            v = r.get(col)
            if v is None:
                v, censored = (r["total_steps"], True) if censor else (np.nan, False)
            vals.append(v)
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if not vals:
            return "-"
        mark = ">" if censored else ""
        mean = np.mean(vals)
        return f"{mark}{mean / 1000:.0f}k" if censor else f"{mean:.1f}"

    cols = ([f"eval_to{b}" for b in BARS]
            + [f"eval@{b // 1000}k" for b in BUDGETS] + ["eval_final", "train_final25"])
    head = (f"{'variant':10s}{'arm':10s}{'n':3s}"
            + "".join(f"{'ev>' + str(b):>9s}" for b in BARS)
            + "".join(f"{'ev@' + str(b // 1000) + 'k':>9s}" for b in BUDGETS)
            + f"{'ev_fin':>9s}{'tr_fin25':>9s}")
    print(head)
    for (variant, arm), g in sorted(groups.items()):
        line = f"{variant:10s}{arm:10s}{len(g):<3d}"
        for c in cols:
            censor = c.startswith("eval_to")
            line += f"{agg(g, c, censor=censor):>9s}"
        print(line)


def main():
    from make_variants import VARIANTS
    names = sys.argv[1:] or list(VARIANTS)
    rows = collect(names)
    if not rows:
        raise SystemExit("no completed runs found")
    fieldnames = sorted({k for r in rows for k in r},
                        key=lambda k: (k not in ("variant", "arm", "seed"), k))
    with open(HERE / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print_table(rows)
    print(f"\nper-seed rows -> {HERE / 'summary.csv'}")


if __name__ == "__main__":
    main()
