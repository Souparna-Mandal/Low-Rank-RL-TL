"""Greedy-eval learning curves per tuning variant.

    cd experiments/stable_baselines_3/mountaincar_tuning
    python plot_curves.py                    # every variant with results
    python plot_curves.py cad2_1 cad16_8

One panel per variant: greedy-eval mean reward vs env steps, one color per
arm — thin lines are individual seeds, the thick line is the seed mean on a
common 5k-step grid (curves are truncated at each arm's shortest run, so the
mean never averages a different number of seeds at different x). Panels land
in figures/<variant>_eval.png plus a combined figures/all_variants.png.
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
COLORS = {"baseline": "#444444", "exp1": "#d62728", "exp2": "#1f77b4",
          "exp3": "#2ca02c", "exp4": "#9467bd"}
GRID = np.arange(0, 150001, 5000)


def arm_curves(vdir, arm, per_seed):
    curves = []
    for seed, rel in sorted(per_seed.items()):
        p = vdir / rel / "eval.csv"
        if not p.exists():
            continue
        t = np.genfromtxt(p, delimiter=",", names=True, ndmin=1)
        curves.append((int(seed), t["env_steps"], t["mean_reward"]))
    return curves


def plot_variant(ax, variant, label_arms=True):
    vdir = HERE / variant
    mpath = vdir / "cached" / "sb3_runs_manifest.json"
    if not mpath.exists():
        return False
    manifest = json.loads(mpath.read_text())
    drew = False
    for arm in sorted(manifest["runs"]):
        curves = arm_curves(vdir, arm, manifest["runs"][arm])
        if not curves:
            continue
        color = COLORS.get(arm, "#7f7f7f")
        shortest = min(c[1][-1] for c in curves)
        grid = GRID[GRID <= shortest]
        interp = [np.interp(grid, s, r) for _, s, r in curves]
        for _, s, r in curves:
            ax.plot(s, r, color=color, alpha=0.25, lw=0.8)
        ax.plot(grid, np.mean(interp, axis=0), color=color, lw=2.2,
                label=arm if label_arms else None)
        drew = True
    ax.set_title(variant)
    ax.set_ylim(-205, -85)
    ax.axhline(-110, color="k", ls=":", lw=0.8, alpha=0.5)
    ax.grid(alpha=0.25)
    return drew


def main():
    from make_variants import VARIANTS
    names = sys.argv[1:] or list(VARIANTS)
    figdir = HERE / "figures"
    figdir.mkdir(exist_ok=True)
    done = []
    for variant in names:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if plot_variant(ax, variant):
            ax.set_xlabel("env steps")
            ax.set_ylabel("greedy eval reward (10 eps, fixed seeds)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figdir / f"{variant}_eval.png", dpi=130)
            done.append(variant)
        plt.close(fig)
    if done:
        ncol = min(4, len(done))
        nrow = int(np.ceil(len(done) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.3 * nrow),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, variant in zip(axes.flat, done):
            plot_variant(ax, variant, label_arms=(variant == done[0]))
        for ax in axes.flat[len(done):]:
            ax.axis("off")
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower right")
        fig.suptitle("MountainCar greedy-eval curves: -- solved bar -110; "
                     "thick = seed mean, thin = seeds")
        fig.tight_layout()
        fig.savefig(figdir / "all_variants.png", dpi=130)
        print(f"figures for {done} -> {figdir}")
    else:
        raise SystemExit("no eval.csv results found")


if __name__ == "__main__":
    main()
