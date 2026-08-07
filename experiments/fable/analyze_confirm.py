"""Pre-registered confirmation analysis: paired one-sided permutation tests
on AUC (primary) and final-quarter mean (secondary), fresh seeds 100-119,
plus publication charts (learning curves with 95% CIs; AUC dot-and-bar).

Run: .venv/bin/python experiments/fable/analyze_confirm.py
Outputs: experiments/fable/results/{stats.json, *.png}
"""
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / "runs" / "confirm"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
SEEDS = list(range(100, 120))
RNG = np.random.default_rng(0)

COLOR = {"baseline": "#2a78d6", "ssm_critic": "#eb6834",
         "ar_explore": "#1baf7a", "robust_hd": "#eda100"}
LABEL = {"baseline": "vanilla PPO", "ssm_critic": "SSM critic",
         "ar_explore": "AR-explore", "robust_hd": "robust-HD"}
TXT, MUT = "#333333", "#8a8a8a"


def load(variant, env):
    out = []
    for s in SEEDS:
        f = RUNS / f"{variant}__{env}__s{s}.json"
        out.append(np.array(json.loads(f.read_text())["returns"]))
    return np.stack(out)  # (seeds, episodes)


def auc(mat):
    return mat.mean(axis=1)


def finalq(mat):
    q = mat.shape[1] // 4
    return mat[:, -q:].mean(axis=1)


def paired_perm_p(deltas, n=20000):
    """One-sided (H1: mean delta > 0) paired sign-flip permutation test."""
    obs = deltas.mean()
    signs = RNG.choice([-1.0, 1.0], size=(n, len(deltas)))
    null = (signs * deltas).mean(axis=1)
    return float((np.sum(null >= obs) + 1) / (n + 1))


def ci95(x, n=10000):
    boots = RNG.choice(x, size=(n, len(x))).mean(axis=1)
    return np.percentile(boots, [2.5, 97.5])


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUT)
    ax.tick_params(colors=TXT, labelsize=9)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.7)
    ax.set_axisbelow(True)


def curve_plot(ax, env, arms, title, roll=10):
    ends = []
    for v in arms:
        mat = load(v, env)
        k = np.ones(roll) / roll
        sm = np.stack([np.convolve(row, k, mode="valid") for row in mat])
        x = np.arange(sm.shape[1]) + roll
        mean = sm.mean(axis=0)
        se = sm.std(axis=0, ddof=1) / np.sqrt(sm.shape[0])
        ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se,
                        color=COLOR[v], alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=COLOR[v], linewidth=2, label=LABEL[v])
        ends.append((v, x[-1], mean[-1]))
    # Direct-label only endpoints that are visually separated; the legend
    # carries identity for the rest.
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for v, xe, ye in ends:
        if all(abs(ye - other) > 0.05 * span for w, _, other in ends if w != v):
            ax.annotate(LABEL[v], (xe, ye), xytext=(4, 0),
                        textcoords="offset points", color=TXT, fontsize=9,
                        va="center")
    ax.set_title(title, color=TXT, fontsize=11, loc="left")
    ax.set_xlabel("episode", color=TXT, fontsize=9)
    ax.set_ylabel("episode return", color=TXT, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=TXT, loc="lower right")
    style(ax)


def main():
    stats = {}
    tests = [("Acrobot-v1", "ssm_critic", "PRIMARY"),
             ("CartPole-v1", "ar_explore", "SECONDARY"),
             ("Acrobot-v1", "robust_hd", "RIDER")]
    for env, v, tag in tests:
        b, m = load("baseline", env), load(v, env)
        for name, fn in [("AUC", auc), ("finalQ", finalq)]:
            d = fn(m) - fn(b)
            lo, hi = ci95(d)
            stats[f"{tag}:{v}:{env}:{name}"] = {
                "baseline_mean": round(float(fn(b).mean()), 2),
                "variant_mean": round(float(fn(m).mean()), 2),
                "delta_mean": round(float(d.mean()), 2),
                "delta_ci95": [round(float(lo), 2), round(float(hi), 2)],
                "seed_wins": f"{int((d > 0).sum())}/{len(d)}",
                "p_one_sided": paired_perm_p(d),
            }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    for k, s in stats.items():
        print(f"{k}: d={s['delta_mean']:+.1f} CI{s['delta_ci95']} "
              f"wins={s['seed_wins']} p={s['p_one_sided']:.4f}")

    # Fig 1: learning curves, one panel per registered env comparison.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=150)
    curve_plot(axes[0], "Acrobot-v1", ["baseline", "ssm_critic", "robust_hd"],
               "Acrobot-v1")
    curve_plot(axes[1], "CartPole-v1", ["baseline", "ar_explore"],
               "CartPole-v1")
    fig.suptitle("Confirmation round — mean episode return ±95% CI "
                 "(N=20 fresh seeds/arm, 10-episode rolling mean)",
                 color=TXT, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "learning_curves.png", bbox_inches="tight",
                facecolor="white")

    # Fig 2: per-seed AUC dot strip + mean ±95% CI marker, per env.
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), dpi=150)
    panels = [("Acrobot-v1", ["baseline", "ssm_critic", "robust_hd"]),
              ("CartPole-v1", ["baseline", "ar_explore"])]
    for ax, (env, arms) in zip(axes, panels):
        for i, v in enumerate(arms):
            vals = auc(load(v, env))
            lo, hi = ci95(vals)
            jit = RNG.uniform(-0.14, 0.14, len(vals))
            ax.scatter(i + jit, vals, s=24, color=COLOR[v], alpha=0.75,
                       edgecolors="white", linewidths=0.8, zorder=3)
            ax.errorbar(i + 0.3, vals.mean(),
                        yerr=[[vals.mean() - lo], [hi - vals.mean()]],
                        fmt="o", color=COLOR[v], markersize=8,
                        markeredgecolor=TXT, capsize=4, linewidth=1.6,
                        zorder=4)
            ax.annotate(f"{vals.mean():.0f}", (i + 0.3, vals.mean()),
                        xytext=(9, 0), textcoords="offset points",
                        va="center", color=TXT, fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([LABEL[v] for v in arms], color=TXT, fontsize=9)
        ax.set_xlim(-0.5, len(arms) - 0.2)
        ax.set_title(env, color=TXT, fontsize=11, loc="left")
        ax.set_ylabel("AUC (mean return / episode)", color=TXT, fontsize=9)
        style(ax)
        ax.grid(axis="x", visible=False)
    fig.suptitle("AUC per fresh seed (dots) with arm mean ±95% CI — "
                 "higher = more sample-efficient",
                 color=TXT, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "auc_per_seed.png", bbox_inches="tight", facecolor="white")
    print(f"figures in {OUT}")


if __name__ == "__main__":
    main()
