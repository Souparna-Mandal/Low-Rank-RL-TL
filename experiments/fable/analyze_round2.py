"""Round-2 consolidated analysis: headline + ablation + rank robustness,
per-env paired ADVANTAGE curves (variant minus baseline, per seed), and the
cross-environment effect map. Tolerates missing ssm_auto files (skips).

Run: .venv/bin/python experiments/fable/analyze_round2.py
Outputs: experiments/fable/results/{summary_headline,advantage_curves,
         cross_env_summary}.png + round2_stats.json
"""
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(0)

COLOR = {"baseline": "#2a78d6", "ssm_critic": "#eb6834",
         "ssm_auto": "#e87ba4", "gru_critic": "#4a3aa7"}
LABEL = {"baseline": "vanilla PPO", "ssm_critic": "SSM critic",
         "ssm_auto": "SSM-auto (AR trick)", "gru_critic": "GRU critic"}
TXT, MUT = "#333333", "#8a8a8a"

# env -> (runs dir, seeds, N label) for the ssm-vs-baseline comparison
SOURCES = {
    "Acrobot-v1": ("runs/confirm", range(100, 120)),
    "LunarLander-v3": ("runs/confirm_lunar", range(100, 120)),
    "CartPole-v1": ("runs/explore", range(5)),
    "Pendulum-v1": ("runs/explore2", range(5)),
    "MountainCar-v0": ("runs/explore2", range(5)),
    "CliffWalking-v1": ("runs/explore2", range(5)),
}
AUTO_DIR = {"CartPole-v1": "runs/explore2", "Acrobot-v1": "runs/explore2",
            "Pendulum-v1": "runs/explore2", "MountainCar-v0": "runs/explore2",
            "CliffWalking-v1": "runs/explore2"}


def load(d, v, env, seeds):
    mats = []
    for s in seeds:
        f = HERE / d / f"{v}__{env}__s{s}.json"
        if f.exists():
            mats.append(json.loads(f.read_text()))
    if not mats:
        return None, None
    return np.stack([np.array(m["returns"]) for m in mats]), mats


def ci95(x, n=10000):
    b = RNG.choice(x, size=(n, len(x))).mean(axis=1)
    return np.percentile(b, [2.5, 97.5])


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUT)
    ax.tick_params(colors=TXT, labelsize=8)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.7)
    ax.set_axisbelow(True)


def roll(mat, k=10):
    ker = np.ones(k) / k
    return np.stack([np.convolve(r, ker, mode="valid") for r in mat])


def curves(ax, series, roll_k=10):
    for v, mat in series:
        sm = roll(mat, roll_k)
        x = np.arange(sm.shape[1]) + roll_k
        mean, se = sm.mean(0), sm.std(0, ddof=1) / np.sqrt(sm.shape[0])
        ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se,
                        color=COLOR[v], alpha=0.14, linewidth=0)
        ax.plot(x, mean, color=COLOR[v], linewidth=2, label=LABEL[v])
    ax.legend(frameon=False, fontsize=8, labelcolor=TXT, loc="lower right")


def main():
    stats = {}

    # ---- Fig A: headline (Acrobot curves incl. ablation) + rank sweep ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)
    b5, _ = load("runs/explore", "baseline", "Acrobot-v1", range(5))
    s5, _ = load("runs/explore", "ssm_critic", "Acrobot-v1", range(5))
    g5, _ = load("runs/explore2", "gru_critic", "Acrobot-v1", range(5))
    curves(axes[0], [("baseline", b5), ("ssm_critic", s5), ("gru_critic", g5)])
    axes[0].set_title("Acrobot-v1 — linear vs nonlinear recurrence (N=5 paired)",
                      color=TXT, fontsize=10, loc="left")
    axes[0].set_xlabel("episode", color=TXT, fontsize=9)
    axes[0].set_ylabel("episode return", color=TXT, fontsize=9)
    style(axes[0])

    ranks, means, los, his = [], [], [], []
    for r, d in [(2, "runs/rank2"), (4, "runs/rank4"), (8, "runs/explore"),
                 (16, "runs/rank16")]:
        m, _ = load(d, "ssm_critic", "Acrobot-v1", range(5))
        aucs = m.mean(axis=1)
        lo, hi = ci95(aucs)
        ranks.append(r); means.append(aucs.mean()); los.append(lo); his.append(hi)
    base_auc = b5.mean(axis=1).mean()
    ax = axes[1]
    ax.axhline(base_auc, color=COLOR["baseline"], linewidth=1.5, linestyle="--")
    ax.annotate("vanilla PPO", (16, base_auc), xytext=(0, 5),
                textcoords="offset points", ha="right", color=TXT, fontsize=8)
    ax.errorbar(ranks, means,
                yerr=[np.array(means) - np.array(los),
                      np.array(his) - np.array(means)],
                fmt="o-", color=COLOR["ssm_critic"], linewidth=2, capsize=4,
                markersize=7, markeredgecolor="white")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks); ax.set_xticklabels(ranks)
    ax.set_title("Acrobot-v1 — AUC vs SSM rank (N=5, ±95% CI)",
                 color=TXT, fontsize=10, loc="left")
    ax.set_xlabel("SSM hidden rank", color=TXT, fontsize=9)
    ax.set_ylabel("AUC", color=TXT, fontsize=9)
    style(ax)
    fig.suptitle("SSM critic: the effect needs the LINEAR low-rank recurrence "
                 "(GRU hurts) and is robust across ranks",
                 color=TXT, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "summary_headline.png", bbox_inches="tight",
                facecolor="white")

    # ---- Fig B: paired ADVANTAGE curves per env ----
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), dpi=150)
    for ax, (env, (d, seeds)) in zip(axes.flat, SOURCES.items()):
        b, _ = load(d, "baseline", env, seeds)
        s, _ = load(d, "ssm_critic", env, seeds)
        delta = roll(s) - roll(b)  # paired per-seed advantage
        x = np.arange(delta.shape[1]) + 10
        mean, se = delta.mean(0), delta.std(0, ddof=1) / np.sqrt(delta.shape[0])
        ax.axhline(0, color=MUT, linewidth=1)
        ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se,
                        color=COLOR["ssm_critic"], alpha=0.18, linewidth=0)
        ax.plot(x, mean, color=COLOR["ssm_critic"], linewidth=2)
        au, _ = load(AUTO_DIR.get(env, ""), "ssm_auto", env, range(5)) \
            if env in AUTO_DIR else (None, None)
        if au is not None and b.shape[0] >= 5:
            b5x, _ = load("runs/explore" if env in ("CartPole-v1",)
                          else AUTO_DIR[env], "baseline", env, range(5))
            if env == "Acrobot-v1":
                b5x, _ = load("runs/explore", "baseline", env, range(5))
            if b5x is not None and b5x.shape[0] == au.shape[0]:
                da = roll(au) - roll(b5x)
                ax.plot(np.arange(da.shape[1]) + 10, da.mean(0),
                        color=COLOR["ssm_auto"], linewidth=1.6)
        ax.set_title(f"{env}  (N={len(list(seeds))})", color=TXT, fontsize=10,
                     loc="left")
        ax.set_xlabel("episode", color=TXT, fontsize=8)
        ax.set_ylabel("Δ return vs baseline", color=TXT, fontsize=8)
        style(ax)
    handles = [plt.Line2D([], [], color=COLOR["ssm_critic"], lw=2,
                          label="SSM critic − baseline (±95% CI)"),
               plt.Line2D([], [], color=COLOR["ssm_auto"], lw=1.6,
                          label="SSM-auto − baseline (N=5)")]
    fig.legend(handles=handles, frameon=False, fontsize=9, labelcolor=TXT,
               loc="lower center", ncol=2)
    fig.suptitle("Advantage curves — paired per-seed improvement over vanilla "
                 "PPO (10-episode rolling)", color=TXT, fontsize=12, x=0.02,
                 ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(OUT / "advantage_curves.png", bbox_inches="tight",
                facecolor="white")

    # ---- Fig C: cross-env effect map (dAUC, normalized per env scale) ----
    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=150)
    rows = []
    for env, (d, seeds) in SOURCES.items():
        b, _ = load(d, "baseline", env, seeds)
        s, _ = load(d, "ssm_critic", env, seeds)
        dauc = s.mean(axis=1) - b.mean(axis=1)
        scale = np.abs(b.mean(axis=1).mean()) or 1.0
        rel = 100 * dauc / scale
        lo, hi = ci95(rel)
        rows.append((env, rel.mean(), lo, hi, len(dauc)))
        stats[f"{env}:dAUC"] = {"abs": round(float(dauc.mean()), 1),
                                "rel_pct": round(float(rel.mean()), 1),
                                "n": len(dauc)}
    rows.sort(key=lambda r: r[1])
    y = np.arange(len(rows))
    ax.axvline(0, color=MUT, linewidth=1)
    for i, (env, m, lo, hi, n) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=COLOR["ssm_critic"], linewidth=2)
        ax.plot(m, i, "o", color=COLOR["ssm_critic"], markersize=8,
                markeredgecolor="white")
        ax.annotate(f"N={n}", (hi, i), xytext=(6, 0),
                    textcoords="offset points", va="center", color=MUT,
                    fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], color=TXT, fontsize=9)
    ax.set_xlabel("ΔAUC vs baseline, % of baseline |AUC| (±95% CI)",
                  color=TXT, fontsize=9)
    ax.set_title("Where the SSM critic helps — cross-environment effect map",
                 color=TXT, fontsize=11, loc="left")
    style(ax)
    ax.grid(axis="x", color="#e8e8e8", linewidth=0.7)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "cross_env_summary.png", bbox_inches="tight",
                facecolor="white")

    # ---- ssm_auto orders chosen per env (the AR-trick diagnostic) ----
    for env in AUTO_DIR:
        _, metas = load(AUTO_DIR[env], "ssm_auto", env, range(5))
        if metas:
            stats[f"{env}:ssm_auto"] = {
                "orders": [m["diag"].get("order") for m in metas],
                "ranks": [m["diag"].get("rank") for m in metas],
                "auc": round(float(np.mean([np.mean(m["returns"]) for m in metas])), 1),
            }
    (OUT / "round2_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
