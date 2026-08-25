"""Error-signal-driven FHR hyperparameter calibration.

Reads a completed PROBE arm (fhr_weight > 0 with an infinite warm-up: trains
bit-identically to the baseline while train_diagnostics.csv logs penalty_raw
next to td_loss) and derives:

  * lambda by MAGNITUDE MATCHING: lambda_rho = rho * median(td_loss) /
    median(penalty_raw) over the tail of training, so the weighted penalty
    contributes a chosen fraction rho of the TD term at equilibrium;
  * fhr_order from the CONVERGED POLICY'S MEASURED RANK: the effective rank
    (99.9% Frobenius energy) of the stacked rollout Hankel of
    Q(s_t, pi(s_t)) under the trained probe policy — Kronecker: a rank-r*
    Hankel sequence satisfies an order-r* recurrence, so r* is the smallest
    order the penalty can enforce without fighting the converged solution.

Run from the experiment dir (like run_sb3_seeds):

    cd experiments/stable_baselines_3/fetch_reach
    python ../src/calibrate_fhr.py --config configs/config_sb3_sac.yaml \
        --probe-arm exp9 --ratios 0.5 2.0
"""
import argparse
import csv
import pathlib
import sys

import numpy as np

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
SRC = SCRIPTS_DIR.parents[2] / "src"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_sb3_seeds as runner                                    # noqa: E402


def _tail_median(rows, col, tail_frac):
    vals = np.array([float(r[col]) for r in rows])
    vals = vals[np.isfinite(vals)]
    tail = vals[int(len(vals) * (1 - tail_frac)):]
    return float(np.median(tail)) if len(tail) else float("nan")


def calibrate(probe_arm="exp9", exp_dir=None, config=runner.CONFIG,
              tail_frac=0.5, ratios=(0.5, 2.0), n_rollouts=3, base_seed=52,
              energy_frac=0.999, device="cpu"):
    """-> dict with per-seed magnitudes/ranks and the suggested lambda/order.

    lambda uses the seed-mean of the tail-median td/penalty ratio; fhr_order
    is the rounded seed-mean effective rank of the converged Q(s, pi(s))
    rollout Hankel (never below 2 — order-1 pure AR cannot represent a
    Bellman-consistent sequence).
    """
    from analysis.low_rank.continuous_rollout import hankel_rollout_continuous
    from analysis.low_rank.rank import energy_rank

    out = {"seeds": [], "td_median": [], "penalty_median": [], "q_rank": [],
           "pi_rank": []}
    for run in runner.load_runs(probe_arm, exp_dir=exp_dir, config=config):
        rows = list(csv.DictReader(open(run["run_dir"]
                                        / "train_diagnostics.csv")))
        td = _tail_median(rows, "td_loss", tail_frac)
        pen = _tail_median(rows, "penalty_raw", tail_frac)
        _, adapter = runner.load_run_model(run["run_dir"], device=device)
        env = runner._make_env(run["cfg"])
        mats = hankel_rollout_continuous(adapter, env, n_rollouts=n_rollouts,
                                         base_seed=base_seed)
        env.close()
        h_q = mats[0] if isinstance(mats, tuple) else mats
        sv_q = np.linalg.svd(h_q, compute_uv=False)
        q_rank = energy_rank(sv_q, energy_frac)
        pi_ranks = []
        if isinstance(mats, tuple):
            for h_a in mats[1:]:
                sv = np.linalg.svd(h_a, compute_uv=False)
                pi_ranks.append(energy_rank(sv, energy_frac))
        out["seeds"].append(run["seed"])
        out["td_median"].append(td)
        out["penalty_median"].append(pen)
        out["q_rank"].append(q_rank)
        out["pi_rank"].append(pi_ranks)

    ratio = float(np.mean(np.array(out["td_median"])
                          / np.array(out["penalty_median"])))
    out["td_over_penalty"] = ratio
    out["lambda"] = {rho: rho * ratio for rho in ratios}
    out["fhr_order"] = max(2, int(round(float(np.mean(out["q_rank"])))))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=runner.CONFIG)
    parser.add_argument("--probe-arm", default="exp9")
    parser.add_argument("--tail-frac", type=float, default=0.5,
                        help="fraction of training rows the medians use "
                             "(from the end)")
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.5, 2.0],
                        help="target penalty/TD contribution ratios rho")
    parser.add_argument("--n-rollouts", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cal = calibrate(probe_arm=args.probe_arm, config=args.config,
                    tail_frac=args.tail_frac, ratios=tuple(args.ratios),
                    n_rollouts=args.n_rollouts, device=args.device)
    for i, seed in enumerate(cal["seeds"]):
        print(f"seed {seed}: td_loss median {cal['td_median'][i]:.4g}, "
              f"penalty_raw median {cal['penalty_median'][i]:.4g}, "
              f"Q-trace rank {cal['q_rank'][i]}, "
              f"pi-trace ranks {cal['pi_rank'][i]}")
    print(f"td/penalty ratio (seed mean): {cal['td_over_penalty']:.3g}")
    for rho, lam in cal["lambda"].items():
        print(f"suggested fhr_weight @ rho={rho:g}: {lam:.3g}")
    print(f"suggested fhr_order (converged Q-trace rank): {cal['fhr_order']}")


if __name__ == "__main__":
    main()
