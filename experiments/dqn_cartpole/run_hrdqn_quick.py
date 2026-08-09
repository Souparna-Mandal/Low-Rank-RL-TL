"""Quick CartPole experiment: HR-DQN (`gated_order1`) vs the same agent with the
penalty switched off, one seed, full artifact trail for the result viewer app.

    python experiments/dqn_cartpole/run_hrdqn_quick.py            # seed 0, both variants
    python experiments/dqn_cartpole/run_hrdqn_quick.py --seed 3 --variants hrdqn

Both variants read hrdqn-config.yaml; the baseline only overrides
`hankel_weight: 0.0`, which reduces HankelDQNAgent to exact classical
Double-DQN (same TD computation, same sampling distribution, no penalty
machinery touched) — so the runs differ in the regulariser and nothing else.

Each run writes runs/<variant>_s<seed>/ via RunLogger:
    rewards.csv, train_diagnostics.csv (one row per train() call: td_loss,
    penalty_raw/weighted, lambda_eff, batch_eff_rank, rel_tail, gate fractions),
    eval.csv (20 greedy episodes), rank_stats.csv, hankel_sweep.csv,
    figures/, trajectories/, checkpoints/, config.yaml
which is exactly what result_viewer_app/rank_viewer.py plots. Results are also
cached as results_hrdqn/<variant>_s<seed>.npz keyed by the resolved config, so
re-running is a no-op until the config actually changes.
"""
import argparse
import csv
import json
import pathlib
import random
import shutil
import sys
import time

import numpy as np
import torch

EXP_DIR = pathlib.Path(__file__).resolve().parent
SRC = EXP_DIR.parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment import load_config, build_env, build_agent, train          # noqa: E402
from agents.hankel_dqn_agent import HankelDQNAgent                         # noqa: E402
from analysis.run_logger import RunLogger                                  # noqa: E402
from analysis.low_rank.hankel_policy import (collect_hankel_sequences,     # noqa: E402
                                             _hankel_from_sequence)
from analysis.low_rank.rank import compute_rank_metrics                    # noqa: E402
from training import _greedy_episode_return                                # noqa: E402

CONFIG = EXP_DIR / "hrdqn-config.yaml"
RESULTS = EXP_DIR / "results_hrdqn"
EVAL_EPISODES = 20
AUC_EPISODES = 600      # campaign metric: mean reward over the first 600 episodes

VARIANTS = {
    # config as-is: r=1, lambda=1e-2, warm-up 2000 + ramp 2000, rho=0.25
    "hrdqn": {},
    # the only difference: no penalty term at all
    "hrdqn-baseline": dict(hankel_weight=0.0),
}


class QNetwork(torch.nn.Module):
    """Maps a state (obs_dim,) -> Q-values (n_actions,). Built by the agent via
    q_network(**nn_extra_kwargs). Same net as exp1_hankel.ipynb."""

    def __init__(self, in_dim, out_dim, hidden_sizes=(128, 128)):
        super().__init__()
        layers, last = [], in_dim
        for h in hidden_sizes:
            layers += [torch.nn.Linear(last, h), torch.nn.ReLU()]
            last = h
        layers.append(torch.nn.Linear(last, out_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def resolve_cfg(overrides: dict, seed: int) -> dict:
    """The exact config a run uses (yaml + variant overrides + seed)."""
    cfg = load_config(CONFIG)
    cfg["experiment"]["seed"] = seed
    cfg["agent"].update(overrides)
    return cfg


def cache_key(cfg: dict) -> str:
    """Canonical JSON of every config section that determines the run's outcome
    (device deliberately excluded), stored inside the npz so a run re-runs
    automatically once the config that produced it no longer matches."""
    parts = {k: cfg[k] for k in ("environment", "network", "agent", "training", "analysis")}
    parts["seed"] = cfg["experiment"]["seed"]
    return json.dumps(parts, sort_keys=True, default=str)


def is_cached(out_path: pathlib.Path, key: str) -> bool:
    if not out_path.exists():
        return False
    with np.load(out_path) as d:
        return "cfg_json" in d.files and str(d["cfg_json"]) == key


def solve_episode(rewards, window: int, threshold: float):
    """First episode whose trailing rolling mean over `window` crosses
    `threshold` — the training loop's own early-stop rule."""
    for ep in range(window, len(rewards)):
        if np.mean(rewards[ep - window:ep]) > threshold:
            return ep
    return None


def auc(rewards, n: int, pad: float):
    """Mean reward over the first n episodes, an early-stopped (solved) run
    padded at `pad` so stopping early is not scored as a gap."""
    r = list(rewards[:n])
    return float(np.mean(r + [pad] * (n - len(r))))


def run_one(cfg: dict, out_path: pathlib.Path, run_id: str, key: str) -> None:
    seed = cfg["experiment"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = build_env(cfg)
    nn_extra = {"in_dim": env.observation_space.shape[0], "out_dim": env.action_space.n,
                "hidden_sizes": cfg["network"]["hidden_sizes"]}
    agent = build_agent(cfg, env, q_network=QNetwork, nn_extra_kwargs=nn_extra,
                        agent_cls=HankelDQNAgent)

    run_dir = EXP_DIR / "runs" / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)  # stale logs from an interrupted / old-config run
    logger = RunLogger(EXP_DIR, config_path=CONFIG, run_id=run_id)
    rewards = train(cfg, agent, env, run_logger=logger)

    eps = agent.epsilon
    agent.epsilon = 0.0  # greedy eval + on-policy rank probe
    evals = [_greedy_episode_return(agent, env, seed=30_000 + i) for i in range(EVAL_EPISODES)]
    seqs = collect_hankel_sequences(agent, env, seed=777)
    eff_rank_q = compute_rank_metrics(_hankel_from_sequence(np.asarray(seqs["Hankel Q"])))[0]
    agent.epsilon = eps

    with open(logger.dir / "eval.csv", "w", newline="") as f:  # viewer eval tile
        w = csv.writer(f)
        w.writerow(["episode", "reward"])
        w.writerows(enumerate(evals))

    np.savez(out_path, rewards=np.array(rewards, float), evals=np.array(evals, float),
             eff_rank_q=eff_rank_q, nan_skips=agent.nan_skips, cfg_json=key)
    env.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0, help="single training seed (default: 0)")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS),
                    choices=list(VARIANTS), help="which variants to run")
    ap.add_argument("--force", action="store_true", help="ignore the npz cache and re-run")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    for variant in args.variants:
        out = RESULTS / f"{variant}_s{args.seed}.npz"
        cfg = resolve_cfg(VARIANTS[variant], args.seed)
        key = cache_key(cfg)
        if not args.force and is_cached(out, key):
            print("cached:", out.name)
            continue
        if out.exists():
            print("re-running:", out.name)
        t0 = time.time()
        run_one(cfg, out, run_id=f"{variant}_s{args.seed}", key=key)
        print(f"{out.name}: {time.time() - t0:.0f}s")

    cfg = resolve_cfg({}, args.seed)
    window = cfg["training"]["early_stopping_patience_eps"]
    threshold = cfg["training"]["solved_reward"]
    print(f"\n{'variant':16s} {'solve-ep':>9s} {'AUC600':>8s} "
          f"{'eval20 mean±std':>20s} {'final Hankel-Q eff-rank':>24s} {'nan_skips':>10s}")
    for variant in args.variants:
        out = RESULTS / f"{variant}_s{args.seed}.npz"
        if not out.exists():
            continue
        with np.load(out) as d:
            rw, ev = d["rewards"], d["evals"]
            solved = solve_episode(rw, window, threshold)
            print(f"{variant:16s} {str(solved) if solved else 'unsolved':>9s} "
                  f"{auc(rw, AUC_EPISODES, 500.0):8.1f} "
                  f"{ev.mean():13.1f} ± {ev.std():4.1f} "
                  f"{int(d['eff_rank_q']):24d} {int(d['nan_skips']):10d}")
    runs = ", ".join(f"{v}_s{args.seed}" for v in args.variants)
    print(f"\nviewer: python result_viewer_app/rank_viewer.py   -> dqn_cartpole runs: {runs}")


if __name__ == "__main__":
    main()
