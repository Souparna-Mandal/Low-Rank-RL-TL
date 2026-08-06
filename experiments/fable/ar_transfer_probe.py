"""4c — AR(2) coefficient probe: are the recurrence coefficients an
ENVIRONMENT property (stable across seeds, distinct across envs)?

For each (env, seed): train baseline PPO; from update 5 on, fit AR(2) by
pooled ridge least squares to each rollout's value sequences (segments of
length >= 12); save the per-update coefficient trajectory and its mean.

Usage: .venv/bin/python experiments/fable/ar_transfer_probe.py \
           --env CartPole-v1 --seed 0 --episodes 100 --out probe.json
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch

torch.set_num_threads(1)

from agents.variants import get_variant
from ppo_training import _collect_rollout
from runner import make_env  # noqa: E402  (same directory)

DEFAULTS = dict(rollout_steps=1024, minibatch_size=128, update_epochs=6)


def fit_ar2(values, seg_bounds):
    X, y = [], []
    for a, b in seg_bounds:
        v = values[a:b]
        if b - a < 12:
            continue
        for t in range(2, len(v)):
            X.append([v[t - 1], v[t - 2]])
            y.append(v[t])
    if len(y) < 8:
        return None
    X, y = np.asarray(X), np.asarray(y)
    return np.linalg.solve(X.T @ X + 1e-8 * np.eye(2), X.T @ y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    env = make_env(a.env)
    agent = get_variant("baseline").build(env, "cpu", dict(DEFAULTS))
    state, _ = env.reset(seed=a.seed)
    ep_ret, n_eps, update, coeffs = 0.0, 0, 0, []
    while n_eps < a.episodes:
        buf, state, ep_ret, finished = _collect_rollout(agent, env, state, ep_ret)
        if update >= 2:
            c = fit_ar2(buf["values"], buf["seg_bounds"])
            if c is not None:
                coeffs.append([float(c[0]), float(c[1])])
        agent.update(buf)
        n_eps += len(finished)
        update += 1
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mean = np.mean(coeffs, axis=0).tolist() if coeffs else None
    out.write_text(json.dumps({"env": a.env, "seed": a.seed,
                               "coeff_mean": mean, "coeffs": coeffs}))
    print(f"done {a.env}/s{a.seed}: mean AR(2) = {mean} ({len(coeffs)} fits)")


if __name__ == "__main__":
    main()
