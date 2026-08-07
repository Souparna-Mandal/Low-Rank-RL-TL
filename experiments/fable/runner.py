"""Run ONE training run of a variant and save per-episode returns as JSON.

Usage:
  .venv/bin/python experiments/fable/runner.py --variant baseline \
      --env CartPole-v1 --seed 0 --episodes 200 --out runs/x.json \
      [--set rollout_steps=1024 aux_weight=0.1 ...]

Fixed episode budget, no solved-early-stop (fair AUC comparison across arms).
Single-threaded torch so four runs can share the box.
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch

torch.set_num_threads(1)

import gymnasium as gym

from agents.variants import get_variant
from environments.base_env import make_environment
from ppo_training import _collect_rollout as default_collect

DEFAULTS = dict(rollout_steps=1024, minibatch_size=128, update_epochs=6)

# Envs needing the wrapper keys from base_env (one-hot obs, discretised
# actions, episode caps). Everything else goes through plain gym.make.
ENV_KWARGS = {
    "Pendulum-v1": {"discrete_action_bins": 9},
    "CliffWalking-v1": {"one_hot_obs": True, "time_limit": 200},
}
COMMON = {"discrete_config": None, "normalise": {"action": {}, "state": {}},
          "clip": {"action": False, "state": []}}


def make_env(env_name, continuous=False, normalise=False, gamma=0.99):
    """continuous=True keeps a Box action space instead of binning it, and
    turns on ClipAction so the unbounded Gaussian's tails are absorbed at the
    env boundary (how SB3/CleanRL bound continuous PPO).

    normalise=True adds the CleanRL running-statistics stack (obs and reward,
    both clipped to +-10). Returns are then reported twice: normalised (what
    the agent optimises) and raw.
    """
    if not (continuous or normalise):
        if env_name in ENV_KWARGS:
            return make_environment(env_name, **ENV_KWARGS[env_name], **COMMON)
        return gym.make(env_name)
    kwargs = dict(COMMON)
    if continuous:
        kwargs["clip"] = {"action": True, "state": []}
    elif env_name in ENV_KWARGS:
        kwargs.update(ENV_KWARGS[env_name])
    if normalise:
        kwargs["normalise"] = dict(
            kwargs["normalise"],
            running={"obs": True, "reward": True, "gamma": gamma,
                     "clip_obs": 10.0, "clip_reward": 10.0})
    return make_environment(env_name, **kwargs)


def parse_value(s):
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def run(variant_name, env_name, seed, episodes, overrides, continuous=False,
        normalise=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mod = get_variant(variant_name)
    env = make_env(env_name, continuous=continuous, normalise=normalise,
                   gamma=overrides.get("discount_factor", 0.99))
    agent = mod.build(env, "cpu", overrides)
    collect = getattr(mod, "collect_rollout", None) or default_collect

    state, _ = env.reset(seed=seed)
    ep_ret = 0.0
    returns, raw_returns = [], []
    t0 = time.time()
    empty = 0
    while len(returns) < episodes:
        buf, state, ep_ret, finished = collect(agent, env, state, ep_ret)
        agent.update(buf)
        returns.extend(finished)
        # equal to `finished` unless reward normalisation is on
        raw_returns.extend(buf.get("raw_returns") or finished)
        empty = 0 if finished else empty + 1
        if empty >= 100:
            raise RuntimeError("no episode finished in 100 rollouts")
    return {
        "variant": variant_name, "env": env_name, "seed": seed,
        "episodes": episodes, "overrides": overrides,
        "continuous": continuous, "normalise": normalise,
        # what the agent optimised (normalised when normalise=True)
        "returns": [float(r) for r in returns[:episodes]],
        # unnormalised, always comparable across runs
        "raw_returns": [float(r) for r in raw_returns[:episodes]],
        "wall_s": round(time.time() - t0, 1),
        "diag": getattr(agent, "diag", None),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--env", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--continuous", action="store_true",
                   help="keep the Box action space (Gaussian policy + "
                        "ClipAction) instead of binning it to Discrete")
    p.add_argument("--normalise", action="store_true",
                   help="running obs/reward normalisation (CleanRL stack); "
                        "results carry both returns and raw_returns")
    a = p.parse_args()
    overrides = dict(DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=", 1)
        overrides[k] = parse_value(v)
    result = run(a.variant, a.env, a.seed, a.episodes, overrides,
                 continuous=a.continuous, normalise=a.normalise)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result))
    raw = f" raw={np.mean(result['raw_returns'][-20:]):.1f}" if a.normalise else ""
    print(f"done {a.variant}/{a.env}/s{a.seed}: "
          f"mean_last20={np.mean(result['returns'][-20:]):.1f}{raw} "
          f"wall={result['wall_s']}s")


if __name__ == "__main__":
    main()
