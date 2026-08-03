"""Track AR coefficients of the value signal across training.

Usage: python drift_collect.py <env_dir> <variant> <seed>
cwd must be the experiment dir. Every PROBE_EVERY episodes: freeze policy,
roll N_PROBE greedy trajectories (fixed seeds, argmax on policy_net — no
agent state touched), fit global AR(2) and AR(1), record coefficients.
Writes results_recurrence/drift_<variant>_s<seed>.npz. Repo home of the scratch drift_collect script.
"""
import pathlib
import random
import sys

import numpy as np
import torch

REPO = pathlib.Path("/Users/souparna/ICL_Thesis/Low-Rank-RL-TL")
sys.path.insert(0, str(REPO / "src"))

from experiment import load_config, build_env, build_agent, train
from agents.hankel_dqn_agent import HankelDQNAgent
from analysis.low_rank.recurrence import fit_ar

VARIANTS = {
    "baseline": dict(hankel_weight=0.0),
    "progress_order2": dict(hankel_weight=1e-2, hankel_order=2, gate_threshold=0.25,
                            engage_reward_threshold=100.0, engage_reward_window=10,
                            ramp_grad_steps=2000),
}
PROBE_EVERY, N_PROBE, PROBE_SEED0 = 25, 8, 40_000


class QNetwork(torch.nn.Module):
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


def greedy_values(agent, env, seed):
    vals = []
    state, _ = env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            q = agent.policy_net(t).squeeze(0)
        a = int(q.argmax())
        vals.append(float(q[a]))
        state, _, terminated, truncated, _ = env.step(a)
    return np.asarray(vals)


def main(variant, seed, out_path):
    cfg = load_config("config_hankel.yaml")
    cfg["experiment"]["seed"] = seed
    cfg["agent"].update(VARIANTS[variant])
    cfg["analysis"] = {"ep_freq": 10 ** 9, "methods": []}
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    env = build_env(cfg)
    probe_env = build_env(cfg)
    nn_extra = {"in_dim": env.observation_space.shape[0], "out_dim": env.action_space.n,
                "hidden_sizes": cfg["network"]["hidden_sizes"]}
    agent = build_agent(cfg, env, q_network=QNetwork, nn_extra_kwargs=nn_extra,
                        agent_cls=HankelDQNAgent)

    rec = {"episode": [], "c1": [], "c2": [], "c1_ar1": [], "mean_len": [],
           "fit_nrmse": []}

    def probe(episode):
        seqs = [greedy_values(agent, probe_env, PROBE_SEED0 + i) for i in range(N_PROBE)]
        seqs = [s for s in seqs if len(s) >= 6]
        rec["episode"].append(episode)
        rec["mean_len"].append(np.mean([len(s) for s in seqs]) if seqs else np.nan)
        if not seqs:
            for k in ("c1", "c2", "c1_ar1", "fit_nrmse"):
                rec[k].append(np.nan)
            return
        c2 = fit_ar(seqs, 2)
        c1 = fit_ar(seqs, 1)
        resid = np.concatenate([(s[2:] - (c2[0] * s[1:-1] + c2[1] * s[:-2])) for s in seqs])
        rms = np.sqrt(np.mean(np.concatenate([s[2:] for s in seqs]) ** 2))
        rec["c1"].append(c2[0]); rec["c2"].append(c2[1]); rec["c1_ar1"].append(c1[0])
        rec["fit_nrmse"].append(float(np.sqrt(np.mean(resid ** 2)) / max(rms, 1e-12)))

    n_calls = [0]

    def train_hook(_orig=agent.train):
        d = _orig()
        n_calls[0] += 1
        if n_calls[0] % PROBE_EVERY == 0:
            probe(n_calls[0])
        return d

    agent.train = train_hook
    rewards = train(cfg, agent, env)
    probe(n_calls[0])  # final policy
    np.savez(out_path, train_rewards=np.array(rewards, float),
             **{k: np.array(v, float) for k, v in rec.items()})
    env.close(); probe_env.close()
    print(f"wrote drift_{variant}_s{seed}: {len(rec['episode'])} probes, "
          f"final c=({rec['c1'][-1]:.3f}, {rec['c2'][-1]:.3f})")


if __name__ == "__main__":
    _env_dir, variant, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    out = pathlib.Path("results_recurrence")
    out.mkdir(exist_ok=True)
    p = out / f"drift_{variant}_s{seed}.npz"
    if p.exists():
        print("cached")
    else:
        main(variant, seed, p)
