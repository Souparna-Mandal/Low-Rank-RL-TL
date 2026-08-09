"""Classical tabular Policy Iteration via generative-model Monte-Carlo evaluation.

 policy
evaluation estimates Q^pi(s, a) for EVERY (bin-centre, action) by teleporting
the env to s, forcing a, then following the greedy policy for up to `horizon`
steps and accumulating the discounted return. Policy improvement is the greedy
argmax over the table; PI has converged when improvement changes no action.

`evaluate_policy` and `_mc_return` are the seams a low-rank variant overrides
(sample fewer entries / truncate rollouts and complete the Q matrix).

The env must be built with `discrete_config.state_bins` (DiscreteStateWrapper:
bin geometry + discretise) and `generative: true` (GenerativeStateWrapper:
teleport). The `policy` array IS pi in tabular form: policy[i] = action taken
in bin i. It is separate from Q so the policy stays frozen while evaluation
overwrites the table.
"""
import numpy as np
import torch
from tqdm import tqdm

from .base_agent import BaseAgent


class _TabularQNet:
    """policy_net stand-in: (B, obs_dim) state tensor -> (B, n_actions) Q-table
    rows, so the Hankel/rank analysis probes a tabular agent unchanged."""

    def __init__(self, agent):
        self._agent = agent

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        states = np.atleast_2d(x.detach().cpu().numpy())
        idxs = [self._agent.discretise(s) for s in states]
        return torch.as_tensor(self._agent.Q[idxs], dtype=torch.float32,
                               device=self._agent.device)


class TabularPolicyIterationAgent(BaseAgent):
    def __init__(self, env, discount_factor: float = 0.99, horizon: int = 500,
                 n_eval_rollouts: int = 1, record_rollouts: bool = False,
                 device: str = "cpu"):
        super().__init__(env)
        self.gamma = discount_factor
        self.horizon = horizon
        self.n_eval_rollouts = n_eval_rollouts
        # When set, evaluate_policy keeps every generative rollout (per-step reward
        # + visited bin) in self.mc_rollouts so a low-rank/truncation study can
        # recompute Q at any horizon offline. The loop persists it per iteration.
        self.record_rollouts = record_rollouts
        self.mc_rollouts = None
        # Analysis code places probe tensors on agent.device; the table itself
        # is numpy — per-step lookups in the rollout loop must stay CPU-cheap.
        self.device = device

        self.discretise = env.get_wrapper_attr("discretise")
        self.state_bins = env.get_wrapper_attr("state_bins")
        self.bin_centres = env.get_wrapper_attr("bin_centres")
        self.n_states = env.get_wrapper_attr("n_states")
        self.n_actions = int(env.action_space.n)

        self.Q = np.zeros((self.n_states, self.n_actions))
        self.policy = np.zeros(self.n_states, dtype=np.int64)
        self.policy_net = _TabularQNet(self)

    # ------------------------------------------------------------------ acting
    def pi(self, state) -> int:
        return int(self.policy[self.discretise(state)])

    def act_greedy(self, state: torch.Tensor) -> int:
        """Batched (1, obs_dim) tensor variant, for the rollout-video helper."""
        return self.pi(state.detach().cpu().numpy().reshape(-1))

    # -------------------------------------------------------------- evaluation
    def _mc_return(self, env, state, action: int, horizon: int = None,
                   record: dict = None) -> float:
        """One generative rollout: teleport to `state`, force `action`, then
        follow pi greedily. Override seam: pass a short `horizon` to truncate.

        Pass a `record` dict {"rewards": [], "bins": []} to log the per-step
        reward and the visited (next-state) bin; on return it also carries
        record["terminated"] (True = absorbing/goal, so the tail value is 0)."""
        horizon = self.horizon if horizon is None else horizon
        obs = env.get_wrapper_attr("teleport")(state)
        obs, reward, terminated, truncated, _ = env.step(action)
        G, disc = float(reward), self.gamma
        if record is not None:
            record["rewards"].append(float(reward)); record["bins"].append(self.discretise(obs))
        for _ in range(horizon - 1):
            if terminated or truncated:
                break
            obs, reward, terminated, truncated, _ = env.step(self.pi(obs))
            G += disc * float(reward)
            disc *= self.gamma
            if record is not None:
                record["rewards"].append(float(reward)); record["bins"].append(self.discretise(obs))
        if record is not None:
            record["terminated"] = bool(terminated)
        return G

    def evaluate_policy(self, env) -> np.ndarray:
        """Exhaustive MC evaluation of the current policy: fills Q^pi for every
        (bin centre, action). Override seam for low-rank/truncated variants."""
        rec = self.record_rollouts
        cols = {k: [] for k in ("start_bin", "action", "return", "length",
                                "terminated", "rewards", "bins")} if rec else None
        for s_idx in tqdm(range(self.n_states), desc="policy evaluation", leave=False):
            centre = self.bin_centres[s_idx]
            for a in range(self.n_actions):
                returns = []
                for _ in range(self.n_eval_rollouts):
                    if rec:
                        steps = {"rewards": [], "bins": []}
                        g = self._mc_return(env, centre, a, record=steps)
                        cols["start_bin"].append(s_idx); cols["action"].append(a)
                        cols["return"].append(g); cols["length"].append(len(steps["rewards"]))
                        cols["terminated"].append(steps["terminated"])
                        cols["rewards"].append(np.asarray(steps["rewards"], dtype=np.float32))
                        cols["bins"].append(np.asarray(steps["bins"], dtype=np.int32))
                    else:
                        g = self._mc_return(env, centre, a)
                    returns.append(g)
                self.Q[s_idx, a] = np.mean(returns)
        if rec:
            self.mc_rollouts = self._pack_rollouts(cols)
        return self.Q

    def _pack_rollouts(self, cols: dict) -> dict:
        """Flatten the per-rollout records into a compact, self-describing set of
        arrays: a (start_bin, action, return, length, terminated) row per rollout,
        plus rewards/bins concatenated across rollouts (slice with the cumulative
        lengths). Bootstrap a truncated Q offline via rewards[:tau] + gamma**tau *
        V(bins[tau-1])."""
        empty_f = np.empty(0, dtype=np.float32); empty_i = np.empty(0, dtype=np.int32)
        return {
            "start_bin": np.asarray(cols["start_bin"], dtype=np.int32),
            "action": np.asarray(cols["action"], dtype=np.int32),
            "return": np.asarray(cols["return"], dtype=np.float32),
            "length": np.asarray(cols["length"], dtype=np.int32),
            "terminated": np.asarray(cols["terminated"], dtype=bool),
            "rewards": np.concatenate(cols["rewards"]) if cols["rewards"] else empty_f,
            "bins": np.concatenate(cols["bins"]) if cols["bins"] else empty_i,
            "state_bins": self.state_bins.astype(np.int32),
            "gamma": np.float64(self.gamma),
            "n_actions": np.int64(self.n_actions),
        }

    # ------------------------------------------------------------- improvement
    def improve_policy(self) -> int:
        """Greedy improvement; returns how many bins changed action (0 = converged)."""
        new_policy = self.Q.argmax(axis=1)
        n_changed = int((new_policy != self.policy).sum())
        self.policy = new_policy
        return n_changed

    # ------------------------------------------------------------- persistence
    def save(self, path):
        torch.save({
            "Q": torch.as_tensor(self.Q),
            "policy": torch.as_tensor(self.policy),
            "state_bins": self.state_bins.tolist(),
            "discount_factor": self.gamma,
            "horizon": self.horizon,
            "n_eval_rollouts": self.n_eval_rollouts,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu")
        assert ckpt["state_bins"] == self.state_bins.tolist(), \
            "checkpoint grid geometry does not match this env's discretisation"
        self.Q = ckpt["Q"].numpy().astype(np.float64)
        self.policy = ckpt["policy"].numpy().astype(np.int64)
