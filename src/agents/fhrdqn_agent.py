"""Fast Hankel-Rank-Regularized DQN (FHR-DQN).

Replaces HR-DQN's truncated-nuclear-norm penalty (per-step SVDs on small Hankel
matrices) with the residual of a learned linear recurrence on Q-values along
replayed trajectories. Kronecker's theorem — a scalar sequence has Hankel rank
<= r iff it satisfies an order-r linear recurrence — makes the recurrence
residual an SVD-free surrogate for the same low-rank prior, and one shared
coefficient vector across all episodes corresponds to low rank of the stacked
(mosaic) Hankel of all trajectories.
"""
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from .q_agent import QAgent
from .hankel_dqn_agent import EpisodicReplayBuffer


class FHRDQNAgent(QAgent):
    """FHR-DQN: classical (Double-)DQN plus a learned-recurrence penalty on the
    Q-values of the TD batch itself. fhr_weight=0 reproduces QAgent training
    exactly (same TD computation, same sampling distribution, no penalty
    machinery touched).

    The penalty rides the TD batch: the B transitions are sampled with their
    (episode, t) handles; every sample with t >= r contributes a residual
    against the AR prediction from its r same-episode predecessors, evaluated
    with the online network on both sides:

        rho_b = Q(s_t, a_t) - sum_j c_j Q(s_{t-j}, a_{t-j})
                            - [reward_lags] sum_k d_k r_{t-k}
        L = L_TD + lambda * mean_b Huber(rho_b)      (lambda = 0 for the first
                                                      warmup_grad_steps)

    c (and d) are learned by the same optimiser as theta, but in their own
    AdamW param group with weight_decay 0 and their own learning rate; they are
    excluded from the target network and from theta's gradient clipping.

    Extra kwargs over QAgent (all mappable to config agent: keys):
        fhr_weight: lambda on the recurrence residual (0 = plain Double-DQN).
        fhr_order: r, the number of Q-value lags (the target Hankel rank).
        reward_lags: also learn d in R^r on the r preceding rewards (ARX). The
            exact on-trajectory Bellman recurrence Q_t = (Q_{t-1} - r_{t-1})/gamma
            is then representable already at order 1 (c_1 = 1/gamma,
            d_1 = -1/gamma, the init used). Without reward lags the init is
            c = (1 + 1/gamma, -1/gamma, 0, ...), which exactly annihilates
            Bellman-consistent sequences under constant per-step rewards but
            needs r >= 2 to do so.
        warmup_grad_steps: hard warm-up K0 — lambda is exactly 0 for the first
            K0 gradient steps, then fhr_weight at full strength (no ramp).
        c_learning_rate: learning rate of the c/d param group.
    """

    def __init__(self, *, fhr_weight: float = 0.0, fhr_order: int = 2,
                 reward_lags: bool = False, warmup_grad_steps: int = 2000,
                 c_learning_rate: float = 5e-3, **q_agent_kwargs):
        super().__init__(**q_agent_kwargs)
        if fhr_order < 1:
            raise ValueError(f"fhr_order must be >= 1, got {fhr_order}")
        if fhr_order == 1 and not reward_lags:
            warnings.warn(
                "Pure AR with fhr_order=1 cannot be satisfied by a "
                "Bellman-consistent Q under constant per-step rewards (residual "
                "floor -reward/gamma at the fixed point); use fhr_order >= 2 or "
                "reward_lags=True.", stacklevel=2)
        self.replay_buffer = EpisodicReplayBuffer(q_agent_kwargs["replay_buffer_capacity"])
        self.fhr_weight = fhr_weight
        self.fhr_order = fhr_order
        self.reward_lags = reward_lags
        self.warmup_grad_steps = warmup_grad_steps

        gamma = self.loss.gamma
        c0 = torch.zeros(fhr_order)
        if reward_lags:
            c0[0] = 1.0 / gamma
            d0 = torch.zeros(fhr_order)
            d0[0] = -1.0 / gamma
        elif fhr_order >= 2:
            c0[0], c0[1] = 1.0 + 1.0 / gamma, -1.0 / gamma
        else:
            c0[0] = 1.0 / gamma
        self.c = torch.nn.Parameter(c0.to(self.device))
        coeffs = [self.c]
        self.d = None
        if reward_lags:
            self.d = torch.nn.Parameter(d0.to(self.device))
            coeffs.append(self.d)
        # Own param group: no weight decay (AdamW's default 0.01 would drag the
        # coefficients toward 0, morphing the penalty into shrink-Q-to-zero)
        # and an independent lr. Excluded from clipping (clip targets
        # policy_net.parameters()) and from the target sync (agent-level
        # parameter, not part of the Q module).
        self.optimiser.add_param_group(
            {"params": coeffs, "lr": c_learning_rate, "weight_decay": 0.0})

        self._grad_steps = 0
        self.nan_skips = 0

    # -- buffer plumbing (episodic; mirrors HankelDQNAgent) -----------------
    def update_buffer(self, state, action, reward, next_state, terminated, truncated=False):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = torch.tensor([action], dtype=torch.long, device=self.device)
        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        next_state = (None if terminated else
                      torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0))
        self.replay_buffer.append(state, action, reward, next_state)
        if terminated or truncated:
            self.replay_buffer.close(terminated)

    def update_buffer_atari(self, state, action, reward, next_state, terminated, truncated=False):
        raise NotImplementedError(
            "FHRDQNAgent's episodic buffer has no atari (uint8/CPU) path yet")

    # -- penalty schedule: hard warm-up, no ramp ----------------------------
    def _lambda_eff(self) -> float:
        return 0.0 if self._grad_steps < self.warmup_grad_steps else self.fhr_weight

    def _companion_radius(self) -> float:
        """Spectral radius of the companion matrix of z^r - c_1 z^{r-1} - ... - c_r.
        The exact constant-reward Bellman recurrence has roots {1, 1/gamma}, so
        ~1/gamma is the healthy reference value."""
        c = self.c.detach().cpu().numpy()
        roots = np.roots(np.concatenate(([1.0], -c)))
        return float(np.abs(roots).max()) if roots.size else 0.0

    def _train_step(self):
        states, actions, rewards, next_list, handles = self.replay_buffer.sample_transitions(
            self.batch_size, with_handles=True)
        non_final_mask = torch.tensor([s is not None for s in next_list], device=self.device)

        Q_s_a = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        Q_s_1_a = torch.zeros(len(next_list), device=self.device)
        if non_final_mask.any():
            next_state = torch.cat([s for s in next_list if s is not None]).to(self.device)
            with torch.no_grad():
                if self.double:
                    next_actions = self.policy_net(next_state).argmax(dim=1, keepdim=True)
                    Q_s_1_a[non_final_mask] = self.target_net(next_state).gather(1, next_actions).squeeze(1)
                else:
                    Q_s_1_a[non_final_mask] = self.target_net(next_state).max(dim=1).values
        loss = self.loss(Q_s_a, rewards, Q_s_1_a)

        lam = self._lambda_eff()
        diag = {"td_loss": float(loss.detach()), "lambda_eff": lam,
                "penalty_raw": np.nan, "penalty_weighted": 0.0,
                "residual_rms": np.nan, "b_h": np.nan, "unique_eps": np.nan,
                "sum_c": float(self.c.detach().sum()),
                "companion_radius": self._companion_radius(),
                "nan_skips": self.nan_skips}
        for j in range(self.fhr_order):
            diag[f"c_{j + 1}"] = float(self.c[j].detach())
            if self.reward_lags:
                diag[f"d_{j + 1}"] = float(self.d[j].detach())

        if self.fhr_weight > 0:
            r = self.fhr_order
            keep = [i for i, (_, t) in enumerate(handles) if t >= r]
            diag["b_h"] = float(len(keep))
            diag["unique_eps"] = float(len({handles[i][0] for i in keep}))
            if keep:
                p_states, p_actions, p_rewards = self.replay_buffer.gather_predecessors(
                    [handles[i] for i in keep], r)
                n = len(keep)
                # lambda = 0 (warm-up): diagnostics only, keep the penalty out
                # of the graph.
                with torch.enable_grad() if lam > 0 else torch.no_grad():
                    # Fused evaluation: one batched forward over all n*r
                    # predecessor states; the anchor Q(s_t, a_t) is shared with
                    # the TD term above. No per-sample loops, no SVD.
                    out = self.policy_net(p_states.reshape(n * r, -1))
                    q_lags = out.gather(1, p_actions.reshape(n * r, 1)).view(n, r)
                    prediction = q_lags @ self.c
                    if self.reward_lags:
                        prediction = prediction + p_rewards @ self.d
                    anchor = Q_s_a[keep]
                    penalty = F.huber_loss(anchor, prediction)
                diag["penalty_raw"] = float(penalty.detach())
                diag["penalty_weighted"] = lam * diag["penalty_raw"]
                diag["residual_rms"] = float(
                    (anchor.detach() - prediction.detach()).pow(2).mean().sqrt())
                if lam > 0:
                    loss = loss + lam * penalty

        self._grad_steps += 1
        if not torch.isfinite(loss):
            self.nan_skips += 1
            diag["nan_skips"] = self.nan_skips
            return diag
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm)
        if not torch.isfinite(grad_norm):
            self.nan_skips += 1
            diag["nan_skips"] = self.nan_skips
            self.optimiser.zero_grad()
            return diag
        self.optimiser.step()
        return diag

    def train(self):
        gd_steps = min(len(self.replay_buffer) // (self.buffer_util * self.batch_size),
                       self.gd_steps_ceil)
        diags = [self._train_step() for _ in range(gd_steps)]
        if not diags:
            return None
        def _nanmean(vals):
            finite = [v for v in vals if not np.isnan(v)]
            return float(np.mean(finite)) if finite else float("nan")
        return {k: _nanmean([d[k] for d in diags]) for k in diags[0]}

    # -- persistence: also carry the recurrence coefficients ----------------
    def save(self, path):
        payload = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            "epsilon": self.epsilon,
            "fhr_c": self.c.detach().cpu(),
        }
        if self.d is not None:
            payload["fhr_d"] = self.d.detach().cpu()
        torch.save(payload, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimiser.load_state_dict(ckpt["optimiser"])
        self.epsilon = ckpt["epsilon"]
        with torch.no_grad():
            self.c.copy_(ckpt["fhr_c"].to(self.device))
            if self.d is not None and "fhr_d" in ckpt:
                self.d.copy_(ckpt["fhr_d"].to(self.device))
