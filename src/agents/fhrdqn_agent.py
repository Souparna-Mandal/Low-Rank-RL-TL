"""Fast Hankel-Rank-Regularized DQN (FHR-DQN).

Replaces HR-DQN's truncated-nuclear-norm penalty (per-step SVDs on small Hankel
matrices) with the residual of a learned linear recurrence on Q-values along
replayed trajectories. Kronecker's theorem — a scalar sequence has Hankel rank
<= r iff it satisfies an order-r linear recurrence — makes the recurrence
residual an SVD-free surrogate for the same low-rank prior, and one shared
coefficient vector across all episodes corresponds to low rank of the stacked
(mosaic) Hankel of all trajectories.
"""
import re
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
        rampdown_reward_threshold: arming this (non-None) enables the automatic
            lambda ramp-DOWN. After warm-up, once the mean episode reward over
            the last rampdown_patience_eps episodes reaches this value while
            the recurrence residual stayed high (below), the penalty is judged
            to be fighting an already-good policy and lambda anneals to 0.
            Requires the training loop to call notify_episode_end() (the
            dqn_training_loop does). One-way: lambda never comes back up.
        rampdown_penalty_threshold: the "residual stayed high" bar — every
            episode in the trigger window must have a mean penalty_raw >= this.
            A number is an absolute bar; a percentage string like "40%" makes
            the bar relative: that fraction of the mean of the
            rampdown_penalty_topk largest per-episode residuals seen so far
            (post-warm-up), so "high" tracks the run's own residual scale. The
            relative bar is undefined (condition fails) until topk episodes
            have been observed. None (with the reward threshold armed) drops
            the residual condition and triggers on reward alone.
        rampdown_penalty_topk: how many of the largest per-episode residuals
            the "NN%" relative bar averages over.
        rampdown_patience_eps: both conditions are evaluated over a window of
            this many consecutive episodes.
        rampdown_episodes: once triggered, lambda scales linearly from
            fhr_weight to 0 over this many episodes; 0 switches it off
            immediately.
    """

    def __init__(self, *, fhr_weight: float = 0.0, fhr_order: int = 2,
                 reward_lags: bool = False, warmup_grad_steps: int = 2000,
                 c_learning_rate: float = 5e-3,
                 rampdown_reward_threshold: float | None = None,
                 rampdown_penalty_threshold: float | str | None = None,
                 rampdown_penalty_topk: int = 20,
                 rampdown_patience_eps: int = 10,
                 rampdown_episodes: int = 0, **q_agent_kwargs):
        super().__init__(**q_agent_kwargs)
        if fhr_order < 1:
            raise ValueError(f"fhr_order must be >= 1, got {fhr_order}")
        if rampdown_episodes < 0:
            raise ValueError(f"rampdown_episodes must be >= 0, got {rampdown_episodes}")
        if rampdown_patience_eps < 1:
            raise ValueError(f"rampdown_patience_eps must be >= 1, got {rampdown_patience_eps}")
        if rampdown_penalty_topk < 1:
            raise ValueError(f"rampdown_penalty_topk must be >= 1, got {rampdown_penalty_topk}")
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

        # -- automatic lambda ramp-down state --
        self.rampdown_reward_threshold = rampdown_reward_threshold
        self.rampdown_penalty_threshold = rampdown_penalty_threshold
        self.rampdown_penalty_topk = rampdown_penalty_topk
        self.rampdown_patience_eps = rampdown_patience_eps
        self.rampdown_episodes = rampdown_episodes
        # absolute bar, or "NN%" of the mean of the topk largest per-episode
        # residuals seen so far — parsed once here
        self._rd_pen_abs = None
        self._rd_pen_frac = None
        if isinstance(rampdown_penalty_threshold, str):
            m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*%\s*",
                             rampdown_penalty_threshold)
            if not m:
                raise ValueError(
                    "rampdown_penalty_threshold must be a number or a "
                    f"percentage string like '40%', got {rampdown_penalty_threshold!r}")
            self._rd_pen_frac = float(m.group(1)) / 100.0
        elif rampdown_penalty_threshold is not None:
            self._rd_pen_abs = float(rampdown_penalty_threshold)
        self._ep_penalty_vals: list[float] = []   # penalty_raw within the episode
        self._rd_window: list[tuple[float, float]] = []  # (reward, ep mean penalty)
        self._pen_top: list[float] = []  # largest post-warm-up per-episode residuals
        self._rd_k = None            # None = not triggered; else episodes since

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

    # -- penalty schedule: hard warm-up, optional triggered ramp-down -------
    def _rampdown_scale(self) -> float:
        """Multiplier on fhr_weight: 1 until the ramp-down triggers, then
        linear to 0 over rampdown_episodes episodes (0 => immediately 0)."""
        if self._rd_k is None:
            return 1.0
        if self.rampdown_episodes <= 0:
            return 0.0
        return max(0.0, 1.0 - self._rd_k / self.rampdown_episodes)

    def _lambda_eff(self) -> float:
        if self._grad_steps < self.warmup_grad_steps:
            return 0.0
        return self.fhr_weight * self._rampdown_scale()

    def _penalty_bar(self) -> float | None:
        """The residual bar an episode must reach to count as "high": the
        absolute threshold, or the "NN%" fraction of the mean of the topk
        largest per-episode residuals seen so far. None while unset — or, in
        relative mode, while fewer than topk episodes have been observed (not
        enough history to call anything relatively high)."""
        if self._rd_pen_abs is not None:
            return self._rd_pen_abs
        if (self._rd_pen_frac is not None
                and len(self._pen_top) >= self.rampdown_penalty_topk):
            return self._rd_pen_frac * float(np.mean(self._pen_top))
        return None

    def notify_episode_end(self, episode: int, episode_reward: float) -> None:
        """Per-episode hook (called by dqn_training_loop): watch the reward
        window and the recurrence residual to decide the lambda ramp-down.

        Trigger = the last rampdown_patience_eps episodes have mean reward >=
        rampdown_reward_threshold AND (when a penalty threshold is set) every
        one of them kept a mean penalty_raw >= rampdown_penalty_threshold —
        i.e. the policy is already good, yet the recurrence keeps mispredicting
        it, so the penalty now only distorts the TD objective. One-way."""
        ep_pen = (float(np.mean(self._ep_penalty_vals))
                  if self._ep_penalty_vals else float("nan"))
        self._ep_penalty_vals = []
        if self._rd_k is not None:          # already ramping — just advance it
            self._rd_k += 1
            return
        if self.rampdown_reward_threshold is None or self.fhr_weight <= 0:
            return              # disabled, or the baseline arm: no lambda to ramp
        if self._grad_steps < self.warmup_grad_steps:
            # warm-up episodes never count toward the trigger window — the
            # penalty is not training yet, so its residual is trivially high
            # and a good-enough policy would fire the trigger on the very
            # first post-warm-up hook, skipping the penalty phase entirely
            return
        if np.isfinite(ep_pen):
            # historical high-residual reference for the "NN%" relative bar
            self._pen_top.append(ep_pen)
            self._pen_top.sort(reverse=True)
            del self._pen_top[self.rampdown_penalty_topk:]
        self._rd_window.append((float(episode_reward), ep_pen))
        del self._rd_window[:-self.rampdown_patience_eps]
        if len(self._rd_window) < self.rampdown_patience_eps:
            return
        rewards = [r for r, _ in self._rd_window]
        pens = [p for _, p in self._rd_window]
        if float(np.mean(rewards)) < self.rampdown_reward_threshold:
            return
        bar = self._penalty_bar()
        if self.rampdown_penalty_threshold is not None and (
                bar is None or not all(np.isfinite(p) and p >= bar
                                       for p in pens)):
            return
        self._rd_k = 1
        gate = ("not gated" if self.rampdown_penalty_threshold is None
                else f"always >= {bar:g}"
                + (f" ({self.rampdown_penalty_threshold} of the top-"
                   f"{self.rampdown_penalty_topk} residual mean)"
                   if self._rd_pen_frac is not None else ""))
        print(f"FHR lambda ramp-down triggered at episode {episode}: "
              f"mean reward {np.mean(rewards):.1f} >= "
              f"{self.rampdown_reward_threshold} over "
              f"{self.rampdown_patience_eps} eps, penalty_raw {gate}"
              f" — lambda -> 0 over {self.rampdown_episodes} episode(s)")

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
                "rampdown_scale": self._rampdown_scale(),
                "rampdown_penalty_bar": (float("nan") if (bar := self._penalty_bar()) is None
                                         else bar),
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
                # feed the ramp-down trigger's per-episode residual average —
                # only while the penalty actually trains (lam > 0): warm-up
                # residuals reflect the frozen init and must not count
                if lam > 0:
                    self._ep_penalty_vals.append(diag["penalty_raw"])
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
