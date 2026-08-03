import numpy as np
import torch
import torch.nn as nn

from agents.hankel_regulariser import HankelRankPenalty


def _mlp(in_dim, out_dim, hidden):
    layers, last = [], in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.Tanh()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class PPOAgent:
    """PPO-clip for discrete actions with optional Hankel-structure additions.

    hankel_weight > 0 adds the HR-DQN truncated-tail penalty on the critic
    along contiguous within-episode windows of the current rollout (on-policy,
    so no gate against policy drift is needed beyond gate_threshold).
    ar_filter="blend" replaces V with an AR(ar_order)-filtered value in the
    GAE deltas; ar_tail_bootstrap closes rollout-truncated segments with the
    AR forecast instead of V(s_T).
    """

    def __init__(self, env, device="cpu", hidden_sizes=(64, 64),
                 nn_learning_rate=3e-4, discount_factor=0.99, gae_lambda=0.95,
                 rollout_steps=2048, minibatch_size=256, update_epochs=10,
                 clip_eps=0.2, vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5,
                 hankel_weight=0.0, hankel_order=2, window_len=16, n_windows=8,
                 gate_threshold=0.25, ramp_updates=10,
                 engage_reward_threshold=None, engage_reward_window=10,
                 ar_filter=None, ar_order=2, ar_alpha=0.5,
                 ar_tail_bootstrap=False,
                 denoise_targets_rank=None, denoise_beta=0.5):
        obs_dim = env.observation_space.shape[0]
        n_act = env.action_space.n
        self.device = torch.device(device)
        self.actor = _mlp(obs_dim, n_act, hidden_sizes).to(self.device)
        self.critic = _mlp(obs_dim, 1, hidden_sizes).to(self.device)
        self.optim = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=nn_learning_rate)
        self.gamma = discount_factor
        self.lam = gae_lambda
        self.rollout_steps = rollout_steps
        self.minibatch_size = minibatch_size
        self.update_epochs = update_epochs
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.hankel_weight = hankel_weight
        self.window_len = window_len
        self.n_windows = n_windows
        self.penalty = (HankelRankPenalty(order=hankel_order,
                                          gate_threshold=gate_threshold)
                        if hankel_weight > 0 else None)
        self.ramp_updates = ramp_updates
        self.engage_reward_threshold = engage_reward_threshold
        self.engage_reward_window = engage_reward_window
        self._recent_returns = []
        self._engaged_at = None
        self._updates = 0

        self.ar_filter = ar_filter
        self.ar_order = ar_order
        self.ar_alpha = ar_alpha
        self.ar_tail_bootstrap = ar_tail_bootstrap
        self.denoise_targets_rank = denoise_targets_rank
        self.denoise_beta = denoise_beta
        self.diag = {}

    # -- acting ------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs):
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.actor(t)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return int(a), float(dist.log_prob(a)), float(self.critic(t).squeeze())

    @torch.no_grad()
    def act_and_value_only(self, obs):
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.critic(t).squeeze())

    @torch.no_grad()
    def act_greedy(self, obs):
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.actor(t).argmax(dim=1))

    def record_episode_return(self, ret):
        self._recent_returns.append(float(ret))
        if len(self._recent_returns) > self.engage_reward_window:
            self._recent_returns.pop(0)

    def _lambda_eff(self):
        if self.hankel_weight == 0:
            return 0.0
        if self.engage_reward_threshold is not None:
            if self._engaged_at is None:
                full = len(self._recent_returns) >= self.engage_reward_window
                if full and np.mean(self._recent_returns) >= self.engage_reward_threshold:
                    self._engaged_at = self._updates
                else:
                    return 0.0
            k = self._updates - self._engaged_at
        else:
            k = self._updates
        return self.hankel_weight * min(1.0, (k + 1) / max(self.ramp_updates, 1))

    # -- AR helpers --------------------------------------------------------
    def _fit_ar(self, segments):
        r = self.ar_order
        X, y = [], []
        for seg in segments:
            v = seg
            for t in range(r, len(v)):
                X.append(v[t - r:t][::-1])
                y.append(v[t])
        if len(y) < 4 * r:
            return None
        X, y = np.asarray(X), np.asarray(y)
        A = X.T @ X + 1e-8 * np.eye(r)
        try:
            return np.linalg.solve(A, X.T @ y)
        except np.linalg.LinAlgError:
            return None

    def _filter_values(self, values, seg_bounds):
        """Blend V with its one-step AR prediction inside each segment."""
        coeffs = self._fit_ar([values[a:b] for a, b in seg_bounds])
        if coeffs is None:
            return values, None
        r, alpha = self.ar_order, self.ar_alpha
        out = values.copy()
        for a, b in seg_bounds:
            v = values[a:b]
            if len(v) <= r:
                continue
            pred = np.array([coeffs @ v[t - r:t][::-1] for t in range(r, len(v))])
            out[a + r:b] = (1 - alpha) * v[r:] + alpha * pred
        return out, coeffs

    # -- update ------------------------------------------------------------
    def update(self, buf):
        """buf: dict of np arrays obs (T,d), acts, logps, rews, values (T,),
        dones (T,), plus next_value (float) and seg_bounds/ep_windows metadata
        built by the training loop."""
        obs = torch.as_tensor(buf["obs"], dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(buf["acts"], device=self.device)
        old_logp = torch.as_tensor(buf["logps"], dtype=torch.float32, device=self.device)

        values = buf["values"]
        seg_bounds = buf["seg_bounds"]
        base_vals, coeffs = (self._filter_values(values, seg_bounds)
                             if self.ar_filter == "blend" else (values, None))

        def gae_pass(vals, ar_tail):
            out = np.zeros(len(vals), dtype=np.float64)
            for (a, b), terminal, boot_v in zip(seg_bounds, buf["seg_terminal"],
                                                buf["seg_boot_value"]):
                if terminal:
                    next_v = 0.0
                elif ar_tail and coeffs is not None and b - a > self.ar_order:
                    # forecast from RAW lags: coeffs were fit on raw values
                    next_v = float(coeffs @ values[b - self.ar_order:b][::-1])
                else:
                    next_v = boot_v
                gae = 0.0
                for t in range(b - 1, a - 1, -1):
                    delta = buf["rews"][t] + self.gamma * next_v - vals[t]
                    gae = delta + self.gamma * self.lam * gae
                    out[t] = gae
                    next_v = vals[t]
            return out

        # Actor advantages come from the filtered baseline; the critic always
        # regresses on the raw lambda-return (the filter shapes the baseline,
        # not what the critic learns).
        T = len(values)
        adv = gae_pass(base_vals, self.ar_tail_bootstrap)
        returns = gae_pass(values, False) + values
        if self.denoise_targets_rank:
            for a0, b0 in seg_bounds:
                returns[a0:b0] = cadzow_denoise(returns[a0:b0],
                                                self.denoise_targets_rank,
                                                self.denoise_beta)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        lam_eff = self._lambda_eff()
        win_idx = buf["ep_windows"]
        pen_raw = float("nan")
        idx = np.arange(T)
        for _ in range(self.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, T, self.minibatch_size):
                mb = idx[start:start + self.minibatch_size]
                logits = self.actor(obs[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(acts[mb])
                ratio = (logp - old_logp[mb]).exp()
                s1 = ratio * adv_t[mb]
                s2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb]
                policy_loss = -torch.min(s1, s2).mean()
                v = self.critic(obs[mb]).squeeze(-1)
                value_loss = 0.5 * ((v - ret_t[mb]) ** 2).mean()
                loss = (policy_loss + self.vf_coef * value_loss
                        - self.ent_coef * dist.entropy().mean())
                if lam_eff > 0 and len(win_idx) > 0:
                    take = win_idx[np.random.choice(len(win_idx),
                                                    min(self.n_windows, len(win_idx)),
                                                    replace=False)]
                    v_windows = self.critic(obs[take.reshape(-1)]).reshape(
                        take.shape[0], take.shape[1])
                    pen, pdiag = self.penalty(v_windows)
                    loss = loss + lam_eff * pen
                    pen_raw = pdiag["penalty_raw"]
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm)
                self.optim.step()
        self._updates += 1
        self.diag = {"lambda_eff": lam_eff, "penalty_raw": pen_raw,
                     "ar_c": coeffs.tolist() if coeffs is not None else None}


def cadzow_denoise(seq, rank, beta):
    """One Cadzow step toward the rank-r Hankel manifold: Hankel lift, SVD
    truncation, anti-diagonal averaging; returns (1-beta)*seq + beta*proj."""
    n = len(seq)
    if n < max(2 * rank + 2, 6):
        return seq
    L = n // 2 + 1
    H = np.lib.stride_tricks.sliding_window_view(seq, L)  # (n-L+1, L)
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    Hr = (U[:, :rank] * s[:rank]) @ Vt[:rank]
    proj = np.zeros(n)
    counts = np.zeros(n)
    for i in range(Hr.shape[0]):
        proj[i:i + L] += Hr[i]
        counts[i:i + L] += 1
    proj /= counts
    return (1 - beta) * seq + beta * proj
