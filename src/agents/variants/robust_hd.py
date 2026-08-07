"""Robust Hankel-denoised critic targets: per-segment RPCA-lite splits the
lambda-return sequence into a low-rank trend (Cadzow) plus sparse spikes, so
rare events (crashes/landings) survive denoising instead of being smeared."""
import numpy as np
import torch
import torch.nn as nn

from agents.ppo_agent import PPOAgent

PPO_KEYS = {"hidden_sizes", "nn_learning_rate", "discount_factor", "gae_lambda",
            "rollout_steps", "minibatch_size", "update_epochs", "clip_eps",
            "vf_coef", "ent_coef", "max_grad_norm", "log_std_init"}


def _cadzow_step(y, r, beta=1.0):
    """One Cadzow iteration: Hankel lift -> rank-r SVD truncation ->
    anti-diagonal averaging. Sequences too short to lift pass through."""
    n = len(y)
    if n < max(2 * r + 2, 6):
        return y
    w = n // 2 + 1
    H = np.lib.stride_tricks.sliding_window_view(y, w)
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    Hr = (U[:, :r] * s[:r]) @ Vt[:r]
    out = np.zeros(n)
    cnt = np.zeros(n)
    for i in range(Hr.shape[0]):
        out[i:i + w] += Hr[i]
        cnt[i:i + w] += 1
    return (1 - beta) * y + beta * out / cnt


class RobustHDAgent(PPOAgent):
    def __init__(self, env, device="cpu", hd_rank=2, hd_k=2.0, hd_blend=0.5,
                 **kwargs):
        super().__init__(env, device=device, **kwargs)
        self.hd_rank = hd_rank
        self.hd_k = hd_k
        self.hd_blend = hd_blend
        self.diag = None

    def _robust_targets(self, y):
        """RPCA-lite: y ~= L (low-Hankel-rank trend) + S (sparse spikes)."""
        if len(y) < max(2 * self.hd_rank + 2, 6):
            return y
        S = np.zeros_like(y)
        for _ in range(3):
            L = _cadzow_step(y - S, self.hd_rank, beta=1.0)
            res = y - L
            tau = self.hd_k * res.std()
            S = np.sign(res) * np.maximum(np.abs(res) - tau, 0.0)
        return self.hd_blend * (L + S) + (1 - self.hd_blend) * y

    def update(self, buf):
        obs = torch.as_tensor(buf["obs"], dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(buf["acts"], device=self.device,
                               dtype=torch.float32 if self.continuous else None)
        old_logp = torch.as_tensor(buf["logps"], dtype=torch.float32, device=self.device)

        values = buf["values"]
        T = len(values)
        adv = np.zeros(T, dtype=np.float64)
        for (a, b), terminal, boot_v in zip(buf["seg_bounds"], buf["seg_terminal"],
                                            buf["seg_boot_value"]):
            next_v = 0.0 if terminal else boot_v
            gae = 0.0
            for t in range(b - 1, a - 1, -1):
                delta = buf["rews"][t] + self.gamma * next_v - values[t]
                gae = delta + self.gamma * self.lam * gae
                adv[t] = gae
                next_v = values[t]
        returns = adv + values

        targets = returns.copy()
        for a, b in buf["seg_bounds"]:
            targets[a:b] = self._robust_targets(returns[a:b])
        self.diag = {"hd_mean_shift": float(np.abs(targets - returns).mean())}

        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.as_tensor(targets, dtype=torch.float32, device=self.device)

        idx = np.arange(T)
        for _ in range(self.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, T, self.minibatch_size):
                mb = idx[start:start + self.minibatch_size]
                dist = self._dist(obs[mb])
                logp = self._logp(dist, acts[mb])
                ratio = (logp - old_logp[mb]).exp()
                s1 = ratio * adv_t[mb]
                s2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb]
                policy_loss = -torch.min(s1, s2).mean()
                v = self.critic(obs[mb]).squeeze(-1)
                value_loss = 0.5 * ((v - ret_t[mb]) ** 2).mean()
                loss = (policy_loss + self.vf_coef * value_loss
                        - self.ent_coef * self._entropy(dist).mean())
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._all_params(), self.max_grad_norm)
                self.optim.step()


def build(env, device, overrides):
    kwargs = {k: v for k, v in overrides.items() if k in PPO_KEYS}
    return RobustHDAgent(env=env, device=device,
                         hd_rank=int(overrides.get("hd_rank", 2)),
                         hd_k=float(overrides.get("hd_k", 2.0)),
                         hd_blend=float(overrides.get("hd_blend", 0.5)),
                         **kwargs)
