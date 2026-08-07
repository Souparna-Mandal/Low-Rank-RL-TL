"""SSM critic with AR-order-adaptive hidden length (the 'AR(2) trick').

Warm up at rank 16; after warmup_updates, fit AR(k) for k=1..8 to the
rollout's value sequences (held-out residual per order), pick the smallest
order within tol of the best, and permanently MASK the hidden state down to
2*order channels (those with the largest |C| readout weights). The prior's
strength is thereby set by the environment's measured recurrence order
instead of a hand-picked rank."""
import numpy as np
import torch

from agents.variants.ssm_critic import (PPO_KEYS, SSMCriticAgent,
                                        collect_rollout)  # noqa: F401 (reused)


def _ar_residual(vals_segs, k):
    """Fit AR(k) on even-indexed segments, return residual on odd ones."""
    def rows(segs):
        X, y = [], []
        for v in segs:
            for t in range(k, len(v)):
                X.append(v[t - k:t][::-1])
                y.append(v[t])
        return np.asarray(X), np.asarray(y)

    fit = vals_segs[::2] or vals_segs
    hold = vals_segs[1::2] or vals_segs
    X, y = rows(fit)
    if len(y) < 4 * k:
        return np.inf, None
    c = np.linalg.solve(X.T @ X + 1e-8 * np.eye(k), X.T @ y)
    Xh, yh = rows(hold)
    # Near-constant value signal: order is unmeasurable (scores would be
    # ratios of numerical noise) — report failure instead of an arbitrary k.
    if len(yh) == 0 or np.std(yh) < 1e-6:
        return np.inf, None
    res = yh - Xh @ c
    return float(np.sqrt(np.mean(res ** 2)) / np.std(yh)), c


class SSMAutoAgent(SSMCriticAgent):
    def __init__(self, env, device="cpu", warmup_updates=6, order_tol=1.05,
                 max_order=8, **kwargs):
        super().__init__(env, device=device, ssm_rank=16, **kwargs)
        self.warmup_updates = warmup_updates
        self.order_tol = order_tol
        self.max_order = max_order
        self._updates = 0
        self.mask = torch.ones(16, device=self.device)
        self.diag = {"order": None, "rank": 16}

    def _choose_order(self, buf):
        segs = [buf["values"][a:b] for a, b in buf["seg_bounds"] if b - a >= 12]
        if len(segs) < 2:
            return None
        scores = {}
        for k in range(1, self.max_order + 1):
            r, _ = _ar_residual(segs, k)
            scores[k] = r
        best = min(scores.values())
        if not np.isfinite(best):
            return None
        return min(k for k, r in scores.items() if r <= self.order_tol * best)

    def _apply_mask(self, order):
        rank = min(16, max(2, 2 * order))
        keep = torch.argsort(
            self.critic.C.weight.detach().abs().squeeze(0), descending=True)[:rank]
        m = torch.zeros(16, device=self.device)
        m[keep] = 1.0
        self.mask = m
        if self.h is not None:
            self.h = self.h * m
        self.diag = {"order": int(order), "rank": int(self.mask.sum().item())}

    # The mask must be ENFORCED in every forward pass, not baked into the
    # weights once: h_dead is forced to exactly 0 before the readout, so
    # dv/dC_dead = h_dead = 0 and dead channels can never revive through
    # gradients or optimizer momentum.
    def _masked_step(self, phi, h):
        c = self.critic
        h = (torch.sigmoid(c.decay_logits) * h + c.B(phi)) * self.mask
        return h, (c.C(h) + c.D(phi)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs):
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        dist = self._dist(t.unsqueeze(0))
        a = dist.sample()
        logp = float(self._logp(dist, a))
        self.h, v = self._masked_step(self.critic.trunk(t), self.h)
        if self.continuous:
            return a.squeeze(0).cpu().numpy(), logp, float(v)
        return int(a), logp, float(v)

    @torch.no_grad()
    def act_and_value_only(self, obs):
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        _, v = self._masked_step(self.critic.trunk(t), self.h)  # peek only
        return float(v)

    def _seq_values(self, obs, seg_bounds, seg_h0):
        phi = self.critic.trunk(obs)
        bx = self.critic.B(phi)
        a = torch.sigmoid(self.critic.decay_logits)
        hs = []
        for (s, e), h in zip(seg_bounds, seg_h0):
            h = h * self.mask  # pre-mask hidden states enter masked
            for t in range(s, e):
                h = (a * h + bx[t]) * self.mask
                hs.append(h)
        return (self.critic.C(torch.stack(hs)) + self.critic.D(phi)).squeeze(-1)

    def update(self, buf):
        if self._updates == self.warmup_updates:
            order = self._choose_order(buf)
            if order is not None:
                self._apply_mask(order)
        self._updates += 1
        super().update(buf)


def build(env, device, overrides):
    kwargs = {k: v for k, v in overrides.items() if k in PPO_KEYS}
    return SSMAutoAgent(env=env, device=device,
                        warmup_updates=int(overrides.get("warmup_updates", 6)),
                        order_tol=float(overrides.get("order_tol", 1.05)),
                        max_order=int(overrides.get("max_order", 8)),
                        **kwargs)
