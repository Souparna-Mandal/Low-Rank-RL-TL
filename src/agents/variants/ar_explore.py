"""PPO with an AR(2) recurrence-violation exploration bonus on the rewards."""
import numpy as np

from agents.ppo_agent import PPOAgent

PPO_KEYS = {"hidden_sizes", "nn_learning_rate", "discount_factor", "gae_lambda",
            "rollout_steps", "minibatch_size", "update_epochs", "clip_eps",
            "vf_coef", "ent_coef", "max_grad_norm", "log_std_init"}


class ARExplorePPO(PPOAgent):
    """Fits a global AR(2) to the rollout's value trace and pays an intrinsic
    bonus where V(s_t) positively surprises the recurrence, annealed linearly
    to zero over anneal_updates agent updates."""

    def __init__(self, env, device="cpu", ar_beta=0.05, anneal_updates=30, **kw):
        super().__init__(env, device=device, **kw)
        self.ar_beta = ar_beta
        self.anneal_updates = anneal_updates
        self.n_updates = 0
        self.diag = {"ar_c": None, "bonus_mean": 0.0}

    def _bonus(self, buf):
        v = buf["values"]
        rows, tgts, idxs = [], [], []
        for a, b in buf["seg_bounds"]:
            for t in range(a + 2, b):
                rows.append((v[t - 1], v[t - 2]))
                tgts.append(v[t])
                idxs.append(t)
        self.diag["ar_c"] = None
        if len(rows) < 8:
            self.diag["bonus_mean"] = 0.0
            return np.zeros_like(v)
        X, y = np.asarray(rows), np.asarray(tgts)
        c = np.linalg.solve(X.T @ X + 1e-8 * np.eye(2), X.T @ y)
        self.diag["ar_c"] = [float(c[0]), float(c[1])]
        e = np.zeros_like(v)
        e[idxs] = np.abs(y - X @ c)
        e = np.clip((e - e.mean()) / (e.std() + 1e-8), 0.0, None)
        ramp = max(0.0, 1.0 - self.n_updates / max(1, self.anneal_updates))
        bonus = self.ar_beta * ramp * e
        self.diag["bonus_mean"] = float(bonus.mean())
        return bonus

    def update(self, buf):
        orig = buf["rews"]
        buf["rews"] = orig + self._bonus(buf)
        try:
            super().update(buf)
        finally:
            buf["rews"] = orig
        self.n_updates += 1


def build(env, device, overrides):
    kwargs = {k: v for k, v in overrides.items() if k in PPO_KEYS}
    return ARExplorePPO(env=env, device=device,
                        ar_beta=overrides.get("ar_beta", 0.05),
                        anneal_updates=overrides.get("anneal_updates", 30),
                        **kwargs)
