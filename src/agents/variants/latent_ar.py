"""PPO with a shared trunk and a self-predictive latent-AR auxiliary loss:
a linear predictor maps concat(z_(t-1), z_(t-2)) -> z_t (stop-grad target),
shaping the shared representation that feeds both policy and value heads."""
import numpy as np
import torch
import torch.nn as nn

from agents.ppo_agent import PPOAgent

PPO_KEYS = {"hidden_sizes", "nn_learning_rate", "discount_factor", "gae_lambda",
            "rollout_steps", "minibatch_size", "update_epochs", "clip_eps",
            "vf_coef", "ent_coef", "max_grad_norm", "log_std_init"}


class LatentARAgent(PPOAgent):
    def __init__(self, env, device="cpu", aux_weight=0.1,
                 nn_learning_rate=3e-4, **kw):
        super().__init__(env, device=device, nn_learning_rate=nn_learning_rate,
                         **kw)
        obs_dim = env.observation_space.shape[0]
        self.trunk = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                   nn.Linear(64, 64), nn.Tanh()).to(self.device)
        # actor_out_dim: n_act logits, or one Gaussian mean per action dim.
        self.pi_head = nn.Linear(64, self.actor_out_dim).to(self.device)
        self.v_head = nn.Linear(64, 1).to(self.device)
        self.pred = nn.Linear(128, 64).to(self.device)
        self.actor = nn.Sequential(self.trunk, self.pi_head)
        self.critic = nn.Sequential(self.trunk, self.v_head)
        self.aux_weight = aux_weight
        self._params = (list(self.trunk.parameters())
                        + list(self.pi_head.parameters())
                        + list(self.v_head.parameters())
                        + list(self.pred.parameters())
                        + ([self.log_std] if self.continuous else []))
        self.optim = torch.optim.Adam(self._params, lr=nn_learning_rate)
        self.diag = {}

    def update(self, buf):
        obs = torch.as_tensor(buf["obs"], dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(buf["acts"], device=self.device,
                               dtype=torch.float32 if self.continuous else None)
        old_logp = torch.as_tensor(buf["logps"], dtype=torch.float32, device=self.device)

        adv, returns = self._gae(buf)
        T = len(adv)

        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # valid AR targets: t >= a+2 within each segment
        ti = np.asarray([t for (a, b) in buf["seg_bounds"]
                         for t in range(a + 2, b)], dtype=np.int64)

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
                if start == 0 and len(ti):  # aux on FULL rollout, once per epoch
                    z = self.trunk(obs)
                    p = self.pred(torch.cat([z[ti - 1], z[ti - 2]], dim=1))
                    aux = ((p - z[ti].detach()) ** 2).mean()
                    loss = loss + self.aux_weight * aux
                    self.diag = {"aux_loss": float(aux.detach()),
                                 "aux_targets": int(len(ti))}
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._params, self.max_grad_norm)
                self.optim.step()


def build(env, device, overrides):
    kwargs = {k: v for k, v in overrides.items() if k in PPO_KEYS}
    return LatentARAgent(env=env, device=device,
                         aux_weight=overrides.get("aux_weight", 0.1), **kwargs)
