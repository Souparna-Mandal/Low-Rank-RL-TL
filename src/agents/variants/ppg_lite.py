"""PPG-lite: shared trunk with policy/value heads. Policy phase trains the
trunk via PPO-clip only (value head fits detached features); every n_pi
updates an aux phase distills value structure into the trunk while a KL term
pins the policy to its pre-aux logits (Cobbe et al., 2020, minimal form)."""
import numpy as np
import torch
import torch.nn as nn

from agents.ppo_agent import PPOAgent

PPO_KEYS = {"hidden_sizes", "nn_learning_rate", "discount_factor", "gae_lambda",
            "rollout_steps", "minibatch_size", "update_epochs", "clip_eps",
            "vf_coef", "ent_coef", "max_grad_norm"}


class PPGLiteAgent(PPOAgent):
    def __init__(self, env, device="cpu", n_pi=4, aux_epochs=4, clone_beta=1.0,
                 **kwargs):
        super().__init__(env, device=device, **kwargs)
        obs_dim = env.observation_space.shape[0]
        n_act = env.action_space.n
        self.trunk = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                   nn.Linear(64, 64), nn.Tanh()).to(self.device)
        self.pi_head = nn.Linear(64, n_act).to(self.device)
        self.v_head = nn.Linear(64, 1).to(self.device)
        # rebind parent nets so act/act_greedy/act_and_value_only work as-is
        self.actor = nn.Sequential(self.trunk, self.pi_head)
        self.critic = nn.Sequential(self.trunk, self.v_head)
        self.params = (list(self.trunk.parameters())
                       + list(self.pi_head.parameters())
                       + list(self.v_head.parameters()))
        self.optim = torch.optim.Adam(self.params,
                                      lr=self.optim.param_groups[0]["lr"])
        self.n_pi = n_pi
        self.aux_epochs = aux_epochs
        self.clone_beta = clone_beta
        self.aux_buf = []  # [(obs np (T,d), returns np (T,)), ...]
        self.n_updates = 0
        self.diag = {"aux_phases": 0}

    def update(self, buf):
        obs = torch.as_tensor(buf["obs"], dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(buf["acts"], device=self.device)
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

        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # -- policy phase: PPO-clip shapes the trunk; value head fits
        # detached features so value gradients stay out of the trunk
        idx = np.arange(T)
        for _ in range(self.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, T, self.minibatch_size):
                mb = idx[start:start + self.minibatch_size]
                phi = self.trunk(obs[mb])
                dist = torch.distributions.Categorical(logits=self.pi_head(phi))
                logp = dist.log_prob(acts[mb])
                ratio = (logp - old_logp[mb]).exp()
                s1 = ratio * adv_t[mb]
                s2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb]
                policy_loss = -torch.min(s1, s2).mean()
                v = self.v_head(phi.detach()).squeeze(-1)
                value_loss = 0.5 * ((v - ret_t[mb]) ** 2).mean()
                loss = (policy_loss + self.vf_coef * value_loss
                        - self.ent_coef * dist.entropy().mean())
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
                self.optim.step()

        self.aux_buf.append((buf["obs"].copy(), returns.astype(np.float32)))
        self.n_updates += 1
        if self.n_updates % self.n_pi == 0:
            self._aux_phase()
            self.aux_buf = []

    def _aux_phase(self):
        obs = torch.as_tensor(np.concatenate([o for o, _ in self.aux_buf]),
                              dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(np.concatenate([r for _, r in self.aux_buf]),
                              dtype=torch.float32, device=self.device)
        with torch.no_grad():  # policy logits at the start of the aux phase
            old_logits = self.pi_head(self.trunk(obs))
        N = len(ret)
        idx = np.arange(N)
        for _ in range(self.aux_epochs):
            np.random.shuffle(idx)
            for start in range(0, N, self.minibatch_size):
                mb = idx[start:start + self.minibatch_size]
                phi = self.trunk(obs[mb])  # undetached: distill into trunk
                v = self.v_head(phi).squeeze(-1)
                kl = torch.distributions.kl_divergence(
                    torch.distributions.Categorical(logits=old_logits[mb]),
                    torch.distributions.Categorical(logits=self.pi_head(phi)))
                loss = (0.5 * ((v - ret[mb]) ** 2).mean()
                        + self.clone_beta * kl.mean())
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
                self.optim.step()
        self.diag["aux_phases"] += 1


def build(env, device, overrides):
    kwargs = {k: v for k, v in overrides.items() if k in PPO_KEYS}
    return PPGLiteAgent(env=env, device=device,
                        n_pi=int(overrides.get("n_pi", 4)),
                        aux_epochs=int(overrides.get("aux_epochs", 4)),
                        clone_beta=float(overrides.get("clone_beta", 1.0)),
                        **kwargs)
