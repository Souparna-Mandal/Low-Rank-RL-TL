import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete


def _mlp(in_dim, out_dim, hidden):
    layers, last = [], in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.Tanh()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class PPOAgent:
    """PPO-clip for discrete actions: separate actor/critic MLPs, GAE(lambda)
    advantages with per-segment bootstrapping, minibatched clip updates."""

    def __init__(self, env, device="cpu", hidden_sizes=(64, 64),
                 nn_learning_rate=3e-4, discount_factor=0.99, gae_lambda=0.95,
                 rollout_steps=2048, minibatch_size=256, update_epochs=10,
                 clip_eps=0.2, vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5):
        assert isinstance(env.action_space, Discrete), (
            f"PPOAgent needs a Discrete action space, got {env.action_space} "
            "(wrap continuous envs with discrete_action_bins)")
        assert isinstance(env.observation_space, Box) and \
            len(env.observation_space.shape) == 1, (
            f"PPOAgent needs a flat Box observation space, got "
            f"{env.observation_space} (wrap Discrete-obs envs with one_hot_obs)")
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

    # -- update ------------------------------------------------------------
    def update(self, buf):
        """buf: dict of np arrays obs (T,d), acts, logps, rews, values (T,),
        plus seg_bounds/seg_terminal/seg_boot_value metadata built by the
        training loop (one segment per episode slice inside the rollout)."""
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
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm)
                self.optim.step()
