"""SSM critic: the value head is a low-order diagonal linear state-space
model along the trajectory (a low-Hankel-rank prior by architecture)."""
import numpy as np
import torch
import torch.nn as nn

from agents.ppo_agent import PPOAgent

PPO_KEYS = {"hidden_sizes", "nn_learning_rate", "discount_factor", "gae_lambda",
            "rollout_steps", "minibatch_size", "update_epochs", "clip_eps",
            "vf_coef", "ent_coef", "max_grad_norm"}


class _SSMCritic(nn.Module):
    """v_t = C h_t + D phi_t with h_t = a * h_(t-1) + B phi_t, a in (0,1)^r."""

    def __init__(self, obs_dim, r, feat=64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, feat), nn.Tanh(),
                                   nn.Linear(feat, feat), nn.Tanh())
        self.B = nn.Linear(feat, r)
        self.C = nn.Linear(r, 1)
        self.D = nn.Linear(feat, 1)
        u = 0.5 + 0.49 * torch.rand(r)  # a ~ U(0.5, 0.99) at init
        self.decay_logits = nn.Parameter(torch.log(u / (1 - u)))

    def step(self, phi, h):
        h = torch.sigmoid(self.decay_logits) * h + self.B(phi)
        return h, (self.C(h) + self.D(phi)).squeeze(-1)


class SSMCriticAgent(PPOAgent):
    def __init__(self, env, device="cpu", ssm_rank=8, **kwargs):
        super().__init__(env, device=device, **kwargs)
        self.r = ssm_rank
        self.critic = _SSMCritic(env.observation_space.shape[0],
                                 ssm_rank).to(self.device)
        self.optim = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.optim.param_groups[0]["lr"])
        self.h = None  # live hidden state; set by begin_episode()

    def begin_episode(self):
        self.h = torch.zeros(self.r, device=self.device)

    @torch.no_grad()
    def act(self, obs):
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        dist = torch.distributions.Categorical(logits=self.actor(t.unsqueeze(0)))
        a = dist.sample()
        self.h, v = self.critic.step(self.critic.trunk(t), self.h)
        return int(a), float(dist.log_prob(a)), float(v)

    @torch.no_grad()
    def act_and_value_only(self, obs):
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        _, v = self.critic.step(self.critic.trunk(t), self.h)  # peek, no commit
        return float(v)

    def _seq_values(self, obs, seg_bounds, seg_h0):
        phi = self.critic.trunk(obs)
        bx = self.critic.B(phi)
        a = torch.sigmoid(self.critic.decay_logits)
        hs = []
        for (s, e), h in zip(seg_bounds, seg_h0):
            for t in range(s, e):
                h = a * h + bx[t]
                hs.append(h)
        return (self.critic.C(torch.stack(hs)) + self.critic.D(phi)).squeeze(-1)

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
        seg_h0 = [h.to(self.device) for h in buf["seg_h0"]]

        params = list(self.actor.parameters()) + list(self.critic.parameters())
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
                v = self._seq_values(obs, buf["seg_bounds"], seg_h0)
                value_loss = 0.5 * ((v[mb] - ret_t[mb]) ** 2).mean()
                loss = (policy_loss + self.vf_coef * value_loss
                        - self.ent_coef * dist.entropy().mean())
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                self.optim.step()


def collect_rollout(agent, env, state, ep_ret):
    n = agent.rollout_steps
    obs_dim = env.observation_space.shape[0]
    buf = {"obs": np.zeros((n, obs_dim), np.float32), "acts": np.zeros(n, np.int64),
           "logps": np.zeros(n, np.float32), "rews": np.zeros(n, np.float64),
           "values": np.zeros(n, np.float64)}
    seg_bounds, seg_terminal, seg_boot, seg_h0 = [], [], [], []
    finished_returns = []
    seg_start = 0
    if agent.h is None:  # hidden persists across rollouts mid-episode
        agent.begin_episode()
    for t in range(n):
        if t == seg_start:
            seg_h0.append(agent.h.detach().clone())
        a, logp, v = agent.act(state)
        buf["obs"][t] = state
        buf["acts"][t], buf["logps"][t], buf["values"][t] = a, logp, v
        state, r, terminated, truncated, _ = env.step(a)
        buf["rews"][t] = r
        ep_ret += r
        if terminated or truncated:
            seg_bounds.append((seg_start, t + 1))
            seg_terminal.append(bool(terminated))
            # env-truncated (TimeLimit): bootstrap with V of the final state
            seg_boot.append(0.0 if terminated else
                            agent.act_and_value_only(state))
            finished_returns.append(ep_ret)
            ep_ret = 0.0
            state, _ = env.reset()
            agent.begin_episode()
            seg_start = t + 1
    if seg_start < n:  # rollout-cut segment: bootstrap with V(next state)
        seg_bounds.append((seg_start, n))
        seg_terminal.append(False)
        seg_boot.append(agent.act_and_value_only(state))
    buf["seg_bounds"] = seg_bounds
    buf["seg_terminal"] = seg_terminal
    buf["seg_boot_value"] = seg_boot
    buf["seg_h0"] = seg_h0
    return buf, state, ep_ret, finished_returns


def build(env, device, overrides):
    kwargs = {k: v for k, v in overrides.items() if k in PPO_KEYS}
    return SSMCriticAgent(env=env, device=device,
                          ssm_rank=int(overrides.get("ssm_rank", 8)), **kwargs)
