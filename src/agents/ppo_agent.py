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
    """PPO-clip for discrete or continuous actions: separate actor/critic MLPs,
    GAE(lambda) advantages with per-segment bootstrapping, minibatched clip
    updates.

    Discrete actions get a Categorical head. Flat Box actions get a diagonal
    Gaussian with a state-independent learnable log_std, matching SB3's
    DiagGaussianDistribution and CleanRL's ppo_continuous_action. Actions are
    NOT squashed: the raw sample is stored and stepped, and bounds are enforced
    env-side by the ClipAction wrapper (config key clip.action), which is what
    SB3 and CleanRL both do.
    """

    def __init__(self, env, device="cpu", hidden_sizes=(64, 64),
                 nn_learning_rate=3e-4, discount_factor=0.99, gae_lambda=0.95,
                 rollout_steps=2048, minibatch_size=256, update_epochs=10,
                 clip_eps=0.2, vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5,
                 log_std_init=0.0):
        act_space = env.action_space
        assert isinstance(act_space, (Discrete, Box)), (
            f"PPOAgent needs a Discrete or Box action space, got {act_space}")
        self.continuous = isinstance(act_space, Box)
        if self.continuous:
            assert len(act_space.shape) == 1, (
                f"PPOAgent needs a flat Box action space, got {act_space}")
            self.act_dim = act_space.shape[0]
            n_out = self.act_dim
        else:
            self.act_dim = None
            n_out = act_space.n
        assert isinstance(env.observation_space, Box) and \
            len(env.observation_space.shape) == 1, (
            f"PPOAgent needs a flat Box observation space, got "
            f"{env.observation_space} (wrap Discrete-obs envs with one_hot_obs)")
        obs_dim = env.observation_space.shape[0]
        # Actor width: n_act logits, or one Gaussian mean per action dim.
        # Variants that build their own policy head must size it from this.
        self.actor_out_dim = n_out
        self.device = torch.device(device)
        self.actor = _mlp(obs_dim, n_out, hidden_sizes).to(self.device)
        self.critic = _mlp(obs_dim, 1, hidden_sizes).to(self.device)
        # State-independent log std, as in SB3 (log_std_init=0.0 -> std 1.0).
        self.log_std = nn.Parameter(
            torch.full((self.act_dim,), float(log_std_init),
                       device=self.device)) if self.continuous else None
        self.optim = torch.optim.Adam(self._all_params(), lr=nn_learning_rate)
        self.gamma = discount_factor
        self.lam = gae_lambda
        self.rollout_steps = rollout_steps
        self.minibatch_size = minibatch_size
        self.update_epochs = update_epochs
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

    # -- policy head -------------------------------------------------------
    def _all_params(self):
        """Every optimised tensor: actor + critic, plus log_std when continuous.

        Subclasses that rebuild the optimiser or clip gradients must go through
        this -- listing actor/critic parameters by hand drops log_std, and the
        policy spread then silently never trains.
        """
        p = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.continuous:
            p.append(self.log_std)
        return p

    def _dist(self, obs_t):
        """Policy distribution for a batch of observations."""
        return self._dist_from(self.actor(obs_t))

    def _dist_from(self, out, std=None):
        """Policy distribution from an already-computed actor output: logits for
        Categorical, the mean for the Gaussian.

        Subclasses with a shared trunk use this so they can reuse the trunk
        activations instead of paying a second forward. Pass std to pin the
        Gaussian spread to a snapshot, e.g. the reference side of a KL term.
        """
        if not self.continuous:
            return torch.distributions.Categorical(logits=out)
        return torch.distributions.Normal(
            out, self.log_std.exp() if std is None else std)

    @staticmethod
    def _kl(p, q):
        """KL per row, summed over action dims for the diagonal Gaussian."""
        kl = torch.distributions.kl_divergence(p, q)
        return kl.sum(-1) if kl.dim() > 1 else kl

    @staticmethod
    def _logp(dist, acts):
        """Log-prob per row; summed over action dims for the diagonal Gaussian
        (Categorical already returns one value per row)."""
        lp = dist.log_prob(acts)
        return lp.sum(-1) if lp.dim() > 1 else lp

    @staticmethod
    def _entropy(dist):
        e = dist.entropy()
        return e.sum(-1) if e.dim() > 1 else e

    # -- acting ------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs):
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self._dist(t)
        a = dist.sample()
        logp = float(self._logp(dist, a))
        v = float(self.critic(t).squeeze())
        if self.continuous:
            # Raw unclipped sample: log_prob is recomputed against it in
            # update(), so bounds stay the env wrapper's job.
            return a.squeeze(0).cpu().numpy(), logp, v
        return int(a), logp, v

    @torch.no_grad()
    def act_and_value_only(self, obs):
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.critic(t).squeeze())

    @torch.no_grad()
    def act_greedy(self, obs):
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        out = self.actor(t)
        if self.continuous:
            return out.squeeze(0).cpu().numpy()  # distribution mean
        return int(out.argmax(dim=1))

    # -- update ------------------------------------------------------------
    def update(self, buf):
        """buf: dict of np arrays obs (T,d), acts (T,) int64 discrete or
        (T,act_dim) float32 continuous, logps, rews, values (T,), plus
        seg_bounds/seg_terminal/seg_boot_value metadata built by the training
        loop (one segment per episode slice inside the rollout)."""
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

        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

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
