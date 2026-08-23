"""Rainbow-style agent (IQN variant) — a benchmark sibling to :class:`QAgent`.

Design: **you pass in only an encoder** (a plain feature extractor, like a Neural network CNN Encoder
*body*), and this module wraps it with the reusable Rainbow head
(:class:`RainbowIQNNetwork`) that adds the distributional + dueling + noisy
network layers. So the algorithm-specific head is centralised here
and shared by every experiment.

Components (Rainbow with IQN swapped in for C51 for distributional head):
  * Double DQN          — ``double`` flag (enables selecting next-action via the policy net)
  * Dueling             — value/advantage streams in the head, combined per quantile
  * Prioritised replay  — :class:`PrioritizedReplayBuffer` (proportional, sum tree)
  * Multi-step returns  — n-step accumulator in :meth:`_ingest`
  * Distributional (IQN)— implicit quantile network, quantile-Huber regression
  * Noisy Nets          — :class:`NoisyLinear` in the head; sole exploration source

The shared training loop (``src/training.py``) is untouched: it still calls
``pi`` / ``update_buffer[_atari]`` / ``decay_epsilon`` / ``update_target_network``
/ ``train`` and reads ``replay_buffer`` / ``epsilon``. Exploration is noisy-only —
the ``EpsilonGreedyExplorer`` mixin is initialised inert (ε ≡ 0) purely so those
hooks keep working. The low-rank Hankel contract is preserved: ``policy_net(x)``
returns expected-value Q ``(B, n_actions)`` and ``value_advantage`` exposes the
dueling streams for a meaningful advantage trace.

Encoder contract (the only thing an experiment supplies)
--------------------------------------------------------
    encoder(obs) -> Tensor (B, feature_dim)

The encoder owns its input normalisation (e.g. uint8 → /255, like ``NatureCNN``).
``feature_dim`` and ``n_actions`` are inferred by the agent from a dummy forward
and ``env`` — so ``nn_extra_kwargs`` need only carry encoder-construction args.
"""
from __future__ import annotations

import math
import random
from collections import deque, namedtuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_agent import BaseAgent, EpsilonGreedyExplorer
from .q_agent import QAgent

# An n-step transition already has its return aggregated. ``discount`` is
# gamma**k where k is the number of real steps folded into ``reward`` (k == n_step
# for full windows, shorter for the tail flushed at episode end); it multiplies the
# bootstrap value of ``next_state`` (which is None on terminal).
NStepTransition = namedtuple("NStepTransition",
                             ("state", "action", "next_state", "reward", "discount"))


class NoisyLinear(nn.Module):
    """Factorised-Gaussian noisy linear layer (Fortunato et al., 2018).

    Replaces ε-greedy: weights are ``mu + sigma * epsilon`` with ``epsilon``
    resampled by :meth:`reset_noise`, so exploration is learned and state-conditioned.
    In ``eval()`` mode only ``mu`` is used, giving a deterministic greedy net.
    """

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma0 = sigma0

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma0 / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma0 / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()  # f(x) = sgn(x) sqrt(|x|)

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(eps_out, eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(x, weight, bias)


class RainbowIQNNetwork(nn.Module):
    """Dueling + noisy Implicit Quantile Network head wrapped around an encoder.

    Given encoder features ``phi = encoder(x)`` of shape ``(B, F)`` and quantile
    fractions ``taus`` of shape ``(B, N)``:

      * embed each tau with a cosine basis and a (plain) linear layer -> ``(B, N, F)``;
      * multiply element-wise into ``phi`` (IQN's Hadamard conditioning) -> ``(B, N, F)``;
      * pass through dueling noisy value/advantage streams and combine per quantile.

    Quantile output has shape ``(B, N, n_actions)``. ``forward`` averages over
    sampled taus to give expected-value Q ``(B, n_actions)`` for greedy acting and
    the Hankel analysis; ``value_advantage`` exposes the (expected) dueling streams.
    """

    def __init__(self, encoder: nn.Module, feature_dim: int, n_actions: int,
                 n_cos: int = 64, head_hidden: int = 512, sigma0: float = 0.5,
                 dueling: bool = True, n_quantiles_act: int = 32):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.n_actions = n_actions
        self.n_cos = n_cos
        self.dueling = dueling
        self.n_quantiles_act = n_quantiles_act

        # Cosine tau-embedding projected into feature space (plain, per IQN).
        self.tau_fc = nn.Linear(n_cos, feature_dim)
        self.register_buffer("_cos_freqs",
                             math.pi * torch.arange(1, n_cos + 1, dtype=torch.float32))

        # Dueling noisy streams operating on the (phi * tau_emb) conditioned features.
        self.adv_hidden = NoisyLinear(feature_dim, head_hidden, sigma0)
        self.adv = NoisyLinear(head_hidden, n_actions, sigma0)
        if dueling:
            self.val_hidden = NoisyLinear(feature_dim, head_hidden, sigma0)
            self.val = NoisyLinear(head_hidden, 1, sigma0)

    def _tau_embed(self, taus: torch.Tensor) -> torch.Tensor:
        # taus: (B, N) -> cosine features (B, N, n_cos) -> (B, N, F)
        cos = torch.cos(taus.unsqueeze(-1) * self._cos_freqs)
        return F.relu(self.tau_fc(cos))

    def quantiles(self, x: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        """Quantile values at the given fractions. Returns (B, N, n_actions)."""
        phi = self.encoder(x)                       # (B, F)
        h = phi.unsqueeze(1) * self._tau_embed(taus)  # (B, N, F)
        adv = self.adv(F.relu(self.adv_hidden(h)))    # (B, N, A)
        if not self.dueling:
            return adv
        val = self.val(F.relu(self.val_hidden(h)))    # (B, N, 1)
        return val + adv - adv.mean(dim=-1, keepdim=True)

    def _sample_taus(self, batch: int, n: int, device) -> torch.Tensor:
        return torch.rand(batch, n, device=device)

    def forward(self, x: torch.Tensor, n_taus: int | None = None) -> torch.Tensor:
        """Expected-value Q ``(B, n_actions)`` = mean of quantiles over sampled taus."""
        n = n_taus or self.n_quantiles_act
        taus = self._sample_taus(x.shape[0], n, x.device)
        return self.quantiles(x, taus).mean(dim=1)

    def value_advantage(self, x: torch.Tensor, n_taus: int | None = None):
        """Expected dueling streams: V ``(B, 1)`` and centred A ``(B, n_actions)``.
        Falls back to (max_a Q, Q - max_a Q) when built without dueling."""
        n = n_taus or self.n_quantiles_act
        taus = self._sample_taus(x.shape[0], n, x.device)
        phi = self.encoder(x)
        h = phi.unsqueeze(1) * self._tau_embed(taus)
        adv = self.adv(F.relu(self.adv_hidden(h))).mean(dim=1)  # (B, A)
        if not self.dueling:
            v = adv.max(dim=1, keepdim=True).values
            return v, adv - v
        val = self.val(F.relu(self.val_hidden(h))).mean(dim=1)  # (B, 1)
        return val, adv - adv.mean(dim=1, keepdim=True)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class SumTree:
    """Array-backed sum tree: leaves hold priorities, internal nodes hold subtree
    sums, so proportional sampling and priority updates are both O(log capacity).
    ``data`` is a ring buffer parallel to the leaves.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.empty(capacity, dtype=object)
        self.ptr = 0
        self.size = 0

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data) -> None:
        leaf = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(leaf, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, leaf: int, priority: float) -> None:
        change = priority - self.tree[leaf]
        self.tree[leaf] = priority
        idx = leaf
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s: float):
        """Return (leaf_index, priority, data) for cumulative sum ``s`` in [0, total)."""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):  # reached a leaf
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Proportional prioritised replay (Schaul et al., 2016).

    Priorities are stored as ``p ** alpha`` in the sum tree; sampling probability
    is ``p_i**alpha / sum_j p_j**alpha``. Importance-sampling weights
    ``(N * P(i))**-beta`` correct the induced bias and are normalised by their max
    so they only scale the loss down. ``beta`` anneals towards 1 each sample.
    """

    def __init__(self, capacity: int, alpha: float, beta_start: float,
                 beta_increment: float, eps: float = 1e-5):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = beta_increment
        self.eps = eps
        self.max_priority = 1.0  # raw priority; new samples enter at the current max

    def push(self, transition) -> None:
        self.tree.add(self.max_priority ** self.alpha, transition)

    def sample(self, batch_size: int):
        batch, idxs, priorities = [], [], []
        total = self.tree.total()
        segment = total / batch_size
        for i in range(batch_size):
            # stratified sampling: one draw per equal-mass segment
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s)
            while data is None:  # guard a boundary draw landing on an unfilled leaf
                idx, p, data = self.tree.get(random.uniform(0, total))
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        probs = np.asarray(priorities, dtype=np.float64) / total
        weights = (self.tree.size * probs) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        return batch, idxs, weights.astype(np.float32)

    def update_priorities(self, idxs, td_errors) -> None:
        for idx, err in zip(idxs, td_errors):
            p = abs(float(err)) + self.eps
            self.max_priority = max(self.max_priority, p)
            self.tree.update(idx, p ** self.alpha)

    def __len__(self) -> int:
        return self.tree.size


class RainbowDQNAgent(QAgent):
    """Noisy, dueling, prioritised, multi-step Double-DQN with IQN distributional RL.

    Reuses ``QAgent.save`` / ``load`` / ``update_target_network`` / ``act_greedy``
    and overrides only what Rainbow changes: it wraps the supplied encoder with a
    :class:`RainbowIQNNetwork` head, buffers n-step transitions into a prioritised
    replay, acts noisy-greedily (no ε), and learns via IS-weighted quantile-Huber
    regression with priority feedback.
    """

    def __init__(
        self,
        replay_buffer_capacity: int,
        q_network,                 # encoder class/callable -> nn.Module (obs -> features)
        batch_size: int,
        nn_learning_rate: float,
        nn_extra_kwargs: dict,     # encoder-construction args only
        env: gym.Env,
        discount_factor: float,
        n_step: int = 3,
        n_quantiles: int = 32,          # online quantile samples N (loss)
        n_quantiles_target: int = 32,   # target quantile samples N' (loss)
        n_quantiles_act: int = 32,      # quantile samples for expected-Q acting
        n_cos: int = 64,                # cosine embedding size for tau
        head_hidden: int = 512,
        huber_kappa: float = 1.0,
        noisy_sigma0: float = 0.5,
        dueling: bool = True,
        per_alpha: float = 0.5,
        per_beta_start: float = 0.4,
        per_beta_increment: float = 1e-5,
        per_eps: float = 1e-5,
        device: str = "cuda",
        TD_LR: float = 0.1,
        buffer_util: int = 1,
        gd_steps_ceil: int = 100,
        grad_clip_norm: float = 10.0,
        double: bool = True,
    ):
        """See the module docstring for the (minimal) encoder contract.

        Args:
            q_network: encoder class; instantiated as ``q_network(**nn_extra_kwargs)``
                and expected to map an observation batch to features ``(B, F)``.
            n_step: multi-step horizon (1 recovers standard 1-step TD).
            n_quantiles / n_quantiles_target: IQN online/target quantile sample counts.
            n_quantiles_act: quantiles averaged for the expected-Q used in acting,
                Double-DQN action selection, and the Hankel Q/V/A traces.
            n_cos / head_hidden / huber_kappa / noisy_sigma0 / dueling: head config.
            per_*: prioritised-replay exponents, IS-weight annealing, priority floor.
            double: Double-DQN next-action selection (recommended True for Rainbow).
            Remaining args mirror ``QAgent``.
        """
        BaseAgent.__init__(self, env)
        # Noisy-only exploration: initialise the explorer inert so decay_epsilon is a
        # no-op and agent.epsilon (read by the shared loop's DEBUG print) stays 0.
        EpsilonGreedyExplorer.__init__(self, eps_start=0.0, eps_min=0.0, decay_rate=1.0)

        self.device = device
        n_actions = int(env.action_space.n)

        def build_net():
            encoder = q_network(**nn_extra_kwargs).to(device)
            feature_dim = self._infer_feature_dim(encoder, env)
            return RainbowIQNNetwork(
                encoder, feature_dim, n_actions,
                n_cos=n_cos, head_hidden=head_hidden, sigma0=noisy_sigma0,
                dueling=dueling, n_quantiles_act=n_quantiles_act).to(device)

        self.policy_net = build_net()
        self.target_net = build_net()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # NB: unlike QAgent we do NOT call target_net.eval(); the target must keep
        # its NoisyLinear noise active (resampled every gradient step).

        self.optimiser = optim.AdamW(self.policy_net.parameters(),
                                     lr=nn_learning_rate, amsgrad=True)
        self.batch_size = batch_size
        self.gamma = discount_factor
        self.TD_LR = TD_LR
        self.buffer_util = buffer_util
        self.gd_steps_ceil = gd_steps_ceil
        self.grad_clip_norm = grad_clip_norm
        self.double = double

        # Multi-step returns
        self.n_step = n_step
        self._nstep: deque = deque(maxlen=n_step)

        # Prioritised replay
        self.replay_buffer = PrioritizedReplayBuffer(
            replay_buffer_capacity, per_alpha, per_beta_start, per_beta_increment, per_eps)

        # IQN loss config
        self.n_quantiles = n_quantiles
        self.n_quantiles_target = n_quantiles_target
        self.huber_kappa = huber_kappa

    @staticmethod
    def _infer_feature_dim(encoder: nn.Module, env: gym.Env) -> int:
        """Feature width from a dummy forward through the encoder (respects a
        ``feature_dim`` attribute if the encoder already exposes one)."""
        if hasattr(encoder, "feature_dim"):
            return int(encoder.feature_dim)
        obs_shape = env.observation_space.shape
        device = next(encoder.parameters()).device
        was_training = encoder.training
        encoder.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape, device=device)
            out = encoder(dummy)
        encoder.train(was_training)
        return int(out.shape[1])

    # ------------------------------------------------------------------ acting
    def pi(self, state: np.ndarray):
        """Noisy-greedy action: resample the network's noise, then argmax expected Q.
        Exploration comes entirely from the NoisyLinear layers — no ε branch."""
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.policy_net.reset_noise()
        return self.act_greedy(state_t)  # inherited: policy_net(state).argmax(1).item()

    # --------------------------------------------------------------- buffering
    def _ingest(self, state_t, action_t, reward, next_state_t, terminated, truncated=False):
        """Feed one env transition into the n-step accumulator, emitting aggregated
        n-step transitions into the prioritised buffer as windows complete. On
        truncation the tail windows are flushed too (bootstrapping from the stored
        next_state, not marked terminal) so no window spans the episode reset."""
        self._nstep.append((state_t, action_t, float(reward), next_state_t, terminated))
        if len(self._nstep) < self.n_step and not (terminated or truncated):
            return
        self._emit_front()
        if terminated or truncated:
            # Flush the shorter tail windows that end in this final transition.
            while len(self._nstep) > 1:
                self._nstep.popleft()
                self._emit_front()
            self._nstep.clear()

    def _emit_front(self):
        """Push the n-step transition whose start is the oldest entry in the deque."""
        R, disc = 0.0, 1.0
        next_state, term = None, False
        for _s, _a, r, ns, d in self._nstep:
            R += disc * r
            disc *= self.gamma
            next_state, term = ns, d
            if d:
                break
        s0, a0 = self._nstep[0][0], self._nstep[0][1]
        reward_t = torch.tensor([R], dtype=torch.float32, device=self.device)
        self.replay_buffer.push(
            NStepTransition(s0, a0, None if term else next_state, reward_t, disc))

    def update_buffer(self, state, action, reward, next_state, terminated, truncated=False):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = torch.tensor([action], dtype=torch.long, device=self.device)
        next_state_t = (None if terminated else
                        torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0))
        self._ingest(state_t, action_t, reward, next_state_t, terminated, truncated)

    def update_buffer_atari(self, state, action, reward, next_state, terminated, truncated=False):
        state_t = torch.tensor(state, dtype=torch.uint8).unsqueeze(0)
        action_t = torch.tensor([action], dtype=torch.long, device=self.device)
        next_state_t = (None if terminated else
                        torch.tensor(next_state, dtype=torch.uint8).unsqueeze(0))
        self._ingest(state_t, action_t, reward, next_state_t, terminated, truncated)

    # --------------------------------------------------------------- learning
    def _quantile_huber(self, td, taus):
        """Quantile-Huber loss element ``rho`` for pairwise TD errors.

        Args:
            td: (B, N, N') pairwise errors target_j - theta_i.
            taus: (B, N) online quantile fractions (indexing dim 1 of ``td``).
        Returns:
            (B, N, N') per-pair loss.
        """
        abs_td = td.abs()
        huber = torch.where(abs_td <= self.huber_kappa,
                            0.5 * td.pow(2),
                            self.huber_kappa * (abs_td - 0.5 * self.huber_kappa))
        # |tau - 1{td < 0}| * huber / kappa
        indicator = (td.detach() < 0).float()
        return (taus.unsqueeze(2) - indicator).abs() * huber / self.huber_kappa

    def _target_quantiles(self, rewards, discounts, non_final_mask, next_states):
        """Bellman-backed target quantiles ``T_theta`` of shape (B, N')."""
        B = self.batch_size
        tau_p = self.policy_net._sample_taus(B, self.n_quantiles_target, self.device)
        next_theta_a = torch.zeros(B, self.n_quantiles_target, device=self.device)
        if non_final_mask.any():
            if self.double:
                next_actions = self.policy_net(next_states).argmax(1)  # expected-Q select
            else:
                next_actions = self.target_net(next_states).argmax(1)
            nf_q = self.target_net.quantiles(next_states, tau_p[non_final_mask])  # (nf,N',A)
            gathered = nf_q.gather(
                2, next_actions.view(-1, 1, 1).expand(-1, self.n_quantiles_target, 1)).squeeze(2)
            next_theta_a[non_final_mask] = gathered  # (nf, N')
        # terminal rows keep next_theta_a = 0 (masked by non_final below)
        nf = non_final_mask.float().unsqueeze(1)
        return rewards.unsqueeze(1) + nf * discounts.unsqueeze(1) * next_theta_a

    def train(self):
        gd_steps = min(len(self.replay_buffer) // (self.buffer_util * self.batch_size),
                       self.gd_steps_ceil)
        for _ in range(gd_steps):
            self.policy_net.reset_noise()
            self.target_net.reset_noise()

            batch, idxs, weights = self.replay_buffer.sample(self.batch_size)
            weights = torch.as_tensor(weights, device=self.device)
            tb = NStepTransition(*zip(*batch))

            states = torch.cat(tb.state).to(self.device)
            actions = torch.cat(tb.action).to(self.device)          # (B,)
            rewards = torch.cat(tb.reward).to(self.device)          # (B,)
            discounts = torch.tensor(tb.discount, dtype=torch.float32, device=self.device)
            non_final_mask = torch.tensor([s is not None for s in tb.next_state],
                                          device=self.device)

            with torch.no_grad():
                next_states = (torch.cat([s for s in tb.next_state if s is not None]).to(self.device)
                               if non_final_mask.any() else None)
                target_theta = self._target_quantiles(
                    rewards, discounts, non_final_mask, next_states)   # (B, N')

            # online quantiles for the taken action
            taus = self.policy_net._sample_taus(self.batch_size, self.n_quantiles, self.device)
            theta = self.policy_net.quantiles(states, taus)           # (B, N, A)
            theta_a = theta.gather(
                2, actions.view(-1, 1, 1).expand(-1, self.n_quantiles, 1)).squeeze(2)  # (B, N)

            # pairwise TD: (B, N, N') = target_j - theta_i
            td = target_theta.unsqueeze(1) - theta_a.unsqueeze(2)
            rho = self._quantile_huber(td, taus)                      # (B, N, N')
            # IQN loss: sum over online quantiles i, mean over target quantiles j
            loss_per_sample = rho.mean(dim=2).sum(dim=1)              # (B,)
            loss = (weights * loss_per_sample).mean()

            self.optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm)
            self.optimiser.step()

            self.replay_buffer.update_priorities(
                idxs, loss_per_sample.detach().cpu().numpy())
