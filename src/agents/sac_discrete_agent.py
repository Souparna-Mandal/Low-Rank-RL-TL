"""SACDiscreteAgent — SAC-Discrete (Christodoulou 2019) for the Atari-100k
suite, with the FHR recurrence penalty on both critics.

Categorical actor + twin expected-Q critics on ONE shared encoder (thin
encoder contract: `q_network`/`nn_extra_kwargs` supply an encoder, heads are
built in-agent; sizes follow CleanRL's sac_atari — 512-hidden heads on the
Nature CNN). Every expectation over actions is computed EXACTLY from the
categorical policy — no sampling, no reparameterisation:

    y          = R_n + gamma^m * sum_a' pi(a'|s_{t+n}) [min_i Qbar_i(s_{t+n},a')
                                                        - alpha log pi(a'|s_{t+n})]
    critic     = 0.5 * (MSE(Q_1(s,a), y) + MSE(Q_2(s,a), y))     [+ FHR penalty]
    actor      = E_s sum_a pi(a|s) (alpha log pi - min_i Q_i(s,a).detach())
    alpha      = -log_alpha * (E_pi[log pi] + target_entropy).detach()
    target_entropy = target_entropy_scale * log|A|   (default 0.89, the
                     CleanRL Atari value; Christodoulou's 0.98 pins pi near
                     uniform)

Composition mirrors EfficientRainbowAgent: :class:`FHRDQNAgent` contributes
the FHR machinery unchanged (c/d group semantics, hard warm-up, lambda
ramp-down, episodic buffer plumbing, diagnostics aggregation); only
``_train_step`` and the acting/persistence surface are new. The penalty rides
BOTH critics with ONE shared c (or one shared c(s,a) head), through the fused
(n, r+1) forward with one shared DrQ shift/intensity draw per sequence —
0.5 * (Huber_1 + Huber_2) keeps penalty_raw on the single-critic scale so
lambda values and ramp-down bars transfer from the effrainbow recipe. The
actor loss is untouched: the KL(pi || exp(Q/alpha)/Z) step consumes the
regularised critics implicitly.

Gradient routing (single fused backward per step): the critics own the
encoder — the actor head reads phi.detach() (SAC-AE/DrQ convention), so FHR
stays the sole critic-side representation shaper; Q is detached inside the
actor loss and probs/log-probs inside the alpha loss, so each loss moves only
its own parameters. Optimiser groups (BBF rebuild pattern, c-group LAST — the
invariant the FHR tests pin): [encoder+critics @ nn_learning_rate, actor @
actor_learning_rate, log_alpha @ alpha_learning_rate, c/c_head @
c_learning_rate].

Exploration is the stochastic policy itself (set eps_start/eps_min to 0 in
configs; the inherited epsilon still gates a uniform-random override so
evaluate_policy_atari's eps=0.001 protocol applies unchanged). ``pi()``
samples pi while ``self.sample_actions`` (training); the launcher sets it
False before eval, making pi() argmax with prob 1-eps — comparable with the
other recipes' greedy eval.

Prioritized replay: ``prioritized_replay: true`` swaps in
PrioritizedEpisodicReplayBuffer; IS weights multiply the per-sample TD terms
only (the FHR penalty stays an unweighted mean), priorities are the mean twin
|TD error|, updated only after a successful (non-NaN-skipped) optimiser step.

c(s, a): ``c_predictor: shared`` only — a zero-init linear head on the shared
encoder features ++ one-hot(a) (Bellman-init bias, so ccond == global c at
initialisation; a "separate" MLP on raw 4x84x84 frames is impractical). The
global ``self.c`` stays as the frozen init reference, receiving no gradients,
exactly as the SB3 ccond path does.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .augmentation import intensity, random_shift
from .fhrdqn_agent import FHRDQNAgent
from .per_episodic_buffer import PrioritizedEpisodicReplayBuffer
from .rainbow_agent import RainbowDQNAgent


class SACDiscreteNetwork(nn.Module):
    """Shared encoder + actor logits head + twin expected-Q heads.
    forward(x) returns the elementwise min of the twin Q rows, (B, |A|) —
    the value surface the actor optimises and the analysis stack's
    ``agent.policy_net(x)`` contract."""

    def __init__(self, encoder: nn.Module, feature_dim: int, n_actions: int,
                 head_hidden: int = 512):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.n_actions = n_actions

        def head():
            return nn.Sequential(nn.Linear(feature_dim, head_hidden),
                                 nn.ReLU(),
                                 nn.Linear(head_hidden, n_actions))
        self.actor = head()
        self.q1 = head()
        self.q2 = head()

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def critic_values(self, phi: torch.Tensor):
        return self.q1(phi), self.q2(phi)

    def actor_logits(self, phi: torch.Tensor) -> torch.Tensor:
        return self.actor(phi)

    @staticmethod
    def policy(logits: torch.Tensor):
        """(probs, log_probs) via log_softmax — never log(probs), so the
        exact-expectation losses stay finite under probability underflow."""
        log_probs = F.log_softmax(logits, dim=1)
        return log_probs.exp(), log_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.critic_values(self.features(x))
        return torch.min(q1, q2)


class CCondHead(nn.Module):
    """Native shared-mode port of sb3_fhr.FHRCoefficientPredictor: one linear
    layer on [encoder features, one-hot(a)] with ZERO initial weights and the
    Bellman-init bias, so a ccond run equals the global-c run until gradients
    differentiate them. Kept small and slow-learning (c_learning_rate group)
    for the same degeneracy reason documented there."""

    def __init__(self, order: int, gamma: float, reward_lags: bool,
                 feature_dim: int, n_actions: int):
        super().__init__()
        self.order = order
        self.reward_lags = reward_lags
        self.n_actions = n_actions
        out_dim = order * (2 if reward_lags else 1)
        self.net = nn.Linear(feature_dim + n_actions, out_dim)
        c0 = torch.zeros(order)
        d0 = torch.zeros(order) if reward_lags else None
        if reward_lags:
            c0[0] = 1.0 / gamma
            d0[0] = -1.0 / gamma
        elif order >= 2:
            c0[0], c0[1] = 1.0 + 1.0 / gamma, -1.0 / gamma
        else:
            c0[0] = 1.0 / gamma
        with torch.no_grad():
            self.net.weight.zero_()
            self.net.bias.copy_(torch.cat([c0, d0]) if reward_lags else c0)

    def forward(self, features: torch.Tensor, actions: torch.Tensor):
        onehot = F.one_hot(actions.long().reshape(-1), self.n_actions).float()
        out = self.net(torch.cat([features, onehot], dim=1))
        if self.reward_lags:
            return out[:, :self.order], out[:, self.order:]
        return out, None


class SACDiscreteAgent(FHRDQNAgent):
    """See the module docstring. Extra kwargs over FHRDQNAgent (all mappable
    to config ``agent:`` keys):

        n_step: multi-step soft-target horizon (sample-time aggregation).
        head_hidden: width of the actor/critic heads (CleanRL sac_atari: 512).
        actor_learning_rate / alpha_learning_rate: per-group lrs (the critic
            + encoder group uses nn_learning_rate).
        target_entropy_scale / init_alpha: temperature machinery.
        use_augmentation / aug_pad / aug_intensity: DrQ augmentation, image
            observations only.
        c_predictor: "none" | "shared" (see CCondHead).
        prioritized_replay + per_alpha / per_beta_start / per_beta_increment /
            per_eps: opt-in PER (PrioritizedEpisodicReplayBuffer).
    """

    def __init__(self, *, q_network, nn_extra_kwargs,
                 n_step: int = 10, head_hidden: int = 512,
                 actor_learning_rate: float = 1e-4,
                 alpha_learning_rate: float = 3e-4,
                 target_entropy_scale: float = 0.89,
                 init_alpha: float = 1.0,
                 use_augmentation: bool = False, aug_pad: int = 4,
                 aug_intensity: float = 0.05,
                 c_predictor: str = "none",
                 prioritized_replay: bool = False, per_alpha: float = 0.5,
                 per_beta_start: float = 0.4,
                 per_beta_increment: float = 3e-6, per_eps: float = 1e-5,
                 **fhr_kwargs):
        if n_step < 1:
            raise ValueError(f"n_step must be >= 1, got {n_step}")
        if c_predictor not in ("none", "shared"):
            raise ValueError("c_predictor must be none|shared for the native "
                             f"SACD agent, got {c_predictor!r}")
        if not target_entropy_scale > 0:
            raise ValueError(f"target_entropy_scale must be > 0, got "
                             f"{target_entropy_scale}")
        if not init_alpha > 0:
            raise ValueError(f"init_alpha must be > 0, got {init_alpha}")

        env = fhr_kwargs["env"]
        n_actions = int(env.action_space.n)

        def net_factory():
            encoder = q_network(**nn_extra_kwargs)
            feature_dim = RainbowDQNAgent._infer_feature_dim(encoder, env)
            return SACDiscreteNetwork(encoder, feature_dim, n_actions,
                                      head_hidden=head_hidden)

        # QAgent's ctor (via FHRDQNAgent) instantiates policy/target
        # containers from the factory and creates the optimiser BEFORE
        # FHRDQNAgent appends the c/d group (the ordering the FHR tests pin);
        # the groups are rebuilt below with the same c-group-last invariant.
        super().__init__(q_network=net_factory, nn_extra_kwargs={},
                         **fhr_kwargs)
        self._net_factory = net_factory

        self.n_step = n_step
        self.use_augmentation = use_augmentation
        self.aug_pad = aug_pad
        self.aug_intensity = aug_intensity
        self.c_predictor = c_predictor
        self.prioritized_replay = prioritized_replay
        self.sample_actions = True     # launcher sets False for greedy eval

        if prioritized_replay:
            self.replay_buffer = PrioritizedEpisodicReplayBuffer(
                fhr_kwargs["replay_buffer_capacity"], alpha=per_alpha,
                beta_start=per_beta_start, beta_increment=per_beta_increment,
                eps=per_eps)

        self.log_alpha = torch.nn.Parameter(torch.tensor(
            math.log(init_alpha), dtype=torch.float32, device=self.device))
        self.target_entropy = float(target_entropy_scale
                                    * math.log(n_actions))

        self.c_head = None
        if c_predictor == "shared":
            # the global self.c stays as the frozen init reference (no
            # gradients), exactly as the SB3 ccond path keeps fhr_head
            self.c_head = CCondHead(self.fhr_order, self.loss.gamma,
                                    self.reward_lags,
                                    self.policy_net.feature_dim,
                                    n_actions).to(self.device)

        # Rebuild the optimiser (BBF precedent): critic(+encoder) / actor /
        # alpha / c — c group LAST with its own lr and wd 0. log_alpha, c and
        # c_head are agent-level, so grad clipping (policy_net.parameters())
        # and the target sync exclude them automatically.
        g0, cg = self.optimiser.param_groups[0], self.optimiser.param_groups[-1]
        critic_params = (list(self.policy_net.encoder.parameters())
                         + list(self.policy_net.q1.parameters())
                         + list(self.policy_net.q2.parameters()))
        c_params = (list(self.c_head.parameters())
                    if self.c_head is not None else cg["params"])
        self.optimiser = optim.AdamW(
            [{"params": critic_params},
             {"params": list(self.policy_net.actor.parameters()),
              "lr": actor_learning_rate},
             {"params": [self.log_alpha], "lr": alpha_learning_rate,
              "weight_decay": 0.0},
             {"params": c_params, "lr": cg["lr"], "weight_decay": 0.0}],
            lr=g0["lr"], eps=g0["eps"], amsgrad=g0["amsgrad"],
            weight_decay=g0["weight_decay"])

    # ------------------------------------------------------------- acting
    def act_greedy(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            logits = self.policy_net.actor_logits(
                self.policy_net.features(state))
            return int(logits.argmax(dim=1).item())

    def pi(self, state: np.ndarray):
        if self.is_random_step():
            return self.agent_env.action_space.sample()
        state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
        if not self.sample_actions:
            return self.act_greedy(state_t)
        with torch.no_grad():
            logits = self.policy_net.actor_logits(
                self.policy_net.features(state_t))
        return int(torch.distributions.Categorical(
            logits=logits).sample().item())

    # ------------------------------------------------------------- learning
    def _augment(self, x, offsets=None, factors=None):
        x, offsets = random_shift(x, self.aug_pad, offsets=offsets)
        if self.aug_intensity > 0:
            x, factors = intensity(x, self.aug_intensity, factors=factors)
        return x, offsets, factors

    def _soft_target(self, returns, discounts, non_final_mask, next_states):
        """Exact-expectation n-step soft target:
        y = R_n + gamma^m * [terminal ? 0 : sum_a' pi(a'|s') (min Qbar - alpha log pi)]
        next_states holds only the non-final bootstrap states, in mask order.
        """
        v_soft = torch.zeros(len(returns), device=self.device)
        if next_states is not None and non_final_mask.any():
            alpha = self.log_alpha.exp()
            next_logits = self.policy_net.actor_logits(
                self.policy_net.features(next_states))
            next_probs, next_log_probs = SACDiscreteNetwork.policy(next_logits)
            q1t, q2t = self.target_net.critic_values(
                self.target_net.features(next_states))
            qbar = torch.min(q1t, q2t)
            v_soft[non_final_mask] = (
                next_probs * (qbar - alpha * next_log_probs)).sum(dim=1)
        return returns + discounts * v_soft

    def _train_step(self):
        if self.prioritized_replay:
            (states, actions, returns, next_list, discounts, handles,
             weights) = self.replay_buffer.sample_nstep_prioritized(
                self.batch_size, self.n_step, self.loss.gamma)
            weights = weights.to(self.device)
        else:
            states, actions, returns, next_list, discounts, handles = \
                self.replay_buffer.sample_nstep_transitions(
                    self.batch_size, self.n_step, self.loss.gamma,
                    with_handles=True)
            weights = None
        states = states.to(self.device)
        actions = actions.to(self.device)
        returns = returns.to(self.device)
        discounts = discounts.to(self.device)
        non_final_mask = torch.tensor([s is not None for s in next_list],
                                      device=self.device)

        augment = self.use_augmentation and states.dim() == 4
        td_states = self._augment(states.float())[0] if augment else states

        with torch.no_grad():
            next_states = None
            if non_final_mask.any():
                next_states = torch.cat(
                    [s for s in next_list if s is not None]).to(self.device)
                if augment:
                    next_states = self._augment(next_states.float())[0]
            y = self._soft_target(returns, discounts, non_final_mask,
                                  next_states)                        # (B,)

        phi = self.policy_net.features(td_states)                     # (B, F)
        q1, q2 = self.policy_net.critic_values(phi)
        q1_a = q1.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_a = q2.gather(1, actions.unsqueeze(1)).squeeze(1)
        l1 = (q1_a - y).pow(2)
        l2 = (q2_a - y).pow(2)
        per_sample = 0.5 * (l1 + l2)
        td_loss = (per_sample if weights is None
                   else weights * per_sample).mean()
        loss = td_loss

        # actor + temperature on the SAME phi, detached — the encoder belongs
        # to the critics; Q detached in the actor loss; probs/log-probs
        # detached in the alpha loss. All expectations exact.
        logits = self.policy_net.actor_logits(phi.detach())
        probs, log_probs = SACDiscreteNetwork.policy(logits)
        alpha_det = self.log_alpha.exp().detach()
        min_q = torch.min(q1, q2).detach()
        actor_loss = (probs * (alpha_det * log_probs - min_q)).sum(dim=1).mean()
        mean_log_prob = (probs.detach() * log_probs.detach()).sum(dim=1)
        alpha_loss = -(self.log_alpha
                       * (mean_log_prob + self.target_entropy)).mean()
        entropy = float(-(probs.detach()
                          * log_probs.detach()).sum(dim=1).mean())
        loss = loss + actor_loss + alpha_loss

        lam = self._lambda_eff()
        diag = {"td_loss": float(td_loss.detach()), "lambda_eff": lam,
                "penalty_raw": np.nan, "penalty_weighted": 0.0,
                "residual_rms": np.nan, "b_h": np.nan, "unique_eps": np.nan,
                "sum_c": float(self.c.detach().sum()),
                "companion_radius": self._companion_radius(),
                "rampdown_scale": self._rampdown_scale(),
                "rampdown_penalty_bar": (float("nan")
                                         if (bar := self._penalty_bar()) is None
                                         else bar),
                "actor_loss": float(actor_loss.detach()),
                "alpha_loss": float(alpha_loss.detach()),
                "alpha": float(alpha_det),
                "entropy": entropy,
                "nan_skips": self.nan_skips}
        if self.c_head is not None:
            diag["c_spread"] = np.nan
        for j in range(self.fhr_order):
            diag[f"c_{j + 1}"] = float(self.c[j].detach())
            if self.reward_lags:
                diag[f"d_{j + 1}"] = float(self.d[j].detach())
        if self.prioritized_replay:
            diag["per_beta"] = float(self.replay_buffer.beta)
            diag["is_weight_mean"] = float(weights.mean())

        if self.fhr_weight > 0:
            r = self.fhr_order
            keep = [i for i, (_, t) in enumerate(handles) if t >= r]
            diag["b_h"] = float(len(keep))
            diag["unique_eps"] = float(len({handles[i][0] for i in keep}))
            if keep:
                p_states, p_actions, p_rewards = \
                    self.replay_buffer.gather_predecessors(
                        [handles[i] for i in keep], r)
                p_states = p_states.to(self.device)
                p_actions = p_actions.to(self.device)
                p_rewards = p_rewards.to(self.device)
                n = len(keep)
                with torch.enable_grad() if lam > 0 else torch.no_grad():
                    # Fused forward over each anchor and its r lags with ONE
                    # shared shift/intensity draw per sequence (the
                    # EfficientRainbow discipline: augmentation noise never
                    # enters the recurrence residual); both critic heads read
                    # the same feature pass.
                    seq = torch.cat(
                        [states[keep].unsqueeze(1), p_states], dim=1)
                    flat = seq.reshape(n * (r + 1), *seq.shape[2:])
                    if augment:
                        offs = torch.randint(
                            0, 2 * self.aug_pad + 1, (n, 2),
                            device=self.device).repeat_interleave(r + 1, dim=0)
                        flat, _ = random_shift(flat.float(), self.aug_pad,
                                               offsets=offs)
                        if self.aug_intensity > 0:
                            fac = (1.0 + self.aug_intensity * torch.randn(
                                n, 1, 1, 1,
                                device=self.device).clamp_(-2.0, 2.0)
                                ).repeat_interleave(r + 1, dim=0)
                            flat, _ = intensity(flat, self.aug_intensity,
                                                factors=fac)
                    # the encoder owns input normalisation (uint8 or 0-255
                    # float alike), exactly as in EfficientRainbowAgent
                    phi_seq = self.policy_net.features(flat)
                    a_seq = torch.cat(
                        [actions[keep].view(n, 1), p_actions], dim=1)
                    idx = a_seq.reshape(n * (r + 1), 1)
                    if self.c_head is not None:
                        phi_anchor = phi_seq.view(
                            n, r + 1, -1)[:, 0]
                        c_pred, d_pred = self.c_head(phi_anchor,
                                                     actions[keep])
                        c_mean = c_pred.detach().mean(dim=0)
                        diag["sum_c"] = float(c_mean.sum())
                        diag["c_spread"] = float(
                            c_pred.detach().std(dim=0).mean())
                        for j in range(r):
                            diag[f"c_{j + 1}"] = float(c_mean[j])
                            if self.reward_lags:
                                diag[f"d_{j + 1}"] = float(
                                    d_pred.detach().mean(dim=0)[j])
                        roots = np.roots(np.concatenate(
                            ([1.0], -c_mean.cpu().numpy())))
                        diag["companion_radius"] = (
                            float(np.abs(roots).max()) if roots.size else 0.0)
                    hubers, sq_res = [], []
                    for q_head in (self.policy_net.q1, self.policy_net.q2):
                        q_seq = q_head(phi_seq).gather(1, idx).view(n, r + 1)
                        anchor = q_seq[:, 0]
                        q_lags = q_seq[:, 1:]
                        if self.c_head is not None:
                            prediction = (q_lags * c_pred).sum(dim=1)
                            if self.reward_lags:
                                prediction = prediction \
                                    + (p_rewards * d_pred).sum(dim=1)
                        else:
                            prediction = q_lags @ self.c
                            if self.reward_lags:
                                prediction = prediction + p_rewards @ self.d
                        hubers.append(F.huber_loss(anchor, prediction))
                        sq_res.append((anchor.detach()
                                       - prediction.detach()).pow(2))
                    penalty = 0.5 * (hubers[0] + hubers[1])
                diag["penalty_raw"] = float(penalty.detach())
                diag["penalty_weighted"] = lam * diag["penalty_raw"]
                if lam > 0:
                    self._ep_penalty_vals.append(diag["penalty_raw"])
                diag["residual_rms"] = float(
                    torch.cat(sq_res).mean().sqrt())
                if lam > 0:
                    loss = loss + lam * penalty

        self._grad_steps += 1
        if not torch.isfinite(loss):
            self.nan_skips += 1
            diag["nan_skips"] = self.nan_skips
            return diag
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.grad_clip_norm)
        if not torch.isfinite(grad_norm):
            self.nan_skips += 1
            diag["nan_skips"] = self.nan_skips
            self.optimiser.zero_grad()
            return diag
        self.optimiser.step()
        if self.prioritized_replay:
            with torch.no_grad():
                td_err = 0.5 * ((q1_a - y).abs() + (q2_a - y).abs())
            self.replay_buffer.update_priorities(
                handles, td_err.cpu().numpy())
        return diag

    # -- persistence: FHR payload + temperature (+ ccond head) --------------
    def save(self, path):
        payload = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            "epsilon": self.epsilon,
            "fhr_c": self.c.detach().cpu(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }
        if self.d is not None:
            payload["fhr_d"] = self.d.detach().cpu()
        if self.c_head is not None:
            payload["c_head"] = self.c_head.state_dict()
        torch.save(payload, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimiser.load_state_dict(ckpt["optimiser"])
        self.epsilon = ckpt["epsilon"]
        with torch.no_grad():
            self.c.copy_(ckpt["fhr_c"].to(self.device))
            if self.d is not None and "fhr_d" in ckpt:
                self.d.copy_(ckpt["fhr_d"].to(self.device))
            self.log_alpha.copy_(ckpt["log_alpha"].to(self.device))
        if self.c_head is not None and "c_head" in ckpt:
            self.c_head.load_state_dict(ckpt["c_head"])
