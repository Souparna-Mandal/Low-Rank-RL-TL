"""FHR (learned-linear-recurrence Hankel-rank) regulariser for the SAC family
on Stable-Baselines3, both action-space regimes:

  * FHRSAC  — stable_baselines3.SAC (continuous, tanh-squashed Gaussian).
  * SACD    — SAC-Discrete (Christodoulou 2019) written SB3-style: categorical
              actor, twin critics emitting full Q(s, .) rows, every expectation
              over actions computed EXACTLY (no sampling, no reparam), and the
              temperature auto-tuned toward target_entropy_scale * log|A|.
  * FHRSACD — SACD + the FHR penalty.

Where FHR enters SAC: the penalty rides the CRITIC loss (whose target is the
soft/max-entropy target y = r + gamma * (min Qbar - alpha log pi)), applied to
BOTH critics with ONE shared c (or one shared c(s,a) predictor):

    penalty = 0.5 * sum_i Huber(Q_i(s_t,a_t),
                                sum_j c_j Q_i(s_{t-j},a_{t-j}) [+ sum_k d_k r_{t-k}])

The actor loss — the KL(pi || exp(Q/alpha)/Z) improvement step — is left
untouched: it consumes the FHR-shaped critics implicitly through min Q, so the
regularised values steer the policy with no direct penalty term of their own.

The continuous-critic hooks (raw-action encoding, float lag/anchor actions,
ContinuousCritic penultimate features, fused twin lag_q closures) live in
_FHRContinuousCriticMixin, shared with FHRTD3 (agents/sb3_td3_fhr.py) — the
two hosts differ only in their train() loops and in how the per-critic Huber
terms combine (SAC "mean" against its 0.5 * sum_i MSE_i TD term; TD3 "sum").
grad_probe_every > 0 adds the gradient-stream probe of agents/sb3_fhr.py.

Design contracts carried over from agents/sb3_fhr.py:
  * fhr_weight=0 with uniform replay reproduces stock SB3 SAC bit-for-bit
    (same RNG streams, same updates); everything non-stock is gated behind
    `if self.fhr_weight > 0` / `if self.prioritized_replay`. SACD has no stock
    counterpart — there its lambda=0 arm IS the baseline.
  * c/d (or the predictor) live in their own param group of the CRITIC
    optimizer (the penalty is a critic-side objective); SAC has no gradient
    clipping and none is added.
  * The coefficient predictor for continuous actions conditions on the raw
    (already tanh-squashed, buffer-scale [-1, 1]) action vector instead of a
    one-hot; "shared" mode reads the online critic's penultimate activations
    (whose input already contains the action).

Prioritized replay (PER) — absent from SB3 2.9.0 entirely — is implemented
here as FHRPrioritizedEpisodicReplayBuffer (proportional priorities on a sum
tree, rainbow_agent.py conventions) and is opt-in per run/arm via
`prioritized_replay: true`: IS weights multiply the per-sample critic TD terms
only (the FHR penalty stays an unweighted mean — it is a prior, not a
per-sample TD), and priorities are refreshed with the mean twin |TD error|.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import SAC
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BaseModel, BasePolicy
from stable_baselines3.common.torch_layers import (FlattenExtractor,
                                                   create_mlp,
                                                   get_actor_critic_arch)
from stable_baselines3.common.utils import (get_parameters_by_name,
                                            polyak_update)

from agents.sb3_fhr import (_FHRMixin, FHREpisodicReplayBuffer,
                            penultimate_features)


class FHRPrioritizedEpisodicReplayBuffer(FHREpisodicReplayBuffer):
    """FHREpisodicReplayBuffer + proportional prioritized sampling.

    A flat sum tree (2 * buffer_size - 1 float64 array) keyed by ring slot:
    leaf i <-> slot i, so `last_batch_inds` stays slot-valued and the FHR
    predecessor path needs zero changes. Conventions follow the repo's
    rainbow_agent.PrioritizedReplayBuffer: new transitions enter at the
    running max priority, sampling is stratified over equal-mass segments,
    IS weights are normalised by their max (they only scale the loss down),
    priorities are |td| + eps raised to per_alpha. The beta exponent is a
    plain attribute (`per_beta`) the training loop anneals from progress.
    Unwritten slots hold priority zero and are never drawn; a ring overwrite
    resets that slot to the current max priority via add().
    """

    def __init__(self, *args, per_alpha: float = 0.6, per_eps: float = 1e-5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.per_alpha = per_alpha
        self.per_eps = per_eps
        self.per_beta = 0.4                  # train() overwrites per burst
        self.max_priority = 1.0
        self.tree = np.zeros(2 * self.buffer_size - 1, dtype=np.float64)
        self.last_batch_weights: np.ndarray | None = None

    # -- sum tree over ring slots ------------------------------------------
    def _tree_update(self, slot: int, priority: float) -> None:
        idx = slot + self.buffer_size - 1
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def _tree_get(self, s: float) -> int:
        idx = 0
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):
                return idx - (self.buffer_size - 1)
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1

    # -- buffer API ---------------------------------------------------------
    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        slot = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        self._tree_update(slot, self.max_priority ** self.per_alpha)

    def sample(self, batch_size: int, env=None):
        total = float(self.tree[0])
        size = self.buffer_size if self.full else self.pos
        slots = np.empty(batch_size, dtype=np.int64)
        segment = total / batch_size
        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            slot = self._tree_get(s)
            if self.tree[slot + self.buffer_size - 1] <= 0.0:
                # float roundoff can land on an unwritten leaf; redraw
                slot = int(np.random.randint(0, size))
            slots[i] = slot
        self.last_batch_inds = slots
        probs = self.tree[slots + self.buffer_size - 1] / total
        weights = (size * probs) ** (-self.per_beta)
        weights = weights / weights.max()
        self.last_batch_weights = weights.astype(np.float32)
        return self._get_samples(slots, env=env)

    def update_priorities(self, slots: np.ndarray, td_errors: np.ndarray) -> None:
        prios = np.abs(np.asarray(td_errors, dtype=np.float64)) + self.per_eps
        self.max_priority = max(self.max_priority, float(prios.max()))
        for slot, p in zip(np.asarray(slots, dtype=np.int64), prios):
            self._tree_update(int(slot), float(p) ** self.per_alpha)


class _FHRSACFamilyMixin(_FHRMixin):
    """The SAC-side specialisations of the FHR mixin hooks shared by FHRSAC
    and FHRSACD: the coefficient group lives on the CRITIC optimizer, the
    penalty reads self.critic, PER is supported, and the ccond predictor's
    construction is RNG-neutral (the SAC family consumes torch RNG on every
    env step, so extra init draws would desync a lambda=0 ccond run)."""

    _fhr_supports_per = True

    def _fhr_coeff_optimizer(self):
        return self.critic.optimizer

    def _fhr_online_qnet(self):
        return self.critic

    def _fhr_build_predictor(self, in_dim: int):
        devices = (list(range(torch.cuda.device_count()))
                   if torch.cuda.is_available() else [])
        with torch.random.fork_rng(devices=devices):
            return super()._fhr_build_predictor(in_dim)

    def _fhr_setup_checks(self):
        if getattr(self.policy, "share_features_extractor", False):
            raise ValueError(
                "the FHR SAC family requires share_features_extractor=False "
                "(the default): a shared extractor would route the critic-side "
                "FHR penalty gradient into the actor trunk")
        if self.prioritized_replay != isinstance(
                self.replay_buffer, FHRPrioritizedEpisodicReplayBuffer):
            raise ValueError(
                f"prioritized_replay={self.prioritized_replay} but the replay "
                f"buffer is {type(self.replay_buffer).__name__}")

    def _fhr_per_kwargs(self, kwargs: dict) -> dict:
        """Route the PER buffer class + alpha into the SB3 constructor kwargs
        (call before super().__init__)."""
        if self.prioritized_replay:
            kwargs.setdefault("replay_buffer_class",
                              FHRPrioritizedEpisodicReplayBuffer)
            if issubclass(kwargs["replay_buffer_class"],
                          FHRPrioritizedEpisodicReplayBuffer):
                rb = dict(kwargs.get("replay_buffer_kwargs") or {})
                rb.setdefault("per_alpha", self.per_alpha)
                kwargs["replay_buffer_kwargs"] = rb
            # a mismatching explicit class is caught by _fhr_setup_checks
        else:
            kwargs.setdefault("replay_buffer_class", FHREpisodicReplayBuffer)
        return kwargs

    def _fhr_anneal_per_beta(self):
        """Linear IS-exponent anneal beta0 -> 1 over the training budget."""
        self.replay_buffer.per_beta = (
            self.per_beta0 + (1.0 - self.per_beta0)
            * (1.0 - self._current_progress_remaining))


class _FHRContinuousCriticMixin(_FHRSACFamilyMixin):
    """Continuous-action (Box) hooks shared by the twin-ContinuousCritic
    hosts — FHRSAC here and FHRTD3 (agents/sb3_td3_fhr.py): raw-action
    encoding for the c(s,a) predictor, float lag/anchor actions (the buffer
    stores the policy's [-1, 1]-scaled actions), the critic's penultimate
    features for the shared predictor, and per-critic lag_q closures with one
    fused feature/head forward per distinct (obs, acts) pair."""

    def _setup_model(self) -> None:
        super()._setup_model()
        self._fhr_setup_checks()

    def _fhr_predictor_action_repr(self):
        return int(np.prod(self.action_space.shape)), "raw"

    def _fhr_lag_actions(self, raw_actions, n: int, r: int):
        # buffer actions are float, already scaled to [-1, 1] by the policy
        return self.replay_buffer.to_torch(
            raw_actions).float().reshape(n * r, -1)

    def _fhr_anchor_actions(self, raw_actions, n: int):
        return self.replay_buffer.to_torch(raw_actions).float().reshape(n, -1)

    def _fhr_shared_features(self, anchor_obs, anchor_acts):
        return penultimate_features(self.critic, anchor_obs, anchor_acts)

    def _lag_q_fns(self):
        """Per-critic lag_q closures sharing one feature/head forward per
        distinct (obs, acts) pair, so the twin-critic penalty costs one fused
        evaluation instead of two."""
        critic = self.critic
        cache: dict = {}

        def make(i):
            def lag_q(obs, acts):
                key = (id(obs), id(acts))
                if key not in cache:
                    feats = critic.extract_features(
                        obs, critic.features_extractor)
                    qin = torch.cat([feats, acts], dim=1)
                    cache[key] = [q_net(qin) for q_net in critic.q_networks]
                return cache[key][i].squeeze(1)
            return lag_q
        return [make(i) for i in range(len(critic.q_networks))]

_FHR_KWARG_DOC = """fhr_weight and friends match FHRDQN (agents/sb3_fhr.py);
    prioritized_replay/per_alpha/per_beta0 opt into the PER buffer."""


class FHRSAC(_FHRContinuousCriticMixin, SAC):
    """stable_baselines3 SAC + the FHR recurrence penalty on both critics.
    fhr_weight=0 with uniform replay is bit-for-bit stock SAC."""

    def __init__(self, *args, fhr_weight: float = 0.0, fhr_order: int = 2,
                 reward_lags: bool = False, warmup_grad_steps: int = 2000,
                 c_learning_rate: float = 5e-3,
                 rampdown_reward_threshold: float | None = None,
                 rampdown_penalty_threshold: float | str | None = None,
                 rampdown_penalty_topk: int = 20,
                 rampdown_patience_eps: int = 10,
                 rampdown_episodes: int = 0, c_predictor: str = "none",
                 prioritized_replay: bool = False, per_alpha: float = 0.6,
                 per_beta0: float = 0.4, window_rank_every: int = 0,
                 window_rank_lags: int = 16, c_init=None,
                 grad_probe_every: int = 0, **kwargs):
        self._set_fhr_config(fhr_weight, fhr_order, reward_lags,
                             warmup_grad_steps, c_learning_rate,
                             rampdown_reward_threshold,
                             rampdown_penalty_threshold,
                             rampdown_penalty_topk, rampdown_patience_eps,
                             rampdown_episodes, c_predictor=c_predictor,
                             prioritized_replay=prioritized_replay,
                             per_alpha=per_alpha, per_beta0=per_beta0,
                             window_rank_every=window_rank_every,
                             window_rank_lags=window_rank_lags, c_init=c_init,
                             grad_probe_every=grad_probe_every)
        kwargs = self._fhr_per_kwargs(kwargs)
        super().__init__(*args, **kwargs)

    def qagent_adapter(self, epsilon: float | None = None):
        return SB3SACAdapter(self, epsilon=epsilon)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # SAC.train (sb3 2.9.0) verbatim, with the FHR penalty riding the
        # critic loss, the FHRDQNAgent NaN guard, and the optional PER
        # weighting/priority feedback. Everything non-stock is gated on
        # `self.fhr_weight > 0` / `self.prioritized_replay`, so the lambda=0
        # uniform run consumes the stock instruction and RNG streams exactly.
        if self._vec_normalize_env is not None:
            raise RuntimeError("FHRSAC does not support VecNormalize — lag "
                               "observations would bypass the normalisation")
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        if self.prioritized_replay:
            self._fhr_anneal_per_beta()

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        step_diags = []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            inds = self.replay_buffer.last_batch_inds
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                ent_coef = torch.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called entropy temperature
            # or alpha in the paper (stock order: BEFORE the critic step)
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with torch.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            if self.prioritized_replay:
                w = self.replay_buffer.to_torch(
                    self.replay_buffer.last_batch_weights).reshape(-1, 1)
                critic_loss = 0.5 * sum(
                    ((current_q - target_q_values).pow(2) * w).mean()
                    for current_q in current_q_values)
            else:
                critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, torch.Tensor)
            critic_losses.append(critic_loss.item())

            lam = self._lambda_eff()
            diag = self._fhr_base_diag(td_loss=critic_losses[-1], lam=lam)
            diag["ent_coef"] = ent_coefs[-1]
            diag["actor_loss"] = np.nan       # filled after the actor step
            probe = self._fhr_grad_probe_due()
            if self.fhr_weight > 0 or probe:
                anchors = [q.squeeze(1) for q in current_q_values]
                penalty = self._fhr_penalty_multi(
                    anchors, self._lag_q_fns(), lam, diag, need_grad=probe)
                if probe and penalty is not None:
                    # on the pure TD term, before the penalty joins the loss
                    self._fhr_grad_probe(critic_loss, penalty, lam, diag)
                if self.fhr_weight > 0 and penalty is not None and lam > 0:
                    critic_loss = critic_loss + lam * penalty
            if self._fhr_window_rank_due():
                self._fhr_window_rank_probe(self._lag_q_fns())
            step_diags.append(diag)
            self._fhr_grad_steps += 1

            if self.fhr_weight > 0 and not torch.isfinite(critic_loss):
                # FHRDQNAgent whole-step skip (never reachable at lambda=0:
                # the guard itself is gated on fhr_weight > 0). The ent-coef
                # step already ran, as in the stock ordering.
                self.nan_skips += 1
                diag["nan_skips"] = self.nan_skips
                continue

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self.prioritized_replay:
                with torch.no_grad():
                    td = torch.stack([(q - target_q_values).abs()
                                      for q in current_q_values]).mean(dim=0)
                self.replay_buffer.update_priorities(
                    inds, td.squeeze(1).cpu().numpy())

            # Compute actor loss
            # Min over all critic networks
            q_values_pi = torch.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())
            diag["actor_loss"] = actor_losses[-1]

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self._fhr_aggregate_pending(step_diags)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss",
                           np.mean(actor_losses) if actor_losses else np.nan)
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


# ---------------------------------------------------------------------------
# SAC-Discrete (Christodoulou 2019), SB3-style
# ---------------------------------------------------------------------------

class CategoricalActor(BasePolicy):
    """Categorical policy head: logits MLP on the extracted features.
    action_probs() returns (probs, log_probs) via log_softmax — never
    log(probs) — so the exact-expectation losses stay finite even when a
    probability underflows."""

    def __init__(self, observation_space: spaces.Space,
                 action_space: spaces.Discrete, net_arch: list[int],
                 features_extractor: nn.Module, features_dim: int,
                 activation_fn: type[nn.Module] = nn.ReLU,
                 normalize_images: bool = True):
        super().__init__(observation_space, action_space,
                         features_extractor=features_extractor,
                         normalize_images=normalize_images,
                         squash_output=False)
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.logits_net = nn.Sequential(*create_mlp(
            features_dim, int(action_space.n), net_arch, activation_fn))

    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        return self.logits_net(features)

    def action_probs(self, obs: torch.Tensor):
        log_probs = F.log_softmax(self.logits(obs), dim=1)
        return log_probs.exp(), log_probs

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        return self._predict(obs, deterministic)

    def _predict(self, observation: torch.Tensor,
                 deterministic: bool = False) -> torch.Tensor:
        logits = self.logits(observation)
        if deterministic:
            return logits.argmax(dim=1)
        return torch.distributions.Categorical(logits=logits).sample()

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data.update(dict(net_arch=self.net_arch,
                         features_dim=self.features_dim,
                         activation_fn=self.activation_fn,
                         features_extractor=self.features_extractor))
        return data


class DiscreteCritic(BaseModel):
    """n_critics Q heads, each emitting the full (B, |A|) row — the discrete
    analogue of ContinuousCritic. Heads are collected in self.q_networks (the
    attribute name the shared-ccond feature helper keys on)."""

    def __init__(self, observation_space: spaces.Space,
                 action_space: spaces.Discrete, net_arch: list[int],
                 features_extractor: nn.Module, features_dim: int,
                 activation_fn: type[nn.Module] = nn.ReLU,
                 normalize_images: bool = True, n_critics: int = 2):
        super().__init__(observation_space, action_space,
                         features_extractor=features_extractor,
                         normalize_images=normalize_images)
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.n_critics = n_critics
        self.q_networks: list[nn.Module] = []
        for idx in range(n_critics):
            q_net = nn.Sequential(*create_mlp(
                features_dim, int(action_space.n), net_arch, activation_fn))
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: torch.Tensor) -> tuple:
        features = self.extract_features(obs, self.features_extractor)
        return tuple(q_net(features) for q_net in self.q_networks)

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data.update(dict(net_arch=self.net_arch,
                         features_dim=self.features_dim,
                         activation_fn=self.activation_fn,
                         n_critics=self.n_critics,
                         features_extractor=self.features_extractor))
        return data


class SACDPolicy(BasePolicy):
    """Actor + twin discrete critics + target critics, mirroring SACPolicy's
    _build: actor and critic each get their own optimizer, the target critic
    starts as a copy of the critic and stays in eval mode. Each module owns
    its own features extractor (no share_features_extractor support — with the
    default parameter-free FlattenExtractor sharing would buy nothing and a
    shared trunk would entangle the FHR penalty with the actor)."""

    def __init__(self, observation_space: spaces.Space,
                 action_space: spaces.Discrete, lr_schedule,
                 net_arch: list[int] | dict | None = None,
                 activation_fn: type[nn.Module] = nn.ReLU,
                 features_extractor_class=FlattenExtractor,
                 features_extractor_kwargs: dict | None = None,
                 normalize_images: bool = True,
                 optimizer_class=torch.optim.Adam,
                 optimizer_kwargs: dict | None = None,
                 n_critics: int = 2):
        super().__init__(observation_space, action_space,
                         features_extractor_class, features_extractor_kwargs,
                         optimizer_class=optimizer_class,
                         optimizer_kwargs=optimizer_kwargs,
                         squash_output=False,
                         normalize_images=normalize_images)
        if net_arch is None:
            net_arch = [256, 256]
        actor_arch, critic_arch = get_actor_critic_arch(net_arch)
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update({"n_critics": n_critics,
                                   "net_arch": critic_arch})
        self._build(lr_schedule)

    def _build(self, lr_schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(), lr=lr_schedule(1),
            **self.optimizer_kwargs)
        self.critic = self.make_critic()
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(), lr=lr_schedule(1),
            **self.optimizer_kwargs)
        self.critic_target = self.make_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.set_training_mode(False)

    def make_actor(self) -> CategoricalActor:
        kwargs = self._update_features_extractor(self.actor_kwargs, None)
        return CategoricalActor(**kwargs).to(self.device)

    def make_critic(self) -> DiscreteCritic:
        kwargs = self._update_features_extractor(self.critic_kwargs, None)
        return DiscreteCritic(**kwargs).to(self.device)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: torch.Tensor,
                 deterministic: bool = False) -> torch.Tensor:
        return self.actor._predict(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode

    def _get_constructor_parameters(self) -> dict:
        data = super()._get_constructor_parameters()
        data.update(dict(
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            n_critics=self.critic_kwargs["n_critics"],
            lr_schedule=self._dummy_schedule,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            features_extractor_class=self.features_extractor_class,
            features_extractor_kwargs=self.features_extractor_kwargs,
        ))
        return data


class SACD(OffPolicyAlgorithm):
    """SAC-Discrete (Christodoulou 2019, arXiv:1910.07207) in SB3 form.

    Same update skeleton and ordering as stable_baselines3.SAC (ent-coef step
    first, then critic, then actor, then interval-keyed polyak), but every
    expectation over actions is computed EXACTLY from the categorical policy:

        y            = r + gamma * (1-d) * sum_a' pi(a'|s') [min_i Qbar_i(s',a')
                                                             - alpha log pi(a'|s')]
        critic loss  = 0.5 * sum_i MSE(Q_i(s_t, a_t), y)          (gather)
        actor loss   = E_s sum_a pi(a|s) (alpha log pi(a|s) - min_i Q_i(s,a))
        alpha loss   = -log_alpha * (E_pi[log pi] + target_entropy).detach()
        target_entropy = target_entropy_scale * log|A|   (default scale 0.98)

    train() consumes no torch RNG (action sampling happens only during
    rollouts), which is what lets the FHR subclass keep lambda=0 exact.
    """

    policy_aliases = {"MlpPolicy": SACDPolicy}

    def __init__(self, policy, env, learning_rate: float = 3e-4,
                 buffer_size: int = 1_000_000, learning_starts: int = 100,
                 batch_size: int = 256, tau: float = 0.005,
                 gamma: float = 0.99, train_freq=1, gradient_steps: int = 1,
                 replay_buffer_class=None, replay_buffer_kwargs=None,
                 optimize_memory_usage: bool = False, n_steps: int = 1,
                 ent_coef="auto", target_update_interval: int = 1,
                 target_entropy="auto", target_entropy_scale: float = 0.98,
                 stats_window_size: int = 100, tensorboard_log=None,
                 policy_kwargs=None, verbose: int = 0, seed=None,
                 device="auto", _init_setup_model: bool = True):
        super().__init__(
            policy, env, learning_rate, buffer_size, learning_starts,
            batch_size, tau, gamma, train_freq, gradient_steps,
            action_noise=None, replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage, n_steps=n_steps,
            policy_kwargs=policy_kwargs, stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log, verbose=verbose, device=device,
            seed=seed, sde_support=False,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=False)
        self.target_entropy = target_entropy
        self.target_entropy_scale = target_entropy_scale
        self.log_ent_coef = None
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        if self.target_entropy == "auto":
            # fraction of the uniform-policy entropy (Christodoulou 2019)
            self.target_entropy = float(
                self.target_entropy_scale * np.log(self.action_space.n))
        else:
            self.target_entropy = float(self.target_entropy)
        # ent_coef machinery verbatim from SAC._setup_model
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"
            self.log_ent_coef = torch.log(
                torch.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam(
                [self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            self.ent_coef_tensor = torch.tensor(
                float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def qagent_adapter(self, epsilon: float | None = None):
        return SB3SACDAdapter(self, epsilon=epsilon)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            # pi(.|s) for the sampled states — reused by the ent-coef and
            # actor losses; every expectation below is exact, no sampling
            probs, log_probs = self.actor.action_probs(replay_data.observations)
            # E_pi[log pi] = -H(pi(.|s)): SAC's sampled log_prob replaced by
            # its exact expectation
            mean_log_prob = (probs * log_probs).sum(dim=1, keepdim=True)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (mean_log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with torch.no_grad():
                next_probs, next_log_probs = self.actor.action_probs(replay_data.next_observations)
                next_q, _ = torch.min(torch.stack(
                    self.critic_target(replay_data.next_observations)), dim=0)
                # soft state value under the current policy, exactly
                next_v = (next_probs * (next_q - ent_coef * next_log_probs)).sum(dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_v

            current_q_values = tuple(
                q.gather(1, replay_data.actions.long())
                for q in self.critic(replay_data.observations))
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values)
                                    for current_q in current_q_values)
            assert isinstance(critic_loss, torch.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # policy improvement — KL(pi || exp(Q/alpha)/Z) up to a constant,
            # computed exactly; the critics are re-evaluated post-step as SAC
            # does, and their stray gradients are cleared by the next
            # critic zero_grad (the stock SAC pattern)
            min_q_pi, _ = torch.min(torch.stack(
                self.critic(replay_data.observations)), dim=0)
            actor_loss = (probs * (ent_coef * log_probs - min_q_pi)).sum(dim=1).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self):
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables


class FHRSACD(_FHRSACFamilyMixin, SACD):
    """SACD + the FHR recurrence penalty on both critics. lambda=0 with
    uniform replay reproduces plain SACD bit-for-bit (same streams)."""

    def __init__(self, *args, fhr_weight: float = 0.0, fhr_order: int = 2,
                 reward_lags: bool = False, warmup_grad_steps: int = 2000,
                 c_learning_rate: float = 5e-3,
                 rampdown_reward_threshold: float | None = None,
                 rampdown_penalty_threshold: float | str | None = None,
                 rampdown_penalty_topk: int = 20,
                 rampdown_patience_eps: int = 10,
                 rampdown_episodes: int = 0, c_predictor: str = "none",
                 prioritized_replay: bool = False, per_alpha: float = 0.6,
                 per_beta0: float = 0.4, window_rank_every: int = 0,
                 window_rank_lags: int = 16, c_init=None,
                 grad_probe_every: int = 0, **kwargs):
        self._set_fhr_config(fhr_weight, fhr_order, reward_lags,
                             warmup_grad_steps, c_learning_rate,
                             rampdown_reward_threshold,
                             rampdown_penalty_threshold,
                             rampdown_penalty_topk, rampdown_patience_eps,
                             rampdown_episodes, c_predictor=c_predictor,
                             prioritized_replay=prioritized_replay,
                             per_alpha=per_alpha, per_beta0=per_beta0,
                             window_rank_every=window_rank_every,
                             window_rank_lags=window_rank_lags, c_init=c_init,
                             grad_probe_every=grad_probe_every)
        kwargs = self._fhr_per_kwargs(kwargs)
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        self._fhr_setup_checks()

    def _lag_q_fns(self):
        """Per-critic lag_q closures sharing one critic forward per obs."""
        cache: dict = {}

        def make(i):
            def lag_q(obs, acts):
                key = id(obs)
                if key not in cache:
                    cache[key] = self.critic(obs)
                return cache[key][i].gather(1, acts).squeeze(1)
            return lag_q
        return [make(i) for i in range(len(self.critic.q_networks))]

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # SACD.train verbatim, with the FHR penalty riding the critic loss,
        # the FHRDQNAgent NaN guard, and the optional PER weighting/priority
        # feedback — the exact splice pattern of FHRSAC.train.
        if self._vec_normalize_env is not None:
            raise RuntimeError("FHRSACD does not support VecNormalize — lag "
                               "observations would bypass the normalisation")
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        if self.prioritized_replay:
            self._fhr_anneal_per_beta()

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        step_diags = []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            inds = self.replay_buffer.last_batch_inds
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            probs, log_probs = self.actor.action_probs(replay_data.observations)
            mean_log_prob = (probs * log_probs).sum(dim=1, keepdim=True)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (mean_log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with torch.no_grad():
                next_probs, next_log_probs = self.actor.action_probs(replay_data.next_observations)
                next_q, _ = torch.min(torch.stack(
                    self.critic_target(replay_data.next_observations)), dim=0)
                next_v = (next_probs * (next_q - ent_coef * next_log_probs)).sum(dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_v

            current_q_values = tuple(
                q.gather(1, replay_data.actions.long())
                for q in self.critic(replay_data.observations))
            if self.prioritized_replay:
                w = self.replay_buffer.to_torch(
                    self.replay_buffer.last_batch_weights).reshape(-1, 1)
                critic_loss = 0.5 * sum(
                    ((current_q - target_q_values).pow(2) * w).mean()
                    for current_q in current_q_values)
            else:
                critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values)
                                        for current_q in current_q_values)
            assert isinstance(critic_loss, torch.Tensor)
            critic_losses.append(critic_loss.item())

            lam = self._lambda_eff()
            diag = self._fhr_base_diag(td_loss=critic_losses[-1], lam=lam)
            diag["ent_coef"] = ent_coefs[-1]
            diag["actor_loss"] = np.nan
            probe = self._fhr_grad_probe_due()
            if self.fhr_weight > 0 or probe:
                anchors = [q.squeeze(1) for q in current_q_values]
                penalty = self._fhr_penalty_multi(
                    anchors, self._lag_q_fns(), lam, diag, need_grad=probe)
                if probe and penalty is not None:
                    # on the pure TD term, before the penalty joins the loss
                    self._fhr_grad_probe(critic_loss, penalty, lam, diag)
                if self.fhr_weight > 0 and penalty is not None and lam > 0:
                    critic_loss = critic_loss + lam * penalty
            if self._fhr_window_rank_due():
                self._fhr_window_rank_probe(self._lag_q_fns())
            step_diags.append(diag)
            self._fhr_grad_steps += 1

            if self.fhr_weight > 0 and not torch.isfinite(critic_loss):
                self.nan_skips += 1
                diag["nan_skips"] = self.nan_skips
                continue

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self.prioritized_replay:
                with torch.no_grad():
                    td = torch.stack([(q - target_q_values).abs()
                                      for q in current_q_values]).mean(dim=0)
                self.replay_buffer.update_priorities(
                    inds, td.squeeze(1).cpu().numpy())

            min_q_pi, _ = torch.min(torch.stack(
                self.critic(replay_data.observations)), dim=0)
            actor_loss = (probs * (ent_coef * log_probs - min_q_pi)).sum(dim=1).mean()
            actor_losses.append(actor_loss.item())
            diag["actor_loss"] = actor_losses[-1]

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self._fhr_aggregate_pending(step_diags)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss",
                           np.mean(actor_losses) if actor_losses else np.nan)
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


# ---------------------------------------------------------------------------
# Analysis adapters
# ---------------------------------------------------------------------------

class MinTwinQNet(nn.Module):
    """(B, n_actions) = elementwise min over the discrete critic heads — the
    value surface the actor optimizes and the one FHR regularises, exposed on
    the repo's (B, n_actions) Q-row analysis contract."""

    def __init__(self, critic: DiscreteCritic):
        super().__init__()
        self.critic = critic

    def forward(self, x):
        qs = self.critic(x)
        out = qs[0]
        for q in qs[1:]:
            out = torch.min(out, q)
        return out


class SB3SACDAdapter:
    """The QAgent surface (device, policy_net, pi, act_greedy, save) over a
    SACD/FHRSACD model: policy_net = min-twin Q rows, greedy = argmax pi.

    epsilon=None samples the categorical policy in pi() (the behaviour
    policy, mirroring SB3QAgentAdapter's use of the live exploration rate);
    set adapter.epsilon = 0.0 for greedy post-training rollouts.
    """

    def __init__(self, model, epsilon: float | None = None):
        self.model = model
        self.epsilon = epsilon
        self.policy_net = MinTwinQNet(model.critic)

    @property
    def device(self):
        return self.model.device

    def act_greedy(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            return int(self.model.actor.logits(state).argmax(dim=1).item())

    def pi(self, state) -> int:
        if self.epsilon is not None and self.epsilon > 0 \
                and np.random.rand() < self.epsilon:
            return int(self.model.action_space.sample())
        state_t = torch.tensor(np.asarray(state), dtype=torch.float32,
                               device=self.device).unsqueeze(0)
        if self.epsilon is None:
            with torch.no_grad():
                logits = self.model.actor.logits(state_t)
            return int(torch.distributions.Categorical(
                logits=logits).sample().item())
        return self.act_greedy(state_t)

    def save(self, path):
        self.model.save(str(path))


class SB3SACAdapter:
    """Continuous-action analysis surface over FHRSAC (and, as
    SB3TD3Adapter, over FHRTD3 — same calls). The discrete
    (B, n_actions) policy_net contract cannot exist here — grid/row analyses
    are disabled for Box action spaces — but the rollout-Hankel analysis
    (analysis method `hankel_rollout_continuous`) uses:

      * pi(state) / act_greedy(state_t) -> np.ndarray env-scale action from
        the deterministic policy (mode of the squashed Gaussian; no RNG),
      * q_value(obs_t, act_t) -> (B,) min-twin soft-Q values.
    """

    policy_net = None

    def __init__(self, model, epsilon: float | None = None):
        self.model = model
        self.epsilon = epsilon      # accepted for API parity; unused

    @property
    def device(self):
        return self.model.device

    def pi(self, state) -> np.ndarray:
        action, _ = self.model.predict(np.asarray(state), deterministic=True)
        return action

    def act_greedy(self, state) -> np.ndarray:
        if isinstance(state, torch.Tensor):
            state = state.squeeze(0).detach().cpu().numpy()
        return self.pi(state)

    def q_value(self, obs_t: torch.Tensor, act_t: torch.Tensor) -> torch.Tensor:
        """min-twin Q(s, a) for env-scale actions (rescaled to the policy's
        [-1, 1] convention internally, matching what the buffer stores)."""
        with torch.no_grad():
            low = torch.as_tensor(self.model.action_space.low,
                                  device=obs_t.device)
            high = torch.as_tensor(self.model.action_space.high,
                                   device=obs_t.device)
            scaled = 2.0 * (act_t - low) / (high - low) - 1.0
            qs = torch.cat(self.model.critic(obs_t, scaled), dim=1)
            return torch.min(qs, dim=1).values

    def save(self, path):
        self.model.save(str(path))
