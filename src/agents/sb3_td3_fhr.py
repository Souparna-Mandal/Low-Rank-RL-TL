"""FHR (learned-linear-recurrence Hankel-rank) regulariser for TD3 on
Stable-Baselines3.

  * FHRTD3 — stable_baselines3.TD3 (twin ContinuousCritic, deterministic
    actor, clipped target-policy smoothing, delayed actor/polyak updates) +
    the FHR recurrence penalty on both critics.

Where FHR enters TD3 — the critic loss, on TD3's OWN scale. Stock TD3
regresses both critics on the plain Bellman target
y = r + gamma * min_i Qbar_i(s', pibar(s') + eps) with

    L_TD = sum_i MSE(Q_i(s,a), y)              (no 1/2, no mean over critics)

and the penalty joins on the same footing — one shared c (or one c(s,a)
predictor) across critics, no division by the number of critics anywhere:

    L_critic = sum_i MSE_i
             + lambda * sum_i Huber(Q_i(s_t,a_t),
                                    sum_j c_j Q_i(s_{t-j},a_{t-j}) [+ sum_k d_k r_{t-k}])

Per critic that is MSE_i + lambda * Huber_i, the single-critic reading of
FHRDQN. `penalty_raw` therefore logs sum_i Huber_i and `td_loss` the stock
sum_i MSE_i, so rho_loss = lambda * penalty_raw / td_loss is the ratio of the
two terms exactly as optimised (`_fhr_critic_reduction = "sum"`; the SAC
family keeps "mean" against its 0.5 * sum_i MSE_i). lambda is thus a knob on
the host algorithm's scale; the quantities that transfer between algorithms
are the stream ratios rho_loss (losses) and grad_rho / grad_ratio (gradients,
agents/sb3_fhr.py::_fhr_grad_probe), which is what the next family's lambda
is read from.

Why TD3 is the cleaner host for the FHR claim than SAC:
  * the target is the plain Bellman backup — no entropy bonus with a moving
    temperature inside the value the recurrence is fitted to — so the
    on-trajectory recurrence is exact up to the zero-mean smoothing noise, and
    the frozen-theory control c = (1 + 1/gamma, -1/gamma) is a genuine
    identity test rather than an approximation;
  * the actor is deterministic and trained on Q_1(s, pi(s)) alone every
    policy_delay critic steps, so the FHR-shaped Q_1 feeds straight into the
    deterministic policy gradient and the critic absorbs the penalty between
    actor steps — FHR as the critic-side stabiliser, on the algorithm whose
    known failure mode is critic instability;
  * exploration is a fixed Gaussian on actions (action_noise), not a learned
    distribution, so seed-to-seed variance is lower and every arm shares one
    exploration schedule. rewards.csv is the noisy behaviour policy's
    training stream; eval.csv is the deterministic actor.

Design contracts carried over from agents/sb3_sac_fhr.py:
  * fhr_weight=0 with uniform replay reproduces stock SB3 TD3 bit-for-bit:
    every non-stock line is gated on `fhr_weight > 0` / `prioritized_replay`
    / a probe tick, the FHR block consumes no torch/np RNG (the target-noise
    draw happens before it, exactly where stock draws it), and `_n_updates`
    is incremented at the loop top as in stock, so the policy_delay parity is
    untouched (a NaN-skipped step forfeits its delayed actor update).
  * c/d (or the predictor) live in their own param group of the CRITIC
    optimizer; the actor optimizer never sees an FHR parameter.
  * PER (FHRPrioritizedEpisodicReplayBuffer) weights the per-sample TD terms
    only; the penalty stays an unweighted mean over the batch.
"""
import numpy as np
import torch
import torch.nn.functional as F

from stable_baselines3 import TD3
from stable_baselines3.common.utils import polyak_update

from agents.sb3_sac_fhr import _FHRContinuousCriticMixin, SB3SACAdapter

__all__ = ["FHRTD3", "SB3TD3Adapter"]


class SB3TD3Adapter(SB3SACAdapter):
    """Continuous-action analysis surface over FHRTD3 — identical mechanics to
    SB3SACAdapter: pi / act_greedy are the deterministic actor
    (predict(deterministic=True) adds no exploration noise) and q_value is
    the min-twin Q(s, a) of the online critics for env-scale actions."""


class FHRTD3(_FHRContinuousCriticMixin, TD3):
    """stable_baselines3 TD3 + the FHR recurrence penalty on both critics.
    fhr_weight=0 with uniform replay is bit-for-bit stock TD3."""

    _fhr_critic_reduction = "sum"      # sum_i MSE_i + lambda * sum_i Huber_i

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
        return SB3TD3Adapter(self, epsilon=epsilon)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # TD3.train (sb3 2.9.0) verbatim, with the FHR penalty riding the
        # critic loss, the FHRDQNAgent NaN guard, the optional PER
        # weighting/priority feedback and the two probes. Everything non-stock
        # is gated on `self.fhr_weight > 0` / `self.prioritized_replay` / a
        # probe tick, so the lambda=0 uniform run consumes the stock
        # instruction and RNG streams exactly.
        if self._vec_normalize_env is not None:
            raise RuntimeError("FHRTD3 does not support VecNormalize — lag "
                               "observations would bypass the normalisation")
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        if self.prioritized_replay:
            self._fhr_anneal_per_beta()

        actor_losses, critic_losses = [], []
        step_diags = []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            inds = self.replay_buffer.last_batch_inds
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = torch.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss (TD3 scale: the plain sum over critics)
            if self.prioritized_replay:
                w = self.replay_buffer.to_torch(
                    self.replay_buffer.last_batch_weights).reshape(-1, 1)
                critic_loss = sum(
                    ((current_q - target_q_values).pow(2) * w).mean()
                    for current_q in current_q_values)
            else:
                critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, torch.Tensor)
            critic_losses.append(critic_loss.item())

            lam = self._lambda_eff()
            diag = self._fhr_base_diag(td_loss=critic_losses[-1], lam=lam)
            diag["actor_loss"] = np.nan       # filled on delayed-update steps
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
                # the guard itself is gated on fhr_weight > 0). _n_updates
                # already advanced, so the policy_delay cadence is untouched;
                # the skipped step simply forfeits its delayed actor update.
                self.nan_skips += 1
                diag["nan_skips"] = self.nan_skips
                continue

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self.prioritized_replay:
                with torch.no_grad():
                    td = torch.stack([(q - target_q_values).abs()
                                      for q in current_q_values]).mean(dim=0)
                self.replay_buffer.update_priorities(
                    inds, td.squeeze(1).cpu().numpy())

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())
                diag["actor_loss"] = actor_losses[-1]

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self._fhr_aggregate_pending(step_diags)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
