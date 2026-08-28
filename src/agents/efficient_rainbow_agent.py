"""EfficientRainbowAgent — a data-efficient Atari-100k learner with FHR support.

DrQ's "Efficient DQN" recipe (Kostrikov et al. 2021, Table 4: double Q +
dueling + n-step 10 + random-shift/intensity augmentation + data-efficient
hyperparameters) with this repo's IQN-dueling head swapped in for the plain
dueling head, plus the FHR recurrence penalty inherited from
:class:`FHRDQNAgent` (``fhr_weight: 0`` is the baseline arm). Deliberately
EXCLUDED, per DrQ: prioritized replay (interface stubbed for later — see
``EpisodicReplayBuffer.update_priorities``), noisy nets (exploration is
ε-greedy), C51.

Composition:
  * :class:`FHRDQNAgent` contributes the penalty machinery unchanged — c/d
    coefficient group, hard warm-up, λ ramp-down, episodic buffer plumbing,
    diagnostics aggregation, save/load.
  * :class:`IQNTDMixin` contributes the quantile-Huber loss and Bellman target
    quantiles (per-sample discounts, so n-step tail windows are native).
  * Only ``_train_step`` is new: n-step sample -> DrQ augmentation -> IQN TD ->
    FHR penalty.

Speed vs the original RainbowDQNAgent: 8 loss taus instead of 32 (each tau
costs ~half an encoder forward), plain linear head layers instead of noisy,
uniform sampling instead of python-loop PER — about 3.3x cheaper per gradient
step.

Augmentation discipline (the FHR-critical part): the TD states and next-states
get independent random shifts (DrQ standard), but the penalty evaluates each
anchor and its r lag states under ONE shared shift/intensity draw per
sequence, in a single fused expected-Q forward on the deterministic midpoint
tau grid. Sharing the transform keeps augmentation noise out of the recurrence
residual — the signal behind penalty_raw / residual_rms and the ramp-down
trigger; the fixed grid keeps the residual free of tau-sampling noise.
``pi()`` and the analysis rollouts are never augmented.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .augmentation import intensity, random_shift
from .fhrdqn_agent import FHRDQNAgent
from .rainbow_agent import IQNTDMixin, RainbowDQNAgent, RainbowIQNNetwork


class EfficientRainbowAgent(IQNTDMixin, FHRDQNAgent):
    """See the module docstring. Extra kwargs over FHRDQNAgent (all mappable
    to config ``agent:`` keys):

        n_step: multi-step return horizon (targets are aggregated at SAMPLE
            time by the episodic buffer; storage stays strictly per-step, so
            the FHR predecessor windows and reward_lags contracts hold).
        n_quantiles / n_quantiles_target: online/target loss tau counts. Also
            reused for Double-DQN action selection (n_quantiles_select).
        n_quantiles_act: taus in the fixed midpoint grid behind expected-Q
            acting and the Hankel Q/V/A traces (deterministic given weights).
        n_quantiles_fhr: taus in the fixed grid behind the FHR anchor/lag
            expected-Q forward.
        n_cos / head_hidden / huber_kappa: IQN head configuration.
        use_augmentation / aug_pad / aug_intensity: DrQ augmentation. Applied
            only when observations are image-shaped (B, C, H, W).
        prioritized_replay: must stay False — reserved config surface for the
            future PER-over-episodic-buffer implementation.

    ``q_network``/``nn_extra_kwargs`` supply an ENCODER (obs -> (B, F)
    features; a ``feature_dim`` attribute skips the dummy forward), which is
    wrapped with the RainbowIQNNetwork head in-agent.
    """

    def __init__(self, *, q_network, nn_extra_kwargs,
                 n_step: int = 10,
                 n_quantiles: int = 8, n_quantiles_target: int = 8,
                 n_quantiles_act: int = 32, n_quantiles_fhr: int = 8,
                 n_cos: int = 64, head_hidden: int = 512,
                 huber_kappa: float = 1.0,
                 use_augmentation: bool = False, aug_pad: int = 4,
                 aug_intensity: float = 0.05,
                 prioritized_replay: bool = False,
                 **fhr_kwargs):
        if prioritized_replay:
            raise NotImplementedError(
                "prioritized_replay is reserved: PER over the episodic buffer "
                "is not implemented yet — set it to false")
        if n_step < 1:
            raise ValueError(f"n_step must be >= 1, got {n_step}")

        env = fhr_kwargs["env"]
        n_actions = int(env.action_space.n)

        def head_factory():
            encoder = q_network(**nn_extra_kwargs)
            feature_dim = RainbowDQNAgent._infer_feature_dim(encoder, env)
            return RainbowIQNNetwork(
                encoder, feature_dim, n_actions,
                n_cos=n_cos, head_hidden=head_hidden, dueling=True,
                noisy=False, fixed_act_taus=True,
                n_quantiles_act=n_quantiles_act)

        # QAgent's ctor (via FHRDQNAgent) instantiates policy/target nets from
        # the factory and creates the optimiser BEFORE FHRDQNAgent appends the
        # c/d param group — the ordering every FHR test pins.
        super().__init__(q_network=head_factory, nn_extra_kwargs={}, **fhr_kwargs)

        # Kept for subclasses that re-instantiate networks (e.g. BBFAgent's
        # shrink-and-perturb resets draw fresh weights from this factory).
        self._net_factory = head_factory
        self.n_step = n_step
        self.n_quantiles = n_quantiles
        self.n_quantiles_target = n_quantiles_target
        self.n_quantiles_fhr = n_quantiles_fhr
        self.n_quantiles_select = n_quantiles      # IQNTDMixin: Double-Q select taus
        self.huber_kappa = huber_kappa
        self.use_augmentation = use_augmentation
        self.aug_pad = aug_pad
        self.aug_intensity = aug_intensity

    # ------------------------------------------------------------- learning
    # Schedule seams: constants here; BBFAgent overrides them with its
    # annealed values. _train_step reads ONLY these, never the raw attrs.
    def _current_n_step(self) -> int:
        return self.n_step

    def _current_gamma(self) -> float:
        return self.loss.gamma

    def _post_optim_step(self):
        """Hook after a successful optimiser.step() (no-op here; BBFAgent
        does its per-gradient-step EMA target update + reset check)."""

    def _augment(self, x, offsets=None, factors=None):
        """One DrQ augmentation pass (shift + intensity) on a 0-255 float
        image batch; returns (augmented, offsets, factors) so a caller can
        replay the identical transform on another batch."""
        x, offsets = random_shift(x, self.aug_pad, offsets=offsets)
        if self.aug_intensity > 0:
            x, factors = intensity(x, self.aug_intensity, factors=factors)
        return x, offsets, factors

    def _train_step(self):
        states, actions, returns, next_list, discounts, handles = \
            self.replay_buffer.sample_nstep_transitions(
                self.batch_size, self._current_n_step(), self._current_gamma(),
                with_handles=True)
        # atari storage is uint8/CPU and moves to the device per batch; the
        # classical path stores on-device float32, for which .to() is a no-op
        states = states.to(self.device)
        actions = actions.to(self.device)
        returns = returns.to(self.device)
        discounts = discounts.to(self.device)
        non_final_mask = torch.tensor([s is not None for s in next_list],
                                      device=self.device)

        augment = self.use_augmentation and states.dim() == 4
        # TD forwards see augmented frames (independent draws for s and s',
        # per DrQ); the raw `states` are kept for the penalty's anchor below.
        td_states = self._augment(states.float())[0] if augment else states

        with torch.no_grad():
            next_states = None
            if non_final_mask.any():
                if self.compile_net:
                    # Static (B, ...) bootstrap batch: CUDA graphs need one
                    # input shape, and the non-final count varies with however
                    # many sampled transitions ended an episode. Terminal rows
                    # are zero-filled here and masked out in _target_quantiles.
                    present = [i for i, s in enumerate(next_list) if s is not None]
                    ref = next_list[present[0]]
                    next_states = torch.zeros((len(next_list), *ref.shape[1:]),
                                              dtype=ref.dtype)
                    next_states[present] = torch.cat(
                        [next_list[i] for i in present])
                    next_states = next_states.to(self.device)
                else:
                    next_states = torch.cat(
                        [s for s in next_list if s is not None]).to(self.device)
                if augment:
                    next_states = self._augment(next_states.float())[0]
            target_theta = self._target_quantiles(
                returns, discounts, non_final_mask, next_states)      # (B, N')

        taus = self.policy_net._sample_taus(self.batch_size, self.n_quantiles,
                                            self.device)
        theta = self.policy_net.quantiles(td_states, taus)            # (B, N, A)
        theta_a = theta.gather(
            2, actions.view(-1, 1, 1).expand(-1, self.n_quantiles, 1)).squeeze(2)
        td = target_theta.unsqueeze(1) - theta_a.unsqueeze(2)         # (B, N, N')
        rho = self._quantile_huber(td, taus)
        loss = rho.mean(dim=2).sum(dim=1).mean()

        lam = self._lambda_eff()
        diag = {"td_loss": float(loss.detach()), "lambda_eff": lam,
                "penalty_raw": np.nan, "penalty_weighted": 0.0,
                "residual_rms": np.nan, "b_h": np.nan, "unique_eps": np.nan,
                "sum_c": float(self.c.detach().sum()),
                "companion_radius": self._companion_radius(),
                "rampdown_scale": self._rampdown_scale(),
                "rampdown_penalty_bar": (float("nan") if (bar := self._penalty_bar()) is None
                                         else bar),
                "nan_skips": self.nan_skips}
        for j in range(self.fhr_order):
            diag[f"c_{j + 1}"] = float(self.c[j].detach())
            if self.reward_lags:
                diag[f"d_{j + 1}"] = float(self.d[j].detach())

        if self.fhr_weight > 0:
            r = self.fhr_order
            keep = [i for i, (_, t) in enumerate(handles) if t >= r]
            diag["b_h"] = float(len(keep))
            diag["unique_eps"] = float(len({handles[i][0] for i in keep}))
            if keep:
                p_states, p_actions, p_rewards = self.replay_buffer.gather_predecessors(
                    [handles[i] for i in keep], r)
                p_states = p_states.to(self.device)
                p_actions = p_actions.to(self.device)
                p_rewards = p_rewards.to(self.device)
                n = len(keep)
                with torch.enable_grad() if lam > 0 else torch.no_grad():
                    # Fused expected-Q forward over each anchor state and its r
                    # lags — (n, r+1, obs) -> (n*(r+1), obs) — on the fixed
                    # midpoint tau grid. One shift/intensity draw per SEQUENCE,
                    # replayed across its r+1 frames, so the recurrence
                    # residual never sees augmentation noise.
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
                                n, 1, 1, 1, device=self.device).clamp_(-2.0, 2.0)
                                ).repeat_interleave(r + 1, dim=0)
                            flat, _ = intensity(flat, self.aug_intensity,
                                                factors=fac)
                    out = self.policy_net(flat, n_taus=self.n_quantiles_fhr)
                    a_seq = torch.cat(
                        [actions[keep].view(n, 1), p_actions], dim=1)   # (n, r+1)
                    q_seq = out.gather(
                        1, a_seq.reshape(n * (r + 1), 1)).view(n, r + 1)
                    anchor = q_seq[:, 0]
                    q_lags = q_seq[:, 1:]
                    prediction = q_lags @ self.c
                    if self.reward_lags:
                        prediction = prediction + p_rewards @ self.d
                    penalty = F.huber_loss(anchor, prediction)
                diag["penalty_raw"] = float(penalty.detach())
                diag["penalty_weighted"] = lam * diag["penalty_raw"]
                if lam > 0:
                    self._ep_penalty_vals.append(diag["penalty_raw"])
                diag["residual_rms"] = float(
                    (anchor.detach() - prediction.detach()).pow(2).mean().sqrt())
                if lam > 0:
                    loss = loss + lam * penalty

        self._grad_steps += 1
        if not torch.isfinite(loss):
            self.nan_skips += 1
            diag["nan_skips"] = self.nan_skips
            return diag
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),
                                                   self.grad_clip_norm)
        if not torch.isfinite(grad_norm):
            self.nan_skips += 1
            diag["nan_skips"] = self.nan_skips
            self.optimiser.zero_grad()
            return diag
        self.optimiser.step()
        self._post_optim_step()
        return diag
