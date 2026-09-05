"""FHR (learned-linear-recurrence Hankel-rank) regulariser bootstrapped onto
Stable-Baselines3's sample-efficient DQN-family methods.

Two algorithm subclasses — FHRDQN (stable_baselines3.DQN) and FHRQRDQN
(sb3_contrib.QRDQN) — reproduce their parent's train() step exactly and add the
FHR penalty of agents/fhrdqn_agent.py on top of the TD batch:

    rho_b = Q(s_t, a_t) - sum_j c_j Q(s_{t-j}, a_{t-j})
                        - [reward_lags] sum_k d_k r_{t-k}
    L = L_TD + lambda * Huber(anchor, prediction)

Design contracts carried over from FHRDQNAgent (src/agents/fhrdqn_agent.py):
  * fhr_weight=0 reproduces the stock SB3 algorithm bit-for-bit: the penalty
    block is skipped entirely, the replay buffer draws the same np.random
    stream as stable_baselines3.common.buffers.ReplayBuffer, and the extra
    c/d optimiser param group receives no gradients.
  * The penalty rides the ordinary i.i.d. TD batch — every sampled transition
    with t >= r contributes, its r same-episode predecessors fetched from the
    (episode, t)-annotated ring buffer. Anchor and lags both use the online
    network; the anchor is the same tensor as the TD term's Q(s_t, a_t)
    (for QR-DQN, the quantile mean of the taken action's quantiles).
  * c (and d for ARX) live in their own param group of the policy optimiser:
    weight_decay 0, independent lr, excluded from gradient clipping (which
    targets policy.parameters()) and from the target-network sync.
  * Hard warm-up: lambda is exactly 0 for the first warmup_grad_steps
    gradient steps, then full strength (no ramp-up). Optional one-way
    automatic ramp-DOWN mirrors FHRDQNAgent.notify_episode_end.
  * No log/sign transform anywhere — the recurrence is on raw Q-values.
  * Twin-critic hosts combine the per-critic Huber terms per
    _fhr_critic_reduction: "mean" for the SAC family (its TD term is
    0.5 * sum_i MSE_i, the per-critic mean) and "sum" for TD3
    (agents/sb3_td3_fhr.py, TD term sum_i MSE_i) — so lambda always weighs
    the penalty against the TD term on the host algorithm's own scale.
  * grad_probe_every > 0 switches on the gradient-stream probe
    (_fhr_grad_probe): |d penalty/d theta| / |d L_TD/d theta| and their cosine
    on the online value network, measured on EVERY arm including the lambda=0
    baseline (never entering the loss) and logged as grad_ratio / grad_rho /
    grad_cos next to the loss-side loss_ratio / rho_loss. These stream ratios,
    not lambda, are the quantities that transfer between algorithms and pick
    lambda for the next family.
  * fhr_lag_source picks who evaluates the recurrence's lag values
    Q(s_{t-j}, a_{t-j}) on the right-hand side: "online" (default) is the
    in-graph online network, so the penalty gradient reaches anchor AND
    lags; "detached" keeps the online network but evaluates the lags
    grad-free; "target" reads them from the algorithm's target twin
    (grad-free, a backward bootstrap). The anchor Q(s_t, a_t) is always the
    in-graph online value and c/d always train, whichever source is chosen.

The rest of this module is the glue that makes an SB3 run indistinguishable
from a repo-native run for the analysis stack and the result viewer app:

  * FHRSB3Callback drives RunLogger (rewards.csv with the steps column,
    train_diagnostics.csv with the FHR metric names the viewer knows,
    checkpoints latest/best/final) and calls training.run_analysis_tick — the
    same Q-matrix rank / Hankel sweep / autoregressive-value-probe dispatch
    the classic dqn_training_loop uses — every analysis.ep_freq episodes.
  * SB3QAgentAdapter exposes the QAgent surface (device, policy_net, pi,
    act_greedy, save) those analyses expect, wrapping q_net (DQN) or the
    quantile-mean of quantile_net (QR-DQN).
"""
import contextlib
import csv
import pathlib
import re
import warnings

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ContinuousCritic

try:
    from sb3_contrib import QRDQN
except ImportError:  # pragma: no cover - sb3_contrib is an optional extra
    QRDQN = None


class FHREpisodicReplayBuffer(ReplayBuffer):
    """SB3 ReplayBuffer + per-slot (episode id, t-within-episode) bookkeeping.

    Sampling draws the identical np.random stream as the stock buffer (the
    two randint calls of BaseBuffer.sample/_get_samples, in the same order),
    so a lambda=0 FHR run and a stock SB3 run consume randomness identically.
    The sampled slot indices are stashed on self.last_batch_inds; the FHR
    train step resolves each index's r ring-predecessors arithmetically —
    slot i-1 holds t-1 of the same episode unless evicted, and eviction or an
    episode boundary shows up as an episode-id mismatch (ids are unique and
    unwritten slots hold -1), which drops the sample from the penalty.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_envs != 1:
            raise ValueError(
                "FHREpisodicReplayBuffer needs n_envs == 1: predecessor "
                f"slots are ring-adjacent only for a single env (got {self.n_envs})")
        if self.optimize_memory_usage:
            raise ValueError("FHREpisodicReplayBuffer does not support "
                             "optimize_memory_usage=True")
        self.episode_ids = np.full((self.buffer_size,), -1, dtype=np.int64)
        self.t_in_episode = np.zeros((self.buffer_size,), dtype=np.int64)
        self._current_episode = 0
        self._current_t = 0
        self.last_batch_inds: np.ndarray | None = None

    def add(self, obs, next_obs, action, reward, done, infos) -> None:
        self.episode_ids[self.pos] = self._current_episode
        self.t_in_episode[self.pos] = self._current_t
        super().add(obs, next_obs, action, reward, done, infos)
        # done covers terminated and truncated alike — an episode boundary
        # either way (matches EpisodicReplayBuffer.close on both)
        if bool(np.asarray(done).reshape(-1)[0]):
            self._current_episode += 1
            self._current_t = 0
        else:
            self._current_t += 1

    def sample(self, batch_size: int, env=None):
        # verbatim BaseBuffer.sample so the RNG stream matches stock SB3;
        # only the stashing of batch_inds is new
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        self.last_batch_inds = batch_inds
        return self._get_samples(batch_inds, env=env)

    def predecessors(self, batch_inds: np.ndarray, order: int):
        """(keep, pred): keep marks batch rows with `order` valid same-episode
        predecessors; pred[(kept) i, j-1] is the slot of t-j (most recent
        first, matching c_1..c_r). Besides the episode-id match, each lag slot
        must hold exactly t-j: when a single episode is >= buffer_size steps
        long the ring wraps within the episode and the seam slots carry the
        same episode id at the wrong t — the exact-t check drops those."""
        lags = np.arange(1, order + 1)
        pred = (batch_inds[:, None] - lags[None, :]) % self.buffer_size
        same_episode = self.episode_ids[pred] == self.episode_ids[batch_inds][:, None]
        lag_t_ok = (self.t_in_episode[pred]
                    == self.t_in_episode[batch_inds][:, None] - lags[None, :])
        keep = ((self.t_in_episode[batch_inds] >= order)
                & (same_episode & lag_t_ok).all(axis=1))
        return keep, pred


class FHRRecurrenceHead(nn.Module):
    """The learned recurrence coefficients c (and d for ARX), Bellman-informed
    init as in FHRDQNAgent: with reward lags c1=1/gamma, d1=-1/gamma; pure AR
    with r>=2 c=(1+1/gamma, -1/gamma, 0, ...); r=1 pure AR c1=1/gamma."""

    def __init__(self, order: int, gamma: float, reward_lags: bool,
                 c_init=None):
        super().__init__()
        if c_init is not None:
            # explicit coefficient init (e.g. theory roots for a frozen-c
            # control arm) — replaces the Bellman-informed default for c only
            c0 = torch.as_tensor(list(c_init), dtype=torch.float32)
            if c0.shape != (order,):
                raise ValueError(f"c_init must have length {order} "
                                 f"(fhr_order), got {list(c_init)!r}")
            d0 = torch.zeros(order)
            if reward_lags:
                d0[0] = -1.0 / gamma
            self.c = nn.Parameter(c0)
            self.d = nn.Parameter(d0) if reward_lags else None
            return
        c0 = torch.zeros(order)
        if reward_lags:
            c0[0] = 1.0 / gamma
            d0 = torch.zeros(order)
            d0[0] = -1.0 / gamma
        elif order >= 2:
            c0[0], c0[1] = 1.0 + 1.0 / gamma, -1.0 / gamma
        else:
            c0[0] = 1.0 / gamma
        self.c = nn.Parameter(c0)
        self.d = nn.Parameter(d0) if reward_lags else None


def _bellman_init(order: int, gamma: float, reward_lags: bool):
    """The Bellman-informed coefficient init shared by every FHR head."""
    c0 = torch.zeros(order)
    d0 = torch.zeros(order) if reward_lags else None
    if reward_lags:
        c0[0] = 1.0 / gamma
        d0[0] = -1.0 / gamma
    elif order >= 2:
        c0[0], c0[1] = 1.0 + 1.0 / gamma, -1.0 / gamma
    else:
        c0[0] = 1.0 / gamma
    return c0, d0


def _head_sequential(qnet):
    """The head Sequential whose last layer defines the penultimate space:
    q_net/quantile_net for the DQN family, q_networks[0] for multi-critic
    modules (SAC's ContinuousCritic, SACD's DiscreteCritic)."""
    seq = getattr(qnet, "q_net", None) or getattr(qnet, "quantile_net", None)
    if seq is None and hasattr(qnet, "q_networks"):
        seq = qnet.q_networks[0]
    if seq is None:
        raise TypeError(f"cannot find the head Sequential on {type(qnet).__name__}")
    return seq


def penultimate_features(qnet, obs: torch.Tensor,
                         actions: torch.Tensor | None = None) -> torch.Tensor:
    """Activations feeding the FINAL linear layer of an SB3 value network:
    features_extractor output pushed through every head module except the
    last. Gradients flow into the shared trunk. For SAC's ContinuousCritic
    the head input is cat(features, actions) — pass the anchor actions; this
    deliberately does NOT go through q1_forward, whose no_grad feature
    extraction would cut the shared-trunk gradient path."""
    seq = _head_sequential(qnet)
    x = qnet.extract_features(obs, qnet.features_extractor)
    if isinstance(qnet, ContinuousCritic):
        if actions is None:
            raise ValueError("ContinuousCritic penultimate features need the "
                             "anchor actions (head input is cat(features, a))")
        x = torch.cat([x, actions], dim=1)
    for module in list(seq)[:-1]:
        x = module(x)
    return x


def penultimate_dim(qnet) -> int:
    return int(list(_head_sequential(qnet))[-1].in_features)


class FHRCoefficientPredictor(nn.Module):
    """State-action-conditioned recurrence coefficients: instead of one global
    c (and d for ARX) shared across all episodes, each anchor transition gets
    c(s_t, a_t) predicted by a network, and the penalty backpropagates into
    the predictor jointly with theta.

        mode "separate": an own MLP on [flatten(obs), one-hot(a)].
        mode "shared":   a single linear head on the ONLINE Q-network's
                         penultimate activations ++ one-hot(a); its gradient
                         flows back into the shared trunk (the point of
                         sharing — the penalty then shapes the representation
                         the way an auxiliary head does).

    The output layer starts with ZERO weights and a Bellman-init bias, so at
    initialisation every anchor receives exactly the coefficients the global
    FHRRecurrenceHead starts with — the state-conditioned run and a global-c
    run are identical until gradients differentiate them.

    Degeneracy caution (why this stays SMALL and slow-learning): each anchor
    contributes one residual equation but receives r free coefficients, so a
    sufficiently expressive per-state c could zero the residual without
    constraining Q at all. What keeps the penalty meaningful is smoothness —
    finite capacity forces nearby states to share coefficients. Keep hidden
    small and the learning rate at c_learning_rate.
    """

    def __init__(self, order: int, gamma: float, reward_lags: bool,
                 mode: str, in_dim: int, n_actions: int, hidden: int = 64,
                 action_encoding: str = "onehot"):
        super().__init__()
        if mode not in ("shared", "separate"):
            raise ValueError(f"c_predictor mode must be shared|separate, got {mode!r}")
        if action_encoding not in ("onehot", "raw"):
            raise ValueError("action_encoding must be onehot|raw, "
                             f"got {action_encoding!r}")
        self.order = order
        self.reward_lags = reward_lags
        self.mode = mode
        self.n_actions = n_actions        # action dim when encoding is "raw"
        self.action_encoding = action_encoding
        out_dim = order * (2 if reward_lags else 1)
        if mode == "separate":
            self.net = nn.Sequential(
                nn.Linear(in_dim + n_actions, hidden), nn.ReLU(),
                nn.Linear(hidden, out_dim))
            final = self.net[-1]
        else:
            self.net = nn.Linear(in_dim + n_actions, out_dim)
            final = self.net
        c0, d0 = _bellman_init(order, gamma, reward_lags)
        with torch.no_grad():
            final.weight.zero_()
            final.bias.copy_(torch.cat([c0, d0]) if reward_lags else c0)

    def forward(self, features: torch.Tensor, actions: torch.Tensor):
        """features: (n, in_dim) — flattened obs (separate) or penultimate
        activations (shared); actions: (n,) int64 for "onehot" or
        (n, act_dim) float for "raw". Returns (c (n, r), d (n, r) or None)."""
        if self.action_encoding == "raw":
            act_repr = actions.float().reshape(-1, self.n_actions)
        else:
            act_repr = F.one_hot(actions.long().reshape(-1),
                                 self.n_actions).float()
        out = self.net(torch.cat([features, act_repr], dim=1))
        if self.reward_lags:
            return out[:, :self.order], out[:, self.order:]
        return out, None


# The FHR parameter set a numbered experiment arm may override — mirrors
# run_fhrdqn_seeds.FHR_PARAMS so launchers/notebooks can validate identically.
FHR_PARAMS = ("fhr_weight", "fhr_order", "reward_lags",
              "warmup_grad_steps", "c_learning_rate", "c_predictor", "c_init",
              "fhr_lag_source",
              "prioritized_replay", "per_alpha", "per_beta0",
              "rampdown_reward_threshold", "rampdown_penalty_threshold",
              "rampdown_penalty_topk", "rampdown_patience_eps",
              "rampdown_episodes", "window_rank_every", "window_rank_lags",
              "grad_probe_every")


class _FHRMixin:
    """Shared FHR machinery for the SB3 subclasses. Assumes the host class is
    an SB3 OffPolicyAlgorithm with self.gamma, self.policy.optimizer, and an
    FHREpisodicReplayBuffer. Config knobs and semantics are 1:1 with
    FHRDQNAgent (see that class's docstring)."""

    # whether this algorithm's train() implements the PER weighting/priority
    # feedback; the DQN family does not (prioritized_replay is rejected there)
    _fhr_supports_per = False
    # how the per-critic Huber terms combine into penalty_raw: "mean" keeps
    # the single-critic scale (DQN, SAC — SAC's TD term is 0.5 * sum_i MSE_i,
    # i.e. the per-critic mean); "sum" matches a host whose TD term is the
    # plain sum over critics (TD3: sum_i MSE_i + lambda * sum_i Huber_i)
    _fhr_critic_reduction = "mean"

    def _set_fhr_config(self, fhr_weight, fhr_order, reward_lags,
                        warmup_grad_steps, c_learning_rate,
                        rampdown_reward_threshold, rampdown_penalty_threshold,
                        rampdown_penalty_topk, rampdown_patience_eps,
                        rampdown_episodes, c_predictor="none",
                        prioritized_replay=False, per_alpha=0.6,
                        per_beta0=0.4, window_rank_every=0,
                        window_rank_lags=16, c_init=None,
                        fhr_lag_source="online", grad_probe_every=0):
        self.fhr_weight = fhr_weight
        self.fhr_lag_source = fhr_lag_source
        self.fhr_order = fhr_order
        self.reward_lags = reward_lags
        self.warmup_grad_steps = warmup_grad_steps
        self.c_learning_rate = c_learning_rate
        self.c_predictor = c_predictor
        self.c_init = list(c_init) if c_init is not None else None
        self.prioritized_replay = prioritized_replay
        self.per_alpha = per_alpha
        self.per_beta0 = per_beta0
        self.rampdown_reward_threshold = rampdown_reward_threshold
        self.rampdown_penalty_threshold = rampdown_penalty_threshold
        self.rampdown_penalty_topk = rampdown_penalty_topk
        self.rampdown_patience_eps = rampdown_patience_eps
        self.rampdown_episodes = rampdown_episodes
        self.window_rank_every = window_rank_every
        self.window_rank_lags = window_rank_lags
        self.grad_probe_every = grad_probe_every

    def _validate_fhr_config(self):
        if self.fhr_order < 1:
            raise ValueError(f"fhr_order must be >= 1, got {self.fhr_order}")
        if self.window_rank_every < 0:
            raise ValueError(
                f"window_rank_every must be >= 0, got {self.window_rank_every}")
        if self.window_rank_lags < 1:
            raise ValueError(
                f"window_rank_lags must be >= 1, got {self.window_rank_lags}")
        if self.grad_probe_every < 0:
            raise ValueError(
                f"grad_probe_every must be >= 0, got {self.grad_probe_every}")
        if self.prioritized_replay and not self._fhr_supports_per:
            raise ValueError(
                "prioritized_replay is only supported by the SAC-family FHR "
                f"algorithms, not {type(self).__name__}")
        if self.fhr_lag_source not in ("online", "detached", "target"):
            raise ValueError("fhr_lag_source must be one of online|detached|"
                             f"target, got {self.fhr_lag_source!r}")
        if self.c_init is not None:
            if len(self.c_init) != self.fhr_order:
                raise ValueError(f"c_init must have length fhr_order="
                                 f"{self.fhr_order}, got {self.c_init!r}")
            if self.c_predictor != "none":
                raise ValueError("c_init only applies to the global c head "
                                 "(c_predictor: none)")
        if self.c_predictor not in ("none", "shared", "separate"):
            raise ValueError("c_predictor must be one of none|shared|separate, "
                             f"got {self.c_predictor!r}")
        if self.rampdown_episodes < 0:
            raise ValueError(f"rampdown_episodes must be >= 0, got {self.rampdown_episodes}")
        if self.rampdown_patience_eps < 1:
            raise ValueError(f"rampdown_patience_eps must be >= 1, got {self.rampdown_patience_eps}")
        if self.rampdown_penalty_topk < 1:
            raise ValueError(f"rampdown_penalty_topk must be >= 1, got {self.rampdown_penalty_topk}")
        if self.fhr_order == 1 and not self.reward_lags and self.fhr_weight > 0:
            warnings.warn(
                "Pure AR with fhr_order=1 cannot be satisfied by a "
                "Bellman-consistent Q under constant per-step rewards; use "
                "fhr_order >= 2 or reward_lags=True.", stacklevel=2)

    def _setup_model(self) -> None:
        super()._setup_model()
        self._validate_fhr_config()
        if not isinstance(self.replay_buffer, FHREpisodicReplayBuffer):
            raise TypeError("FHR algorithms need replay_buffer_class="
                            "FHREpisodicReplayBuffer (the default) — got "
                            f"{type(self.replay_buffer).__name__}")
        if getattr(self, "n_steps", 1) > 1:
            raise ValueError(
                "FHR algorithms do not support n_steps > 1: forcing the FHR "
                "buffer bypasses SB3's NStepReplayBuffer selection, and the "
                "recurrence penalty assumes 1-step adjacency anyway")
        self.fhr_head = FHRRecurrenceHead(
            self.fhr_order, self.gamma, self.reward_lags,
            c_init=self.c_init).to(self.device)
        # State-conditioned coefficients (c_predictor shared|separate): the
        # predictor replaces the global c/d in the penalty AND in the c-lr
        # optimiser group; the global head stays constructed (it is the init
        # reference and keeps checkpoints uniform) but receives no gradients.
        self.fhr_predictor = None
        if self.c_predictor != "none":
            online = self._fhr_online_qnet()
            if self.c_predictor == "shared":
                in_dim = penultimate_dim(online)
            else:
                in_dim = int(np.prod(self.observation_space.shape))
            self.fhr_predictor = self._fhr_build_predictor(in_dim)
            coeffs = list(self.fhr_predictor.parameters())
        else:
            coeffs = [self.fhr_head.c]
            if self.fhr_head.d is not None:
                coeffs.append(self.fhr_head.d)
        # Own param group: no weight decay, independent lr; excluded from
        # gradient clipping (which targets policy.parameters()) and from the
        # target-net polyak sync (not part of q_net). NB for the "shared"
        # predictor: only the coefficient HEAD lives in this group — the
        # trunk it reads penultimate features from is already in group 0 and
        # receives the penalty gradient at the ordinary learning rate.
        self._fhr_coeff_optimizer().add_param_group(
            {"params": coeffs, "lr": self.c_learning_rate, "weight_decay": 0.0})
        self._fhr_group_index = len(self._fhr_coeff_optimizer().param_groups) - 1

        # "NN%" relative penalty bar parsed once, as in FHRDQNAgent
        self._rd_pen_abs = None
        self._rd_pen_frac = None
        if isinstance(self.rampdown_penalty_threshold, str):
            m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*%\s*",
                             self.rampdown_penalty_threshold)
            if not m:
                raise ValueError(
                    "rampdown_penalty_threshold must be a number or a "
                    "percentage string like '40%', got "
                    f"{self.rampdown_penalty_threshold!r}")
            self._rd_pen_frac = float(m.group(1)) / 100.0
        elif self.rampdown_penalty_threshold is not None:
            self._rd_pen_abs = float(self.rampdown_penalty_threshold)
        # mutable training state — keep restored values on a .load() resume
        if not hasattr(self, "_fhr_grad_steps"):
            self._fhr_grad_steps = 0
            self.nan_skips = 0
            # penalty_raw values bucketed by the episode in progress when the
            # gradient burst ran (FHRSB3Callback advances the hint at every
            # rollout end); notify_episode_end(e) consumes bucket e
            self._ep_penalty_buckets: dict[int, list[float]] = {}
            self._fhr_episode_hint = 0
            self._rd_window: list[tuple[float, float]] = []
            self._pen_top: list[float] = []
            self._rd_k = None
            self._pending_diags: list[dict] = []
            self._pending_window_rank: list[dict] = []
            self._pending_window_arrays: dict[str, np.ndarray] = {}

    def _fhr_online_qnet(self):
        """The online value network the shared predictor reads features from."""
        return getattr(self, "q_net", None) or self.quantile_net

    # -- hooks whose defaults reproduce the DQN-family behavior exactly; the
    # -- SAC-family subclasses override them (see agents/sb3_sac_fhr.py) -----
    def _fhr_coeff_optimizer(self):
        """The optimizer that hosts the c/d (or predictor) param group."""
        return self.policy.optimizer

    def _fhr_predictor_action_repr(self):
        """(dim, encoding) the coefficient predictor uses for actions."""
        return int(self.action_space.n), "onehot"

    def _fhr_build_predictor(self, in_dim: int) -> "FHRCoefficientPredictor":
        n_act, encoding = self._fhr_predictor_action_repr()
        return FHRCoefficientPredictor(
            self.fhr_order, self.gamma, self.reward_lags,
            self.c_predictor, in_dim, n_act,
            action_encoding=encoding).to(self.device)

    def _fhr_lag_actions(self, raw_actions: np.ndarray, n: int, r: int):
        """Buffer actions of the flattened predecessor batch, shaped for
        lag_q_fn: an int64 index column for discrete gathers (default)."""
        return self.replay_buffer.to_torch(
            raw_actions).long().reshape(n * r, -1)[:, :1]

    def _fhr_anchor_actions(self, raw_actions: np.ndarray, n: int):
        """Buffer actions of the anchor rows, shaped for the coefficient
        predictor: (n,) int64 for one-hot encoding (default)."""
        return self.replay_buffer.to_torch(
            raw_actions).long().reshape(n, -1)[:, 0]

    def _fhr_shared_features(self, anchor_obs: torch.Tensor,
                             anchor_acts: torch.Tensor) -> torch.Tensor:
        """Online-network penultimate features for the shared predictor."""
        return penultimate_features(self._fhr_online_qnet(), anchor_obs)

    def _get_torch_save_params(self):
        state_dicts, tensors = super()._get_torch_save_params()
        state_dicts = state_dicts + ["fhr_head"]
        if getattr(self, "fhr_predictor", None) is not None:
            state_dicts = state_dicts + ["fhr_predictor"]
        return state_dicts, tensors

    def _update_learning_rate(self, optimizers) -> None:
        # SB3's schedule update overwrites every param group's lr — restore
        # the coefficients' independent learning rate afterwards
        super()._update_learning_rate(optimizers)
        self._fhr_coeff_optimizer().param_groups[self._fhr_group_index]["lr"] = \
            self.c_learning_rate

    # -- penalty schedule: hard warm-up, optional triggered ramp-down -------
    def _rampdown_scale(self) -> float:
        if self._rd_k is None:
            return 1.0
        if self.rampdown_episodes <= 0:
            return 0.0
        return max(0.0, 1.0 - self._rd_k / self.rampdown_episodes)

    def _lambda_eff(self) -> float:
        if self._fhr_grad_steps < self.warmup_grad_steps:
            return 0.0
        return self.fhr_weight * self._rampdown_scale()

    def _penalty_bar(self) -> float | None:
        if self._rd_pen_abs is not None:
            return self._rd_pen_abs
        if (self._rd_pen_frac is not None
                and len(self._pen_top) >= self.rampdown_penalty_topk):
            return self._rd_pen_frac * float(np.mean(self._pen_top))
        return None

    def notify_episode_end(self, episode: int, episode_reward: float) -> None:
        """Per-episode hook (FHRSB3Callback calls it): the lambda ramp-down
        trigger of FHRDQNAgent.notify_episode_end, adapted to SB3's burst
        scheduling. The classic loop trains inside every episode, so each
        episode has its own residuals; SB3 trains in bursts sparser than
        episodes, so residuals are bucketed by the episode in progress when
        the burst ran, and an episode whose bucket is empty (no burst during
        its lifetime) counts as no-data: the penalty gate below ignores it
        instead of letting it veto the trigger (v1's all-finite gate would
        deadlock under burst scheduling)."""
        vals = self._ep_penalty_buckets.pop(episode, [])
        ep_pen = float(np.mean(vals)) if vals else float("nan")
        if self._rd_k is not None:
            self._rd_k += 1
            return
        if self.rampdown_reward_threshold is None or self.fhr_weight <= 0:
            return
        if self._fhr_grad_steps < self.warmup_grad_steps:
            return
        if np.isfinite(ep_pen):
            self._pen_top.append(ep_pen)
            self._pen_top.sort(reverse=True)
            del self._pen_top[self.rampdown_penalty_topk:]
        self._rd_window.append((float(episode_reward), ep_pen))
        del self._rd_window[:-self.rampdown_patience_eps]
        if len(self._rd_window) < self.rampdown_patience_eps:
            return
        rewards = [r for r, _ in self._rd_window]
        pens = [p for _, p in self._rd_window]
        if float(np.mean(rewards)) < self.rampdown_reward_threshold:
            return
        bar = self._penalty_bar()
        finite_pens = [p for p in pens if np.isfinite(p)]
        if self.rampdown_penalty_threshold is not None and (
                bar is None or not finite_pens
                or not all(p >= bar for p in finite_pens)):
            return
        self._rd_k = 1
        print(f"FHR lambda ramp-down triggered at episode {episode}: "
              f"mean reward {np.mean(rewards):.1f} >= "
              f"{self.rampdown_reward_threshold} over "
              f"{self.rampdown_patience_eps} eps — lambda -> 0 over "
              f"{self.rampdown_episodes} episode(s)")

    def _companion_radius(self) -> float:
        c = self.fhr_head.c.detach().cpu().numpy()
        roots = np.roots(np.concatenate(([1.0], -c)))
        return float(np.abs(roots).max()) if roots.size else 0.0

    # -- per-gradient-step penalty + diagnostics ----------------------------
    def _fhr_base_diag(self, td_loss: float, lam: float) -> dict:
        diag = {"td_loss": td_loss, "lambda_eff": lam,
                "penalty_raw": np.nan, "penalty_weighted": 0.0,
                "residual_rms": np.nan, "b_h": np.nan, "unique_eps": np.nan,
                "sum_c": (np.nan if self.fhr_predictor is not None
                          else float(self.fhr_head.c.detach().sum())),
                "companion_radius": (np.nan if self.fhr_predictor is not None
                                     else self._companion_radius()),
                **({"c_spread": np.nan} if self.fhr_predictor is not None else {}),
                "rampdown_scale": self._rampdown_scale(),
                "rampdown_penalty_bar": (float("nan")
                                         if (bar := self._penalty_bar()) is None
                                         else bar),
                "nan_skips": self.nan_skips,
                # stream ratios: loss-side filled by the penalty, gradient-side
                # by the probe (nan on non-probe steps; the burst nanmean and
                # the notebook binning both skip nans)
                "loss_ratio": np.nan, "rho_loss": np.nan,
                "grad_norm_td": np.nan, "grad_norm_pen": np.nan,
                "grad_ratio": np.nan, "grad_rho": np.nan, "grad_cos": np.nan}
        for j in range(self.fhr_order):
            if self.fhr_predictor is not None:
                # per-state coefficients: the penalty fills batch means in
                diag[f"c_{j + 1}"] = np.nan
                if self.reward_lags:
                    diag[f"d_{j + 1}"] = np.nan
            else:
                diag[f"c_{j + 1}"] = float(self.fhr_head.c[j].detach())
                if self.reward_lags:
                    diag[f"d_{j + 1}"] = float(self.fhr_head.d[j].detach())
        return diag

    def _fhr_penalty(self, anchor_q_sa: torch.Tensor, lag_q_fn, lam: float,
                     diag: dict) -> torch.Tensor | None:
        """The recurrence-residual penalty on the current TD batch.

        anchor_q_sa: (B,) online Q(s_t, a_t) — the same tensor the TD loss
        uses, so the anchor gradient is shared. lag_q_fn(obs, actions) -> (N,)
        evaluates online Q(s, a) for the flattened predecessor batch (under
        fhr_lag_source="detached" it runs grad-free; under "target" it is
        replaced by the algorithm's _fhr_target_lag_q_fns twin).
        Returns the Huber penalty (in the graph iff lam > 0) or None when no
        batch sample has r valid predecessors; fills diag in place.
        """
        return self._fhr_penalty_multi([anchor_q_sa], [lag_q_fn], lam, diag)

    def _fhr_target_lag_q_fns(self):
        """Target-net twins of the online lag_q closures, one per critic —
        consumed when fhr_lag_source="target". Each concrete algorithm
        supplies its own (it knows its target module's name and shape)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support fhr_lag_source='target'")

    def _fhr_penalty_multi(self, anchor_q_sas: list, lag_q_fns: list,
                           lam: float, diag: dict,
                           need_grad: bool = False) -> torch.Tensor | None:
        """_fhr_penalty over N online critics sharing ONE c (or one predictor).

        The per-critic Huber terms combine per `_fhr_critic_reduction`:
        "mean" -> sum_i Huber_i / N (DQN, SAC: for twin critics exactly
        0.5 * (Huber_1 + Huber_2), the single-critic scale matching SAC's
        0.5 * sum_i MSE_i TD term); "sum" -> sum_i Huber_i (TD3, whose TD
        term is the plain sum_i MSE_i, so lambda weighs sum against sum). The
        single-critic path is bit-identical to the pre-refactor _fhr_penalty
        either way.

        need_grad keeps the penalty in the graph even at lambda = 0 (for the
        gradient-stream probe, which differentiates it without adding it to
        the loss); the default keeps warm-up / baseline diagnostics out of
        the graph exactly as FHRDQNAgent does. Under fhr_lag_source
        "detached" / "target" the lag values enter that graph as constants,
        so the probe measures the penalty gradient that would actually act
        (anchor and c/d only), not the online in-graph reading.
        """
        buf = self.replay_buffer
        inds = buf.last_batch_inds
        keep, pred = buf.predecessors(inds, self.fhr_order)
        diag["b_h"] = float(keep.sum())
        diag["unique_eps"] = float(
            np.unique(buf.episode_ids[inds[keep]]).size) if keep.any() else 0.0
        if not keep.any():
            return None
        r = self.fhr_order
        pred_k = pred[keep]                                   # (n, r)
        n = pred_k.shape[0]
        flat = pred_k.reshape(-1)
        obs = buf.to_torch(buf.observations[flat, 0]).float()
        acts = self._fhr_lag_actions(buf.actions[flat, 0], n, r)
        keep_t = torch.as_tensor(np.flatnonzero(keep), device=self.device)
        # lambda = 0 (warm-up / baseline diagnostics): keep the penalty out of
        # the graph, exactly as FHRDQNAgent does
        with torch.enable_grad() if (lam > 0 or need_grad) else torch.no_grad():
            c_pred = d_pred = None
            if self.fhr_predictor is not None:
                anchor_inds = inds[keep]
                anchor_obs = buf.to_torch(buf.observations[anchor_inds, 0]).float()
                anchor_acts = self._fhr_anchor_actions(
                    buf.actions[anchor_inds, 0], n)
                if self.c_predictor == "shared":
                    feats = self._fhr_shared_features(anchor_obs, anchor_acts)
                else:
                    feats = anchor_obs.reshape(n, -1)
                c_pred, d_pred = self.fhr_predictor(feats, anchor_acts)
                c_mean = c_pred.detach().mean(dim=0)
                diag["sum_c"] = float(c_mean.sum())
                diag["c_spread"] = float(c_pred.detach().std(dim=0).mean())
                for j in range(r):
                    diag[f"c_{j + 1}"] = float(c_mean[j])
                    if self.reward_lags:
                        diag[f"d_{j + 1}"] = float(d_pred.detach().mean(dim=0)[j])
                roots = np.roots(np.concatenate(([1.0], -c_mean.cpu().numpy())))
                diag["companion_radius"] = (float(np.abs(roots).max())
                                            if roots.size else 0.0)
            rews = None
            if self.reward_lags:
                rews = buf.to_torch(buf.rewards[pred_k, 0]).float()  # (n, r)
            # fhr_lag_source routing: "detached" keeps the online closures but
            # evaluates them grad-free; "target" swaps in the target-net twins
            # (also grad-free). The anchors are untouched either way.
            if self.fhr_lag_source == "target":
                lag_q_fns = self._fhr_target_lag_q_fns()
            lag_grad = (contextlib.nullcontext
                        if self.fhr_lag_source == "online" else torch.no_grad)
            hubers, sq_residuals = [], []
            for anchor_q_sa, lag_q_fn in zip(anchor_q_sas, lag_q_fns):
                with lag_grad():
                    q_lags = lag_q_fn(obs, acts).view(n, r)
                if c_pred is not None:
                    prediction = (q_lags * c_pred).sum(dim=1)
                    if self.reward_lags:
                        prediction = prediction + (rews * d_pred).sum(dim=1)
                else:
                    prediction = q_lags @ self.fhr_head.c
                    if self.reward_lags:
                        prediction = prediction + rews @ self.fhr_head.d
                anchor = anchor_q_sa[keep_t]
                hubers.append(F.huber_loss(anchor, prediction))
                sq_residuals.append(
                    (anchor.detach() - prediction.detach()).pow(2))
            if len(hubers) == 1:
                penalty = hubers[0]
            elif self._fhr_critic_reduction == "sum":
                penalty = sum(hubers)
            else:
                penalty = sum(hubers) / len(hubers)
        diag["penalty_raw"] = float(penalty.detach())
        diag["penalty_weighted"] = lam * diag["penalty_raw"]
        td = diag.get("td_loss", np.nan)
        if np.isfinite(td) and td > 0:
            diag["loss_ratio"] = diag["penalty_raw"] / td
            diag["rho_loss"] = lam * diag["loss_ratio"]
        if lam > 0:
            self._ep_penalty_buckets.setdefault(
                self._fhr_episode_hint, []).append(diag["penalty_raw"])
        diag["residual_rms"] = float(
            torch.cat(sq_residuals).mean().sqrt())
        return penalty

    # -- gradient-stream probe ----------------------------------------------
    def _fhr_grad_probe_due(self) -> bool:
        """Whether this gradient step measures the TD-vs-FHR gradient streams.
        Like the window probe it is NOT gated on fhr_weight: on the lambda=0
        baseline the penalty's gradient (c at its Bellman init) is measured
        without ever entering the loss — the calibration signal
        lambda* = target / grad_ratio for the next family."""
        return (self.grad_probe_every > 0
                and self._fhr_grad_steps % self.grad_probe_every == 0)

    def _fhr_grad_probe(self, td_loss_t: torch.Tensor, penalty: torch.Tensor,
                        lam: float, diag: dict) -> None:
        """Gradient-stream diagnostics on the online value network's
        parameters theta, from the two loss terms of the SAME batch:

            g_td  = d L_TD / d theta        g_pen = d penalty_raw / d theta
            grad_ratio = |g_pen| / |g_td|          (unweighted: what lambda scales)
            grad_rho   = lambda_eff * grad_ratio   (the ratio actually acting)
            grad_cos   = <g_td, g_pen> / (|g_td| |g_pen|)   (< 0: conflict)

        torch.autograd.grad with retain_graph: no parameter's .grad is touched,
        the subsequent backward() sees the same graph, and no RNG is consumed,
        so a lambda=0 run with the probe on stays bit-for-bit stock.
        td_loss_t must be the TD term alone (before the penalty is added).
        Under fhr_lag_source "detached" / "target" the lag values are
        constants in the penalty graph, so g_pen is the gradient that would
        actually act on theta (through the anchor only) — a different
        reading from the "online" in-graph one of the same batch, which adds
        the lag-path term (the two can partially cancel, so neither is
        guaranteed larger).
        """
        params = [p for p in self._fhr_online_qnet().parameters()
                  if p.requires_grad]

        def grads(loss_t):
            gs = torch.autograd.grad(loss_t, params, retain_graph=True,
                                     allow_unused=True)
            return [torch.zeros_like(p) if g is None else g
                    for g, p in zip(gs, params)]

        g_td, g_pen = grads(td_loss_t), grads(penalty)
        with torch.no_grad():
            n_td = float(torch.sqrt(sum((g * g).sum() for g in g_td)))
            n_pen = float(torch.sqrt(sum((g * g).sum() for g in g_pen)))
            dot = float(sum((a * b).sum() for a, b in zip(g_td, g_pen)))
        diag["grad_norm_td"] = n_td
        diag["grad_norm_pen"] = n_pen
        ratio = n_pen / n_td if n_td > 0 else float("nan")
        diag["grad_ratio"] = ratio
        diag["grad_rho"] = lam * ratio
        diag["grad_cos"] = (dot / (n_td * n_pen)
                            if n_td > 0 and n_pen > 0 else float("nan"))

    # -- penalised-window spectrum probe ------------------------------------
    def _fhr_window_rank_due(self) -> bool:
        """Whether the current gradient step is a probe tick. Deliberately NOT
        gated on fhr_weight: the lambda=0 baseline measures the same windows,
        so its spectrum is the control the FHR arms compare against."""
        return (self.window_rank_every > 0
                and self._fhr_grad_steps % self.window_rank_every == 0)

    def _fhr_window_rank_probe(self, lag_q_fns) -> None:
        """Singular-value spectrum of the replay windows the penalty acts on.

        Uses the CURRENT TD batch's sampled slots (buf.last_batch_inds — no new
        randomness) and the same predecessor machinery as the penalty, but
        fetches window_rank_lags predecessors so the window is long enough for
        a meaningful spectrum. For each online critic it builds the stacked
        window matrix W (n_windows x (L+1)) of Q(s, a) along buffer
        transitions in time order [t-L, ..., t] and records the singular
        values of W and of its trailing (fhr_order+1)-column block — the
        EXACT sub-window the penalty residual is formed on. All rows share one
        padded schema (nan-filled) so the CSV columns are stable.

        Everything runs under no_grad and consumes no torch/np RNG, so a
        lambda=0 run with the probe enabled stays bit-for-bit stock.
        """
        buf = self.replay_buffer
        inds = buf.last_batch_inds
        L = max(self.window_rank_lags, self.fhr_order)
        keep, pred = buf.predecessors(inds, L)
        pen_cols = self.fhr_order + 1
        base = {"grad_step": int(self._fhr_grad_steps),
                "env_steps": int(self.num_timesteps),
                "window_len": L + 1, "penalty_len": pen_cols,
                "n_windows": int(keep.sum()),
                "unique_eps": (int(np.unique(buf.episode_ids[inds[keep]]).size)
                               if keep.any() else 0)}

        def padded_row(critic_idx, sv_full=None, sv_pen=None):
            row = {**base, "critic": critic_idx}
            for j in range(L + 1):
                row[f"sv_{j + 1:02d}"] = (float(sv_full[j]) if sv_full is not None
                                          and j < len(sv_full) else float("nan"))
            for j in range(pen_cols):
                row[f"pen_sv_{j + 1:02d}"] = (float(sv_pen[j]) if sv_pen is not None
                                              and j < len(sv_pen) else float("nan"))
            return row

        if not keep.any():
            for i in range(len(lag_q_fns)):
                self._pending_window_rank.append(padded_row(i))
            return
        # window slots in time order [t-L, ..., t-1, t]: pred columns are
        # most-recent-first (t-1 ... t-L), so reverse and append the anchor
        win = np.concatenate([pred[keep][:, ::-1], inds[keep][:, None]], axis=1)
        n = win.shape[0]
        flat = win.reshape(-1)
        obs = buf.to_torch(buf.observations[flat, 0]).float()
        acts = self._fhr_lag_actions(buf.actions[flat, 0], n, L + 1)
        with torch.no_grad():
            for i, lag_q in enumerate(lag_q_fns):
                W = lag_q(obs, acts).view(n, L + 1).detach().cpu().numpy()
                sv_full = np.linalg.svd(W, compute_uv=False)
                sv_pen = np.linalg.svd(W[:, -pen_cols:], compute_uv=False)
                self._pending_window_rank.append(padded_row(i, sv_full, sv_pen))
                key = (f"gs{base['grad_step']:08d}_"
                       f"ws{base['env_steps']:08d}_c{i}")
                self._pending_window_arrays[key] = W

    def drain_window_rank(self):
        """(rows, arrays) accumulated since the last drain — the callback
        flushes rows to window_hankel.csv and arrays to window_matrices/."""
        rows, self._pending_window_rank = self._pending_window_rank, []
        arrays, self._pending_window_arrays = self._pending_window_arrays, {}
        return rows, arrays

    def _fhr_aggregate_pending(self, step_diags: list[dict]) -> None:
        """nanmean-aggregate one train() call's per-step diagnostics into a
        single row (the FHRDQNAgent.train contract) and queue it for the
        callback to flush into train_diagnostics.csv."""
        if not step_diags:
            return

        def _nanmean(vals):
            finite = [v for v in vals if not np.isnan(v)]
            return float(np.mean(finite)) if finite else float("nan")

        self._pending_diags.append(
            {k: _nanmean([d[k] for d in step_diags]) for k in step_diags[0]})

    def drain_diagnostics(self) -> list[dict]:
        rows, self._pending_diags = self._pending_diags, []
        return rows


class FHRDQN(_FHRMixin, DQN):
    """stable_baselines3 DQN + the FHR recurrence penalty. fhr_weight=0 is
    bit-for-bit stock DQN (same RNG stream, same updates)."""

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
                 fhr_lag_source: str = "online",
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
                             fhr_lag_source=fhr_lag_source,
                             grad_probe_every=grad_probe_every)
        kwargs.setdefault("replay_buffer_class", FHREpisodicReplayBuffer)
        super().__init__(*args, **kwargs)

    def _lag_q_fns(self):
        def lag_q(obs, acts):
            return self.q_net(obs).gather(1, acts).squeeze(1)
        return [lag_q]

    def _fhr_target_lag_q_fns(self):
        def lag_q(obs, acts):
            return self.q_net_target(obs).gather(1, acts).squeeze(1)
        return [lag_q]

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # DQN.train (sb3 2.9.0) verbatim, with the FHR penalty riding the
        # TD batch and the FHRDQNAgent NaN guard.
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        if self._vec_normalize_env is not None:
            raise RuntimeError("FHRDQN does not support VecNormalize — lag "
                               "observations would bypass the normalisation")

        losses = []
        step_diags = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with torch.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            lam = self._lambda_eff()
            diag = self._fhr_base_diag(td_loss=losses[-1], lam=lam)
            probe = self._fhr_grad_probe_due()
            if self.fhr_weight > 0 or probe:
                def lag_q(obs, acts):
                    return self.q_net(obs).gather(1, acts).squeeze(1)
                penalty = self._fhr_penalty_multi(
                    [current_q_values.squeeze(1)], [lag_q], lam, diag,
                    need_grad=probe)
                if probe and penalty is not None:
                    self._fhr_grad_probe(loss, penalty, lam, diag)
                if self.fhr_weight > 0 and penalty is not None and lam > 0:
                    loss = loss + lam * penalty
            if self._fhr_window_rank_due():
                self._fhr_window_rank_probe(self._lag_q_fns())
            step_diags.append(diag)
            self._fhr_grad_steps += 1

            if not torch.isfinite(loss):
                self.nan_skips += 1
                diag["nan_skips"] = self.nan_skips
                continue
            self.policy.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            if not torch.isfinite(grad_norm):
                self.nan_skips += 1
                diag["nan_skips"] = self.nan_skips
                self.policy.optimizer.zero_grad()
                continue
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self._fhr_aggregate_pending(step_diags)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


if QRDQN is not None:
    from sb3_contrib.common.utils import quantile_huber_loss

    class FHRQRDQN(_FHRMixin, QRDQN):
        """sb3_contrib QR-DQN + the FHR recurrence penalty on the quantile-mean
        Q-values. fhr_weight=0 is bit-for-bit stock QR-DQN."""

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
                     fhr_lag_source: str = "online",
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
                                 fhr_lag_source=fhr_lag_source,
                                 grad_probe_every=grad_probe_every)
            kwargs.setdefault("replay_buffer_class", FHREpisodicReplayBuffer)
            super().__init__(*args, **kwargs)

        def _lag_q_fns(self):
            def lag_q(obs, acts):
                quantiles = self.quantile_net(obs)
                idx = acts[..., None].expand(-1, self.n_quantiles, 1)
                return quantiles.gather(dim=2, index=idx).squeeze(dim=2).mean(dim=1)
            return [lag_q]

        def _fhr_target_lag_q_fns(self):
            def lag_q(obs, acts):
                quantiles = self.quantile_net_target(obs)
                idx = acts[..., None].expand(-1, self.n_quantiles, 1)
                return quantiles.gather(dim=2, index=idx).squeeze(dim=2).mean(dim=1)
            return [lag_q]

        def train(self, gradient_steps: int, batch_size: int = 100) -> None:
            # QRDQN.train (sb3_contrib 2.9.0) verbatim + FHR penalty on the
            # quantile means, with the FHRDQNAgent NaN guard.
            self.policy.set_training_mode(True)
            self._update_learning_rate(self.policy.optimizer)
            if self._vec_normalize_env is not None:
                raise RuntimeError("FHRQRDQN does not support VecNormalize — "
                                   "lag observations would bypass the normalisation")

            losses = []
            step_diags = []
            for _ in range(gradient_steps):
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

                with torch.no_grad():
                    next_quantiles = self.quantile_net_target(replay_data.next_observations)
                    next_greedy_actions = next_quantiles.mean(dim=1, keepdim=True).argmax(dim=2, keepdim=True)
                    next_greedy_actions = next_greedy_actions.expand(batch_size, self.n_quantiles, 1)
                    next_quantiles = next_quantiles.gather(dim=2, index=next_greedy_actions).squeeze(dim=2)
                    target_quantiles = replay_data.rewards + (1 - replay_data.dones) * discounts * next_quantiles

                current_quantiles = self.quantile_net(replay_data.observations)
                actions = replay_data.actions[..., None].long().expand(batch_size, self.n_quantiles, 1)
                current_quantiles = torch.gather(current_quantiles, dim=2, index=actions).squeeze(dim=2)

                loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=True)
                losses.append(loss.item())

                lam = self._lambda_eff()
                diag = self._fhr_base_diag(td_loss=losses[-1], lam=lam)
                probe = self._fhr_grad_probe_due()
                if self.fhr_weight > 0 or probe:
                    def lag_q(obs, acts):
                        quantiles = self.quantile_net(obs)
                        idx = acts[..., None].expand(-1, self.n_quantiles, 1)
                        return quantiles.gather(dim=2, index=idx).squeeze(dim=2).mean(dim=1)
                    penalty = self._fhr_penalty_multi(
                        [current_quantiles.mean(dim=1)], [lag_q], lam, diag,
                        need_grad=probe)
                    if probe and penalty is not None:
                        self._fhr_grad_probe(loss, penalty, lam, diag)
                    if self.fhr_weight > 0 and penalty is not None and lam > 0:
                        loss = loss + lam * penalty
                if self._fhr_window_rank_due():
                    self._fhr_window_rank_probe(self._lag_q_fns())
                step_diags.append(diag)
                self._fhr_grad_steps += 1

                if not torch.isfinite(loss):
                    self.nan_skips += 1
                    diag["nan_skips"] = self.nan_skips
                    continue
                self.policy.optimizer.zero_grad()
                loss.backward()
                grad_norm = None
                if self.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                if grad_norm is not None and not torch.isfinite(grad_norm):
                    self.nan_skips += 1
                    diag["nan_skips"] = self.nan_skips
                    self.policy.optimizer.zero_grad()
                    continue
                self.policy.optimizer.step()

            self._n_updates += gradient_steps
            self._fhr_aggregate_pending(step_diags)

            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/loss", np.mean(losses))


class QuantileMeanQNet(nn.Module):
    """quantile_net wrapped to the (B, n_actions) Q-row surface the repo's
    analyses expect: the mean over the quantile dimension."""

    def __init__(self, quantile_net: nn.Module):
        super().__init__()
        self.quantile_net = quantile_net

    def forward(self, x):
        return self.quantile_net(x).mean(dim=1)


class SB3QAgentAdapter:
    """The QAgent surface (device, policy_net, pi, act_greedy, save) that the
    repo's analysis stack expects, over a trained/training SB3 model. All Q
    evaluations use the online network: q_net for DQN, the quantile mean of
    quantile_net for QR-DQN.

    epsilon=None uses the model's current exploration_rate for pi();
    set adapter.epsilon = 0.0 for greedy post-training rollouts.
    """

    def __init__(self, model, epsilon: float | None = None):
        self.model = model
        self.epsilon = epsilon
        if hasattr(model, "quantile_net"):
            self.policy_net = QuantileMeanQNet(model.quantile_net)
        else:
            self.policy_net = model.q_net

    @property
    def device(self):
        return self.model.device

    def act_greedy(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            return int(self.policy_net(state).argmax(dim=1).item())

    def pi(self, state: np.ndarray) -> int:
        eps = self.model.exploration_rate if self.epsilon is None else self.epsilon
        if np.random.rand() < eps:
            return int(self.model.action_space.sample())
        state_t = torch.tensor(np.asarray(state), dtype=torch.float32,
                               device=self.device).unsqueeze(0)
        return self.act_greedy(state_t)

    def save(self, path):
        # SB3 zip archive (despite the .pt name RunLogger uses); reload with
        # FHRDQN.load / FHRQRDQN.load
        self.model.save(str(path))


class BoundedObservations(gym.Wrapper):
    """Override an env's observation space with finite bounds (observations
    pass through unchanged). q_matrix_dqn builds its state grid from
    observation_space.low/high, which is infinite on CartPole's velocity dims;
    the bounds here mirror the normalise-wrapper ranges the classic-control
    configs use, so Q-matrix grids stay comparable across code paths."""

    def __init__(self, env, low, high):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.asarray(low, dtype=np.float32),
            high=np.asarray(high, dtype=np.float32),
            dtype=np.float32)


class GreedyEvalCallback(BaseCallback):
    """Periodic deterministic-policy evaluation on a dedicated env, appended to
    <run_dir>/eval.csv (env_steps, mean/std/min/max_reward, n_episodes).

    Training-episode rewards are eps-greedy (exploration_final_eps stays >= 0.05
    on the classic-control configs), which understates and blurs the policy's
    actual progress — exactly where "when does learning start" is measured.
    Episode k of every tick resets with seed `seed + k`: the same start states
    every tick and across arms, so curves are paired. Actions come from
    model.predict(deterministic=True) — no global np.random draw anywhere, so
    the training stream is identical with and without this callback.
    """

    def __init__(self, eval_env: gym.Env, out_dir, freq_steps: int = 5000,
                 n_episodes: int = 10, seed: int = 9000,
                 max_episode_steps: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.out_path = pathlib.Path(out_dir) / "eval.csv"
        self.freq_steps = freq_steps
        self.n_episodes = n_episodes
        self.seed = seed
        self.max_episode_steps = max_episode_steps   # backstop for envs without TimeLimit
        self.rows: list[dict] = []
        self._next_eval = 0          # first tick fires on step 1: untrained net

    def _evaluate(self):
        rewards = []
        for k in range(self.n_episodes):
            obs, _ = self.eval_env.reset(seed=self.seed + k)
            total, steps, done = 0.0, 0, False
            while not done and steps < self.max_episode_steps:
                action, _ = self.model.predict(obs, deterministic=True)
                if isinstance(self.eval_env.action_space, gym.spaces.Discrete):
                    action = int(action)
                obs, r, term, trunc, _ = self.eval_env.step(action)
                total += float(r)
                steps += 1
                done = term or trunc
            rewards.append(total)
        r = np.asarray(rewards)
        self.rows.append({"env_steps": self.num_timesteps,
                          "mean_reward": float(r.mean()),
                          "std_reward": float(r.std()),
                          "min_reward": float(r.min()),
                          "max_reward": float(r.max()),
                          "n_episodes": len(r)})
        with open(self.out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._evaluate()
            self._next_eval = self.num_timesteps + self.freq_steps
        return True

    def _on_training_end(self) -> None:
        self._evaluate()


class FHRSB3Callback(BaseCallback):
    """Drives the repo's per-run artifact contract during SB3 training:

      * rewards.csv (episode,reward,steps — full rewrite each tick)
      * train_diagnostics.csv (one nanmean row per train() burst, FHR metric
        names — drained from the model at every episode end)
      * checkpoints/{latest,best,final}.pt (SB3 zip archives)
      * every analysis.ep_freq episodes: training.run_analysis_tick on a
        dedicated analysis env (Q-matrix rank -> rank_stats.csv + figures,
        Hankel sweep -> hankel_sweep.csv + trajectories, AR value probe ->
        autoregressive_*.csv + rollouts)
      * optional early stopping at rolling-mean >= solved_reward, mirroring
        the classic dqn_training_loop protocol

    Requires the training env to be Monitor-wrapped (SB3 does this by
    default) and n_envs == 1.
    """

    def __init__(self, run_logger=None, analysis_config: dict | None = None,
                 analysis_env: gym.Env | None = None,
                 training_config: dict | None = None, verbose: int = 0):
        super().__init__(verbose)
        self.run_logger = run_logger
        self.analysis_config = analysis_config or {}
        self.analysis_env = analysis_env
        self.training_config = training_config or {}
        self.episode_rewards: list[float] = []
        self.episode_steps: list[int] = []
        self._best_window_mean = -np.inf
        self._stop = False

    def _adapter(self):
        # models that define their own analysis adapter (the SAC family)
        # provide qagent_adapter(); the DQN family uses the default
        make = getattr(self.model, "qagent_adapter", None)
        return make() if make is not None else SB3QAgentAdapter(self.model)

    def _log_rewards(self):
        if self.run_logger is not None and self.episode_rewards:
            self.run_logger.log_rewards(self.episode_rewards,
                                        steps=self.episode_steps)

    def _drain_diagnostics(self, episode: int):
        if self.run_logger is None or not hasattr(self.model, "drain_diagnostics"):
            return
        for row in self.model.drain_diagnostics():
            self.run_logger.log_train_diagnostics(episode, **row)
        self._drain_window_rank()

    def _drain_window_rank(self):
        if self.run_logger is None or not hasattr(self.model, "drain_window_rank"):
            return
        rows, arrays = self.model.drain_window_rank()
        if rows:
            self.run_logger.log_window_hankel(rows)
        if arrays:
            self.run_logger.save_window_matrices(arrays)

    def _analysis_tick(self, episode: int):
        if self.analysis_env is None:
            return
        from training import run_analysis_tick   # lazy: needs src/ on sys.path
        run_analysis_tick(self._adapter(), self.analysis_env,
                          self.analysis_config, self.run_logger, episode)
        # the Hankel/AR rollouts leave the analysis env terminated
        self.analysis_env.reset()

    def _on_rollout_end(self) -> None:
        # fires right before every train() burst: tag the burst's penalty
        # residuals with the episode currently in progress, so the lambda
        # ramp-down consumes per-episode buckets (see notify_episode_end)
        if hasattr(self.model, "_fhr_episode_hint"):
            self.model._fhr_episode_hint = len(self.episode_rewards)

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if not done:
                continue
            ep = info.get("episode")
            if ep is None:      # Monitor wrapper missing — nothing to log
                continue
            episode = len(self.episode_rewards)          # 0-based, as classic
            self.episode_rewards.append(float(ep["r"]))
            self.episode_steps.append(int(ep["l"]))
            if hasattr(self.model, "notify_episode_end"):
                self.model.notify_episode_end(episode, float(ep["r"]))
            self._drain_diagnostics(episode)

            # best-checkpoint gate on the classic cadence: evaluated every
            # no_eps_to_avg episodes, short window allowed at the start
            window = self.training_config.get("no_eps_to_avg", 10)
            if window and episode % window == 0:
                mean = float(np.mean(self.episode_rewards[-window:]))
                if mean > self._best_window_mean:
                    self._best_window_mean = mean
                    if self.run_logger is not None:
                        self.run_logger.checkpoint(self._adapter(), "best")

            ep_freq = self.analysis_config.get("ep_freq")
            if ep_freq and episode % ep_freq == 0:
                self._log_rewards()
                if self.run_logger is not None:
                    self.run_logger.checkpoint(self._adapter(), "latest")
                self._analysis_tick(episode)

            # early stop on the classic dqn_training_loop gate: strictly
            # greater, first evaluated once episode > patience
            solved = self.training_config.get("solved_reward")
            patience = self.training_config.get("early_stopping_patience_eps")
            if (solved is not None and patience
                    and episode > patience
                    and float(np.mean(self.episode_rewards[-patience:])) > solved):
                print(f"early stop at episode {episode}: rolling-{patience} "
                      f"mean > {solved}")
                self._stop = True
        return not self._stop

    def _on_training_end(self) -> None:
        self._log_rewards()
        self._drain_diagnostics(max(len(self.episode_rewards) - 1, 0))
        if self.run_logger is not None:
            self.run_logger.checkpoint(self._adapter(), "final")
