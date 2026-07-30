import random

import numpy as np
import torch

from .q_agent import QAgent
from .hankel_regulariser import HankelRankPenalty


class EpisodicReplayBuffer:
    """Replay buffer whose storage unit is the episode, so both i.i.d. 1-step
    transitions (classical TD) and episode-contiguous windows (Hankel penalty)
    can be sampled from the same data. Capacity is counted in transitions with
    FIFO eviction of whole episodes (a single episode longer than capacity is
    kept whole, so len can exceed capacity in that corner). The in-progress
    episode is sampleable for transitions (matching ReplayBuffer, which sees
    every pushed step at once) but not for windows."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._episodes: list[dict] = []
        self._closed_len = 0
        self._open_states, self._open_actions, self._open_rewards = [], [], []
        self._pending_next = None

    def append(self, state, action, reward, next_state) -> None:
        """Add one transition to the in-progress episode. next_state is None on
        termination (QAgent convention)."""
        self._open_states.append(state)
        self._open_actions.append(action)
        self._open_rewards.append(reward)
        self._pending_next = next_state
        while self._closed_len + len(self._open_states) > self.capacity and self._episodes:
            self._closed_len -= len(self._episodes.pop(0)["states"])

    def close(self, terminated: bool) -> None:
        """Seal the in-progress episode (call on terminated OR truncated)."""
        if not self._open_states:
            return
        self._episodes.append({
            "states": torch.cat(self._open_states),
            "actions": torch.cat(self._open_actions),
            "rewards": torch.cat(self._open_rewards),
            "final_next_state": self._pending_next,
            "terminated": terminated,
        })
        self._closed_len += len(self._open_states)
        self._open_states, self._open_actions, self._open_rewards = [], [], []
        self._pending_next = None
        while self._closed_len > self.capacity and len(self._episodes) > 1:
            self._closed_len -= len(self._episodes.pop(0)["states"])

    def __len__(self) -> int:
        return self._closed_len + len(self._open_states)

    def _transition(self, ep_idx: int, t: int):
        """(state, action, reward, next_state_or_None) at flat position (ep, t)."""
        if ep_idx == len(self._episodes):  # open episode
            state = self._open_states[t]
            nxt = (self._open_states[t + 1] if t + 1 < len(self._open_states)
                   else self._pending_next)
            return state, self._open_actions[t], self._open_rewards[t], nxt
        ep = self._episodes[ep_idx]
        last = len(ep["states"]) - 1
        nxt = ep["states"][t + 1:t + 2] if t < last else ep["final_next_state"]
        return ep["states"][t:t + 1], ep["actions"][t:t + 1], ep["rewards"][t:t + 1], nxt

    def sample_transitions(self, batch_size: int):
        """Uniform without replacement over all stored transitions — same
        distribution as ReplayBuffer.sample. Returns (states, actions, rewards,
        next_states) with next_states a list of (1, obs) tensors or None."""
        lengths = [len(ep["states"]) for ep in self._episodes] + [len(self._open_states)]
        cum = np.cumsum([0] + lengths)
        flat = random.sample(range(cum[-1]), batch_size)
        states, actions, rewards, nexts = [], [], [], []
        for f in flat:
            ep_idx = int(np.searchsorted(cum, f, side="right")) - 1
            s, a, r, nxt = self._transition(ep_idx, f - cum[ep_idx])
            states.append(s); actions.append(a); rewards.append(r); nexts.append(nxt)
        return torch.cat(states), torch.cat(actions), torch.cat(rewards), nexts

    def sample_windows(self, n_windows: int, window_len: int, exclude_terminal: bool = True,
                       with_next: bool = False, half_life: float | None = None):
        """n_windows fixed-length contiguous (states, actions) windows, uniform
        with replacement over all valid single-episode windows. With
        exclude_terminal, windows ending on a terminal step are not eligible.
        Returns (states (n, T, obs), actions (n, T), rewards (n, T)) or None
        if no episode is long enough. with_next appends a list of each window's
        post-window state — (1, obs), or None when the window ends on a
        terminal step — so callers can form all T TD pairs. half_life (in
        episodes) recency-biases episode choice by 0.5**(age/half_life), so
        windows come from behaviour close to the current policy."""
        eligible = []
        for idx, ep in enumerate(self._episodes):
            n_starts = len(ep["states"]) - window_len + 1
            if exclude_terminal and ep["terminated"]:
                n_starts -= 1
            if n_starts > 0:
                eligible.append((ep, n_starts, idx))
        if not eligible:
            return None
        newest = eligible[-1][2]
        weights = [n * (0.5 ** ((newest - idx) / half_life) if half_life else 1.0)
                   for _, n, idx in eligible]
        cum = np.cumsum([0.0] + weights)
        states, actions, rewards, last_nexts = [], [], [], []
        for f in (random.random() * cum[-1] for _ in range(n_windows)):
            ep_idx = min(int(np.searchsorted(cum, f, side="right")) - 1, len(eligible) - 1)
            ep, n_starts, _ = eligible[ep_idx]
            start = random.randrange(n_starts)
            end = start + window_len
            states.append(ep["states"][start:end])
            actions.append(ep["actions"][start:end])
            rewards.append(ep["rewards"][start:end])
            if with_next:
                last_nexts.append(ep["states"][end:end + 1] if end < len(ep["states"])
                                  else ep["final_next_state"])
        out = (torch.stack(states), torch.stack(actions), torch.stack(rewards))
        return out + (last_nexts,) if with_next else out


class HankelDQNAgent(QAgent):
    """HR-DQN: classical (Double-)DQN plus a truncated-nuclear-norm penalty on
    Hankel matrices of predicted Q-values along replayed sub-trajectories.
    hankel_weight=0 reproduces QAgent training exactly (same TD computation,
    same sampling distribution, no penalty machinery touched).

    Extra kwargs over QAgent (all mappable to config agent: keys):
        hankel_weight: λ. hankel_order: target rank r. window_len/n_windows:
        penalty batch shape. gate_threshold: skip windows with relative tail
        energy above this (None = no gate). warmup/ramp/decay_grad_steps: λ is
        0 for the first warmup steps, ramps linearly over ramp steps, and/or
        decays linearly to 0 over decay steps.
        hankel_signal: "q" penalises Q(s_t,a_t) along the window, "v" penalises
        max_a Q(s_t,·) — the sequence that drives the greedy policy.
        penalize_terminal_windows: allow windows ending on a terminal step.
        td_source: "iid" (default) or "windows" (TD pairs taken from inside the
        penalty windows — correlated batches, kept for ablation).
        window_half_life: recency-bias the penalty windows (in episodes; None =
        uniform over the buffer) so they come from near-current behaviour.
    """

    def __init__(self, *, hankel_weight: float = 0.0, hankel_order: int = 2,
                 window_len: int = 16, n_windows: int = 8,
                 gate_threshold: float | None = None, warmup_grad_steps: int = 0,
                 ramp_grad_steps: int = 0, decay_grad_steps: int = 0,
                 penalize_terminal_windows: bool = False,
                 td_source: str = "iid", hankel_signal: str = "q",
                 hankel_jitter: float = 0.0, td_gate_scale: float | None = None,
                 window_half_life: float | None = None, **q_agent_kwargs):
        super().__init__(**q_agent_kwargs)
        assert td_source in ("iid", "windows")
        assert hankel_signal in ("q", "v")
        self.replay_buffer = EpisodicReplayBuffer(q_agent_kwargs["replay_buffer_capacity"])
        self.hankel_weight = hankel_weight
        self.window_len = window_len
        self.n_windows = n_windows
        self.penalize_terminal_windows = penalize_terminal_windows
        self.td_source = td_source
        self.window_half_life = window_half_life
        self.warmup_grad_steps = warmup_grad_steps
        self.ramp_grad_steps = ramp_grad_steps
        self.decay_grad_steps = decay_grad_steps
        self.hankel_signal = hankel_signal
        self.td_gate_scale = td_gate_scale
        self.hankel_penalty = HankelRankPenalty(hankel_order, gate_threshold,
                                                jitter=hankel_jitter)
        self._grad_steps = 0
        self.nan_skips = 0

    def update_buffer(self, state, action, reward, next_state, terminated, truncated=False):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = torch.tensor([action], dtype=torch.long, device=self.device)
        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        next_state = (None if terminated else
                      torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0))
        self.replay_buffer.append(state, action, reward, next_state)
        if terminated or truncated:
            self.replay_buffer.close(terminated)

    def update_buffer_atari(self, state, action, reward, next_state, terminated, truncated=False):
        raise NotImplementedError(
            "HankelDQNAgent's episodic buffer has no atari (uint8/CPU) path yet")

    def _lambda_eff(self) -> float:
        k = self._grad_steps
        if k < self.warmup_grad_steps:
            return 0.0
        k -= self.warmup_grad_steps
        factor = 1.0
        if self.ramp_grad_steps > 0:
            factor *= min(1.0, k / self.ramp_grad_steps)
        if self.decay_grad_steps > 0:
            factor *= max(0.0, 1.0 - k / self.decay_grad_steps)
        return self.hankel_weight * factor

    def _td_batch(self):
        """(states, actions, rewards, next_states-list) for the TD loss."""
        if self.td_source == "windows":
            # TD sees everything, including terminal-ending windows — otherwise
            # the non-bootstrapped target=r anchor never enters the loss.
            win = self.replay_buffer.sample_windows(
                self.n_windows, self.window_len, exclude_terminal=False, with_next=True)
            if win is not None:
                w_states, w_actions, w_rewards, last_nexts = win
                n, T = w_actions.shape
                states = w_states.reshape(n * T, -1)
                actions = w_actions.reshape(-1)
                rewards = w_rewards.reshape(-1)
                nexts = []
                for b in range(n):
                    nexts += list(w_states[b, 1:].unsqueeze(1))
                    nexts.append(last_nexts[b])  # None iff window ends at a terminal
                return states, actions, rewards, nexts
        return self.replay_buffer.sample_transitions(self.batch_size)

    def _train_step(self):
        states, actions, rewards, next_list = self._td_batch()
        non_final_mask = torch.tensor([s is not None for s in next_list], device=self.device)

        Q_s_a = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        Q_s_1_a = torch.zeros(len(next_list), device=self.device)
        if non_final_mask.any():
            next_state = torch.cat([s for s in next_list if s is not None]).to(self.device)
            with torch.no_grad():
                if self.double:
                    next_actions = self.policy_net(next_state).argmax(dim=1, keepdim=True)
                    Q_s_1_a[non_final_mask] = self.target_net(next_state).gather(1, next_actions).squeeze(1)
                else:
                    Q_s_1_a[non_final_mask] = self.target_net(next_state).max(dim=1).values
        loss = self.loss(Q_s_a, rewards, Q_s_1_a)

        lam = self._lambda_eff()
        diag = {"td_loss": float(loss.detach()), "lambda_eff": lam,
                "penalty_raw": np.nan, "penalty_weighted": 0.0, "gate_frac": np.nan,
                "converged_frac": np.nan, "ext_gate_frac": np.nan,
                "batch_eff_rank": np.nan, "rel_tail": np.nan,
                "nan_skips": self.nan_skips}
        if self.hankel_weight > 0:
            win = self.replay_buffer.sample_windows(
                self.n_windows, self.window_len,
                exclude_terminal=not self.penalize_terminal_windows,
                half_life=self.window_half_life)
            if win is not None:
                w_states, w_actions, w_rewards = win
                n, T = w_actions.shape
                # λ=0 (warm-up): diagnostics only, keep the SVD out of the graph.
                with torch.enable_grad() if lam > 0 else torch.no_grad():
                    out = self.policy_net(w_states.reshape(n * T, -1))
                    seq = (out.max(dim=1).values if self.hankel_signal == "v" else
                           out.gather(1, w_actions.reshape(n * T, 1)).squeeze(1)).view(n, T)
                    keep_mask = None
                    if self.td_gate_scale is not None:
                        # Regularise only windows whose own bootstrap residual is
                        # already small — i.e. where TD locally agrees with Q.
                        with torch.no_grad():
                            q_sa = out.gather(1, w_actions.reshape(n * T, 1)).view(n, T)
                            nxt = w_states[:, 1:].reshape(n * (T - 1), -1)
                            if self.double:
                                na = self.policy_net(nxt).argmax(dim=1, keepdim=True)
                                qn = self.target_net(nxt).gather(1, na).squeeze(1)
                            else:
                                qn = self.target_net(nxt).max(dim=1).values
                            resid = (q_sa[:, :-1] - w_rewards[:, :-1]
                                     - self.loss.gamma * qn.view(n, T - 1)).abs().mean(dim=1)
                            keep_mask = resid <= self.td_gate_scale * float(loss.detach())
                    penalty, pdiag = self.hankel_penalty(seq, keep_mask=keep_mask)
                diag.update(pdiag)
                diag["penalty_weighted"] = lam * pdiag["penalty_raw"]
                if lam > 0:
                    loss = loss + lam * penalty

        self._grad_steps += 1
        if not torch.isfinite(loss):
            self.nan_skips += 1
            diag["nan_skips"] = self.nan_skips
            return diag
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm)
        if not torch.isfinite(grad_norm):
            self.nan_skips += 1
            diag["nan_skips"] = self.nan_skips
            self.optimiser.zero_grad()
            return diag
        self.optimiser.step()
        return diag

    def train(self):
        gd_steps = min(len(self.replay_buffer) // (self.buffer_util * self.batch_size),
                       self.gd_steps_ceil)
        diags = [self._train_step() for _ in range(gd_steps)]
        if not diags:
            return None
        def _nanmean(vals):
            finite = [v for v in vals if not np.isnan(v)]
            return float(np.mean(finite)) if finite else float("nan")
        return {k: _nanmean([d[k] for d in diags]) for k in diags[0]}
