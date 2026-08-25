"""Prioritized replay over the episodic buffer — the missing piece behind the
`prioritized_replay` config knob (EfficientRainbowAgent has reserved it since
v1; SACDiscreteAgent is the first consumer).

Design: EpisodicReplayBuffer's storage (whole episodes, per-step entries — the
FHR predecessor/reward-lag contracts live on it) is kept untouched; a SumTree
(rainbow_agent.py conventions: proportional priorities, alpha exponent,
max-priority insert, stratified segment sampling, IS weights normalised by
their max, beta annealed toward 1 per sample call) is laid over it by
UNWRAPPED transition position: the k-th transition ever appended owns leaf
k % capacity. Whole-episode eviction zeroes the evicted episode's leaf range;
eviction happens BEFORE the incoming transition claims its leaf, so a leaf is
never shared between the evictee and the newcomer. Handles stay the parent's
(ep_idx, t) pairs, so gather_predecessors — and therefore the FHR penalty —
work identically under prioritized sampling, and update_priorities() fills
the parent's stub signature (handles double as priority keys).

The uniform samplers are inherited untouched: a `prioritized_replay: false`
agent constructs the parent class and its RNG stream is byte-identical to
before this module existed.

Corner the parent tolerates but PER cannot: a single episode longer than
capacity would wrap the ring onto itself, so append() raises instead
(unreachable on Atari-100k — the ALE episode cap is ~27k steps vs the 100k
buffer)."""
import bisect

import numpy as np
import torch

from .hankel_dqn_agent import EpisodicReplayBuffer
from .rainbow_agent import SumTree


class PrioritizedEpisodicReplayBuffer(EpisodicReplayBuffer):
    def __init__(self, capacity: int, alpha: float = 0.5,
                 beta_start: float = 0.4, beta_increment: float = 3e-6,
                 eps: float = 1e-5):
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = beta_increment
        self.eps = eps
        self.max_priority = 1.0
        self.tree = SumTree(capacity)      # .data unused: slots ARE positions
        self._head_pos = 0                 # unwrapped: total appended ever
        self._tail_pos = 0                 # unwrapped: oldest live transition
        self._open_base_pos = 0            # unwrapped start of the open episode
        self._bases: list[int] = []        # base_pos per closed episode (sorted)

    # ---------------------------------------------------------- storage
    def append(self, state, action, reward, next_state) -> None:
        # pre-evict so the incoming transition's leaf is never the evictee's
        while (self._closed_len + len(self._open_states) + 1 > self.capacity
               and self._episodes):
            self._pop_oldest()
        if self._head_pos - self._tail_pos >= self.capacity:
            raise RuntimeError(
                f"a single episode longer than capacity={self.capacity} would "
                "wrap the priority ring onto itself — PER needs capacity > "
                "max episode length")
        leaf = self._head_pos % self.capacity + self.capacity - 1
        self.tree.update(leaf, self.max_priority ** self.alpha)
        self._head_pos += 1
        super().append(state, action, reward, next_state)

    def close(self, terminated: bool) -> None:
        had_open = bool(self._open_states)
        super().close(terminated)
        if had_open:
            self._episodes[-1]["base_pos"] = self._open_base_pos
            self._bases.append(self._open_base_pos)
            self._open_base_pos = self._head_pos

    def _pop_oldest(self) -> dict:
        ep = super()._pop_oldest()
        base = ep["base_pos"]
        for pos in range(base, base + len(ep["states"])):
            self.tree.update(pos % self.capacity + self.capacity - 1, 0.0)
        self._tail_pos = base + len(ep["states"])
        del self._bases[0]
        return ep

    # ---------------------------------------------------------- mapping
    def _pos_to_handle(self, pos: int):
        """Unwrapped position -> the parent's (ep_idx, t) handle."""
        if pos >= self._open_base_pos:
            return len(self._episodes), pos - self._open_base_pos
        i = bisect.bisect_right(self._bases, pos) - 1
        return i, pos - self._bases[i]

    def _handle_to_leaf(self, ep_idx: int, t: int) -> int:
        base = (self._open_base_pos if ep_idx == len(self._episodes)
                else self._episodes[ep_idx]["base_pos"])
        return (base + t) % self.capacity + self.capacity - 1

    # ---------------------------------------------------------- sampling
    def sample_nstep_prioritized(self, batch_size: int, n_step: int,
                                 gamma: float):
        """Proportional stratified draw (with replacement) of n-step
        transitions. Returns (states, actions, returns, next_states,
        discounts, handles, weights) — the uniform sample_nstep_transitions
        tuple plus the IS weights; handles are the priority keys for
        update_priorities()."""
        total = self.tree.total()
        n_live = len(self)
        if total <= 0 or n_live == 0:
            raise RuntimeError("cannot sample from an empty prioritized buffer")
        segment = total / batch_size
        handles, probs = [], []
        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            leaf, priority, _ = self.tree.get(s)
            pos = leaf - (self.capacity - 1)
            pos = self._tail_pos + (pos - self._tail_pos) % self.capacity
            if priority <= 0.0 or not (self._tail_pos <= pos < self._head_pos):
                # float roundoff landed on a dead/unwritten leaf — redraw
                # uniformly over the live span
                pos = int(np.random.randint(self._tail_pos, self._head_pos))
                priority = self.tree.tree[pos % self.capacity
                                          + self.capacity - 1]
            handles.append(self._pos_to_handle(pos))
            probs.append(max(priority, 1e-12) / total)
        states, actions, returns, nexts, discounts = [], [], [], [], []
        for ep_idx, t in handles:
            s, a, R, nxt, disc = self._nstep_transition(ep_idx, t, n_step,
                                                        gamma)
            states.append(s)
            actions.append(a)
            returns.append(R)
            nexts.append(nxt)
            discounts.append(disc)
        weights = (n_live * np.asarray(probs)) ** (-self.beta)
        weights = weights / weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        return (torch.cat(states), torch.cat(actions), torch.cat(returns),
                nexts, torch.tensor(discounts, dtype=torch.float32), handles,
                torch.tensor(weights, dtype=torch.float32))

    def update_priorities(self, handles, td_errors) -> None:
        """Fill the parent's stub: |error| + eps, alpha-exponentiated, keyed
        by the (ep_idx, t) handles the prioritized sampler returned."""
        errs = np.abs(np.asarray(
            [float(e) for e in td_errors], dtype=np.float64)) + self.eps
        self.max_priority = max(self.max_priority, float(errs.max()))
        for (ep_idx, t), err in zip(handles, errs):
            self.tree.update(self._handle_to_leaf(ep_idx, t),
                             float(err) ** self.alpha)
