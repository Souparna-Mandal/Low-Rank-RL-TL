"""Deep Q-Network with experience replay and soft target updates."""

from __future__ import annotations

import math
import pathlib
import random
from collections import deque, namedtuple
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from low_rank_rl.agents.base import BaseAgent

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self._memory: deque[Transition] = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward) -> None:
        self._memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self._memory, batch_size)

    def __len__(self) -> int:
        return len(self._memory)


class QNetwork(nn.Module):
    def __init__(self, n_obs: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent(BaseAgent):
    def __init__(
        self,
        n_obs: int,
        n_actions: int,
        hidden: int = 128,
        batch_size: int = 128,
        gamma: float = 0.99,
        lr: float = 3e-4,
        tau: float = 0.005,
        eps_start: float = 0.9,
        eps_end: float = 0.01,
        eps_decay: int = 2500,
        buffer_capacity: int = 10_000,
        device: Optional[str] = None,
    ):
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = torch.device(device)

        self.policy_net = QNetwork(n_obs, n_actions, hidden).to(self.device)
        self.target_net = QNetwork(n_obs, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.buffer = ReplayBuffer(buffer_capacity)
        self._steps_done = 0

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training:
            eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
                -self._steps_done / self.eps_decay
            )
            self._steps_done += 1
            if random.random() < eps:
                return random.randrange(self.n_actions)

        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.policy_net(s).argmax(dim=1).item())

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: Optional[np.ndarray],
        done: bool,
    ) -> dict[str, float]:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = torch.tensor([[action]], dtype=torch.long, device=self.device)
        r = torch.tensor([reward], dtype=torch.float32, device=self.device)
        ns = (
            torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            if next_state is not None else None
        )
        self.buffer.push(s, a, ns, r)

        if len(self.buffer) < self.batch_size:
            return {}
        return self._gradient_step()

    def q_matrix(self, states: np.ndarray) -> np.ndarray:
        self.policy_net.eval()
        with torch.no_grad():
            s = torch.tensor(states, dtype=torch.float32, device=self.device)
            q = self.policy_net(s).cpu().numpy()
        self.policy_net.train()
        return q.astype(np.float64)

    def save(self, path: str | pathlib.Path) -> None:
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self._steps_done,
        }, path)

    def load(self, path: str | pathlib.Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._steps_done = ckpt.get("steps_done", 0)

    @property
    def epsilon(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -self._steps_done / self.eps_decay
        )

    def _gradient_step(self) -> dict[str, float]:
        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            [s is not None for s in batch.next_state],
            device=self.device, dtype=torch.bool,
        )
        non_final_next = torch.cat([s for s in batch.next_state if s is not None])
        state_batch  = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        q_sa = self.policy_net(state_batch).gather(1, action_batch)

        next_v = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_v[non_final_mask] = self.target_net(non_final_next).max(1).values

        target = (next_v * self.gamma) + reward_batch
        loss = F.smooth_l1_loss(q_sa, target.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        for p, tp in zip(self.policy_net.parameters(), self.target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return {"loss": float(loss.item()), "epsilon": self.epsilon}

    def __repr__(self) -> str:
        return (
            f"DQNAgent(n_obs={self.n_obs}, n_actions={self.n_actions}, "
            f"device={self.device}, eps={self.epsilon:.3f})"
        )
