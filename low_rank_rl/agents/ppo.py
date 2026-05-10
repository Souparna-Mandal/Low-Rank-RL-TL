"""Proximal Policy Optimisation (clipped surrogate + GAE).

The agent follows OpenAI Spinning Up's PPO pseudocode
(https://spinningup.openai.com/en/latest/algorithms/ppo.html) and the common
Also this:
(https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py):

- Separate actor and critic MLPs with Tanh activations.
- Generalised Advantage Estimation (GAE-λ) computed backwards with
  ``next_val`` seeded by ``last_value`` to bootstrap truncated episodes.
- Clipped surrogate objective L_CLIP = E[min(rₜ Aₜ, clip(rₜ, 1±ε) Aₜ)].
- Entropy bonus and squared-error value loss combined in a single optimiser
  step over several epochs of random mini-batches.
- Per-rollout advantage normalisation and global-norm gradient clipping at 0.5.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from low_rank_rl.agents.base import BaseAgent


class ActorNetwork(nn.Module):
    def __init__(self, n_obs: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def distribution(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(x))


class CriticNetwork(nn.Module):
    def __init__(self, n_obs: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PPOAgent(BaseAgent):
    def __init__(
        self,
        n_obs: int,
        n_actions: int,
        hidden: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        device: Optional[str] = None,
    ):
        self.n_obs            = n_obs
        self.n_actions        = n_actions
        self.gamma            = gamma
        self.gae_lambda       = gae_lambda
        self.clip_eps         = clip_eps
        self.entropy_coef     = entropy_coef
        self.value_coef       = value_coef
        self.ppo_epochs       = ppo_epochs
        self.mini_batch_size  = mini_batch_size

        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = torch.device(device)

        self.actor     = ActorNetwork(n_obs, n_actions, hidden).to(self.device)
        self.critic    = CriticNetwork(n_obs, hidden).to(self.device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        self._states:    list[np.ndarray] = []
        self._actions:   list[int]        = []
        self._log_probs: list[float]      = []
        self._rewards:   list[float]      = []
        self._dones:     list[bool]       = []
        self._values:    list[float]      = []

    def act(self, state: np.ndarray, training: bool = True) -> int:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist   = self.actor.distribution(s)
            action = dist.sample() if training else dist.probs.argmax(dim=-1)
            self._last_log_prob = float(dist.log_prob(action).item())
            self._last_value    = float(self.critic(s).item())
        return int(action.item())

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: Optional[np.ndarray] = None,
        done: bool = False,
    ) -> dict[str, float]:
        self._states.append(state)
        self._actions.append(action)
        self._log_probs.append(self._last_log_prob)
        self._rewards.append(reward)
        self._dones.append(done)
        self._values.append(self._last_value)
        return {}

    def end_episode(self, last_value: float = 0.0) -> dict[str, float]:
        returns, advantages = self._compute_gae(last_value)
        metrics = self._ppo_update(returns, advantages)
        self._clear_buffer()
        return metrics

    def q_matrix(self, states: np.ndarray) -> np.ndarray:
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            s      = torch.tensor(states, dtype=torch.float32, device=self.device)
            logits = self.actor(s)
            v      = self.critic(s).unsqueeze(1)
            adv    = logits - logits.mean(dim=1, keepdim=True)
            q      = v + adv
        self.actor.train()
        self.critic.train()
        return q.cpu().numpy().astype(np.float64)

    def save(self, path: str | pathlib.Path) -> None:
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str | pathlib.Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

    def _compute_gae(self, last_value: float) -> tuple[np.ndarray, np.ndarray]:
        T          = len(self._rewards)
        advantages = np.zeros(T, dtype=np.float64)
        returns    = np.zeros(T, dtype=np.float64)
        gae        = 0.0
        next_val   = last_value

        for t in reversed(range(T)):
            mask          = 0.0 if self._dones[t] else 1.0
            delta         = self._rewards[t] + self.gamma * next_val * mask - self._values[t]
            gae           = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t]    = gae + self._values[t]
            next_val      = self._values[t]

        return returns, advantages

    def _ppo_update(self, returns: np.ndarray, advantages: np.ndarray) -> dict[str, float]:
        states    = torch.tensor(np.array(self._states), dtype=torch.float32, device=self.device)
        actions   = torch.tensor(self._actions,   dtype=torch.long,    device=self.device)
        old_lp    = torch.tensor(self._log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns,         dtype=torch.float32, device=self.device)
        adv_t     = torch.tensor(advantages,      dtype=torch.float32, device=self.device)
        adv_t     = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        T          = len(states)
        total_loss = 0.0

        for _ in range(self.ppo_epochs):
            for mb in torch.randperm(T).split(self.mini_batch_size):
                dist        = self.actor.distribution(states[mb])
                lp          = dist.log_prob(actions[mb])
                entropy     = dist.entropy().mean()
                values      = self.critic(states[mb])

                ratio       = torch.exp(lp - old_lp[mb])
                surr1       = ratio * adv_t[mb]
                surr2       = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb]
                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(values, returns_t[mb])
                loss        = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), 0.5
                )
                self.optimizer.step()
                total_loss += float(loss.item())

        return {"loss": total_loss, "mean_advantage": float(adv_t.mean().item())}

    def _clear_buffer(self) -> None:
        self._states.clear(); self._actions.clear(); self._log_probs.clear()
        self._rewards.clear(); self._dones.clear();  self._values.clear()

    def __repr__(self) -> str:
        return f"PPOAgent(n_obs={self.n_obs}, n_actions={self.n_actions}, device={self.device})"
