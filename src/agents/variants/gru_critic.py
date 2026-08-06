"""GRU critic — ablation for ssm_critic: same trajectory-recurrent critic
shape, but a nonlinear GRU recurrence instead of the diagonal LINEAR one.
Outputs need not satisfy any low-order linear recurrence, so comparing this
arm against ssm_critic separates 'low-rank linear prior' from 'recurrence
of any kind'."""
import torch
import torch.nn as nn

from agents.variants.ssm_critic import (PPO_KEYS, SSMCriticAgent,
                                        collect_rollout)  # noqa: F401 (reused)


class _GRUCritic(nn.Module):
    def __init__(self, obs_dim, r, feat=64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, feat), nn.Tanh(),
                                   nn.Linear(feat, feat), nn.Tanh())
        self.cell = nn.GRUCell(feat, r)
        self.C = nn.Linear(r, 1)
        self.D = nn.Linear(feat, 1)

    def step(self, phi, h):
        h = self.cell(phi.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
        return h, (self.C(h) + self.D(phi)).squeeze(-1)


class GRUCriticAgent(SSMCriticAgent):
    def __init__(self, env, device="cpu", ssm_rank=8, **kwargs):
        super().__init__(env, device=device, ssm_rank=ssm_rank, **kwargs)
        self.critic = _GRUCritic(env.observation_space.shape[0],
                                 ssm_rank).to(self.device)
        self.optim = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.optim.param_groups[0]["lr"])

    def _seq_values(self, obs, seg_bounds, seg_h0):
        phi = self.critic.trunk(obs)
        hs = []
        for (s, e), h in zip(seg_bounds, seg_h0):
            hseq = h.unsqueeze(0)
            for t in range(s, e):
                hseq = self.critic.cell(phi[t].unsqueeze(0), hseq)
                hs.append(hseq.squeeze(0))
        return (self.critic.C(torch.stack(hs)) + self.critic.D(phi)).squeeze(-1)


def build(env, device, overrides):
    kwargs = {k: v for k, v in overrides.items() if k in PPO_KEYS}
    return GRUCriticAgent(env=env, device=device,
                          ssm_rank=int(overrides.get("ssm_rank", 8)), **kwargs)
