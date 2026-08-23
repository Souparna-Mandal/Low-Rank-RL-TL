"""Atari Q-networks shared by runners and notebooks.

The Nature-DQN CNN (Mnih et al. 2015) that the per-game exp1 notebooks define
inline lives here too, so subprocess launchers (e.g. the Atari-100k FHR-DQN
runner) can build byte-identical agents without importing notebook code. Keep
this class in sync with the notebook copies: same architecture, same /255
normalisation in forward, so checkpoints are interchangeable.
"""
import torch
import torch.nn as nn


class NatureCNN(nn.Module):
    """Maps a (C, 84, 84) uint8 frame stack to Q-values of shape (n_actions,).

    Mnih et al. (2015) architecture: Conv(C->32, k8 s4) -> Conv(32->64, k4 s2)
    -> Conv(64->64, k3 s1) -> Linear(3136, fc_hidden) -> Linear(fc_hidden, A).
    Observations stay uint8 in the replay buffer to save memory; the forward
    pass casts to float and divides by 255.
    """

    def __init__(self, in_channels: int, n_actions: int, fc_hidden: int = 512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),          nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),          nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            flat_dim = self.features(dummy).shape[1]
        self.head = nn.Sequential(
            nn.Linear(flat_dim, fc_hidden), nn.ReLU(),
            nn.Linear(fc_hidden, n_actions),
        )

    def forward(self, x):
        x = x.float() / 255.0
        return self.head(self.features(x))
