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


class _ImpalaResidualBlock(nn.Module):
    """Impala residual block: x + Conv3x3(ReLU(Conv3x3(ReLU(x))))."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.conv0(torch.relu(x))
        h = self.conv1(torch.relu(h))
        return x + h


class ImpalaCNNEncoder(nn.Module):
    """BBF's encoder (Schwarzer et al. 2023, arXiv:2305.19452): the Impala-CNN
    ResNet (Espeholt et al. 2018) at ``width_scale`` 4 — 3 stages of
    [Conv3x3 -> MaxPool3x3/2 -> 2 residual blocks] over base widths (16, 32,
    32), i.e. channels (64, 128, 128), 15 conv layers total. 84x84 input pools
    to 11x11, so ``feature_dim`` = 128 * 11 * 11 = 15488 at width 4.

    Faithful to the reference ``spr_networks.py``: no layer/batch/spectral
    norm, ReLU only, and (``renormalize=True``, BBF's default) a per-sample
    min-max normalisation of the final feature map to [0, 1] before
    flattening — the network's only normalisation. Input convention matches
    NatureCNNEncoder: uint8-scale frames, /255 happens HERE, so callers that
    augment first must feed 0-255-scaled floats.
    """

    def __init__(self, in_channels: int, width_scale: int = 4,
                 dims: tuple = (16, 32, 32), num_blocks: int = 2,
                 renormalize: bool = True, input_hw: int = 84):
        super().__init__()
        self.renormalize = renormalize
        stages = []
        prev = in_channels
        for base in dims:
            ch = base * width_scale
            stages += [nn.Conv2d(prev, ch, kernel_size=3, padding=1),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
            stages += [_ImpalaResidualBlock(ch) for _ in range(num_blocks)]
            prev = ch
        self.stages = nn.Sequential(*stages)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_hw, input_hw)
            self.feature_dim = int(self.stages(dummy).flatten(1).shape[1])

    def forward(self, x):
        h = torch.relu(self.stages(x.float() / 255.0))
        if self.renormalize:
            flat = h.flatten(1)
            mn = flat.min(dim=1, keepdim=True).values
            mx = flat.max(dim=1, keepdim=True).values
            flat = (flat - mn) / (mx - mn + 1e-5)
            return flat
        return h.flatten(1)


class NatureCNNEncoder(nn.Module):
    """The Nature-DQN conv trunk alone: (C, 84, 84) frames -> (B, 3136) features.

    Encoder for head-owning agents (e.g. EfficientRainbowAgent wraps it with the
    RainbowIQNNetwork dueling head). Same conv stack and /255 normalisation as
    NatureCNN; ``feature_dim`` is exposed so agents skip the dummy forward.
    Accepts uint8 or float input — but note the /255 happens HERE, so callers
    that augment observations first must feed 0-255-scaled floats.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),          nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),          nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            self.feature_dim = int(self.features(dummy).shape[1])

    def forward(self, x):
        return self.features(x.float() / 255.0)
