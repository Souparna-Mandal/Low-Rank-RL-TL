"""DrQ-style image augmentation (Kostrikov et al. 2021, arXiv:2004.13649).

Pure, vectorised functions applied INSIDE the gradient step only — never in
``pi()`` or analysis rollouts, and always after the batch has moved to the
device (replay storage stays uint8/CPU). Both take and return 0-255-scaled
float tensors of shape (B, C, H, W).
"""
import torch
import torch.nn.functional as F


def random_shift(x: torch.Tensor, pad: int = 4, offsets: torch.Tensor | None = None):
    """DrQ random shift: replicate-pad each side by ``pad``, then crop each
    sample back to (H, W) at an integer offset drawn uniformly from
    [0, 2*pad]^2 (offset ``pad`` = identity crop).

    Passing ``offsets`` (B, 2) replays those exact crops — used to give an FHR
    anchor and its lag states the SAME shift, and by the tests. Returns
    ``(augmented, offsets)``.
    """
    B, C, H, W = x.shape
    padded = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    if offsets is None:
        offsets = torch.randint(0, 2 * pad + 1, (B, 2), device=x.device)
    else:
        if offsets.shape != (B, 2):
            raise ValueError(f"offsets must be ({B}, 2), got {tuple(offsets.shape)}")
        if bool((offsets < 0).any()) or bool((offsets > 2 * pad).any()):
            raise ValueError(f"offsets must lie in [0, {2 * pad}]")
    rows = offsets[:, 0].view(B, 1, 1) + torch.arange(H, device=x.device).view(1, H, 1)
    cols = offsets[:, 1].view(B, 1, 1) + torch.arange(W, device=x.device).view(1, 1, W)
    batch = torch.arange(B, device=x.device).view(B, 1, 1)
    # Advanced indices split by the C slice land in front: (B, H, W, C).
    out = padded[batch, :, rows, cols].permute(0, 3, 1, 2)
    return out.contiguous(), offsets


def intensity(x: torch.Tensor, scale: float = 0.05,
              factors: torch.Tensor | None = None):
    """DrQ intensity jitter: multiply each sample by ``1 + scale * clip(eps, ±2)``
    with eps ~ N(0, 1). Passing ``factors`` (B, 1, 1, 1) replays those exact
    multipliers (FHR sequences share one factor). Returns ``(augmented, factors)``.
    """
    if factors is None:
        noise = torch.randn(x.shape[0], 1, 1, 1, device=x.device).clamp_(-2.0, 2.0)
        factors = 1.0 + scale * noise
    return x * factors, factors
