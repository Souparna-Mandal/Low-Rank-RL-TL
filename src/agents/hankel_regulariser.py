import torch
import torch.nn as nn


def _energy_rank(svals: torch.Tensor, energy_frac: float = 0.999) -> torch.Tensor:
    """Batched analogue of analysis.low_rank.rank.energy_rank: smallest k whose
    squared singular values capture energy_frac of the Frobenius energy
    (0 for an all-zero spectrum, matching the reference). The default tracks the
    reference's, so `batch_eff_rank` and the sweep's eff_rank mean the same thing."""
    energy = svals.pow(2)
    total = energy.sum(dim=1, keepdim=True)
    cum = energy.cumsum(dim=1) / total.clamp_min(1e-12)
    rank = (cum < energy_frac).sum(dim=1) + 1
    return torch.where(total.squeeze(1) > 0, rank, torch.zeros_like(rank))


class HankelRankPenalty(nn.Module):
    """Truncated-nuclear-norm penalty on Hankel matrices of value sequences.

    Each row of value_seqs (B, T) is lifted to a Hankel matrix H of shape
    (T-L+1, L), L = ceil(T/2). The penalty is the singular-value tail beyond
    `order`, normalised by the (stop-gradient) total: zero iff rank(H) <= order.
    Windows whose relative tail energy exceeds gate_threshold are excluded
    (off-manifold windows give arbitrary gradients), as are windows already at
    rank <= order (nothing to optimise, degenerate spectra).

    log_transform lifts each window through the signed log sign(v)*log1p(|v|)
    before the Hankel construction, so the tail is measured on the
    log-magnitude value sequence. The signed form is needed because value
    sequences can be negative or zero (a plain log would be undefined); it is
    monotone and, for |v| >> 1, log|v_t| of a geometric |v_t| is affine in t —
    Hankel rank 2.
    """

    def __init__(self, order: int = 2, gate_threshold: float | None = None,
                 detach_denominator: bool = True, jitter: float = 0.0,
                 log_transform: bool = False):
        super().__init__()
        self.order = order
        self.gate_threshold = gate_threshold
        self.detach_denominator = detach_denominator
        self.jitter = jitter
        self.log_transform = log_transform

    def forward(self, value_seqs: torch.Tensor, keep_mask: torch.Tensor | None = None):
        """value_seqs: (B, T) predicted values along episode-contiguous windows.
        keep_mask: optional (B,) bool — external per-window gate (e.g. TD
        consistency) ANDed with the internal ones. Returns (penalty, diag)."""
        B, T = value_seqs.shape
        L = (T + 1) // 2
        diag = {"penalty_raw": 0.0, "gate_frac": 0.0, "converged_frac": 0.0,
                "ext_gate_frac": 0.0, "batch_eff_rank": float("nan"),
                "rel_tail": float("nan")}
        if min(T - L + 1, L) <= self.order:
            return value_seqs.new_zeros(()), diag

        seqs = value_seqs
        if self.log_transform:
            seqs = torch.sign(seqs) * torch.log1p(seqs.abs())
        if self.jitter > 0:
            seqs = seqs + self.jitter * seqs.detach().std().clamp_min(1e-8) * torch.randn_like(seqs)
        H = seqs.unfold(dimension=1, size=L, step=1)  # (B, T-L+1, L)
        # float64 for a stable svdvals backward; MPS has no float64, so hop to
        # CPU first, then widen (the combined .to(cpu, float64) is rejected on
        # MPS, and so is its backward replay — autograd flows across both hops).
        H = H.to("cpu").double() if H.device.type == "mps" else H.double()
        svals = torch.linalg.svdvals(H)  # (B, min(T-L+1, L))
        tail = svals[:, self.order:].sum(dim=1)
        total = svals.sum(dim=1).clamp_min(1e-12)

        rel_tail = (tail / total).detach()
        converged = rel_tail <= 1e-6   # already at rank <= order
        gated = (rel_tail > self.gate_threshold if self.gate_threshold is not None
                 else torch.zeros_like(converged))
        keep_b = ~converged & ~gated
        if keep_mask is not None:
            ext = keep_mask.to(keep_b.device)
            diag["ext_gate_frac"] = float(1.0 - ext.float().mean())
            keep_b = keep_b & ext
        keep = keep_b.to(svals.dtype)
        denom = total.detach() if self.detach_denominator else total
        # Dropped windows get zero cotangents on their σs; svdvals' backward
        # (U diag(g) Vᵀ) stays finite even for their degenerate spectra, so
        # they contribute exactly zero gradient.
        per_window = (svals[:, self.order:] * keep.unsqueeze(1)).sum(dim=1) / denom
        penalty = per_window.sum() / keep.sum().clamp_min(1.0)

        with torch.no_grad():
            diag["penalty_raw"] = float(penalty)
            diag["gate_frac"] = float(gated.float().mean())        # above ρ (off-manifold)
            diag["converged_frac"] = float(converged.float().mean())  # tail already ~0
            diag["batch_eff_rank"] = float(_energy_rank(svals).float().mean())
            diag["rel_tail"] = float(rel_tail.mean())
        # Cast dtype first, then move device: the backward replay of the
        # reversed ops is MPS-legal in this order (float64 never touches MPS).
        return penalty.to(value_seqs.dtype).to(value_seqs.device), diag
