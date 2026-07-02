import torch


def resolve_device(requested: str = "auto") -> str:
    """Resolve a requested torch device string to one actually available on this machine.

    "auto" (and any unavailable "cuda"/"mps" request) falls back in priority
    order CUDA > MPS > CPU, so the same config works on CUDA (Linux/HPC),
    MPS (Apple Silicon) and CPU-only machines without editing per-machine.
    """
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested not in ("auto", "cuda", "mps", "cpu"):
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"