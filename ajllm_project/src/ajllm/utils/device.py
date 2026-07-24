"""Device selection helpers."""

from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def explain_device_selection(requested: str, selected: torch.device) -> str | None:
    """Explain an automatic fallback when no accelerator was selected."""

    if requested != "auto" or selected.type != "cpu":
        return None
    if torch.version.cuda is None:
        return "CUDA unavailable: the installed PyTorch build has no CUDA support (CPU-only wheel)."
    return "CUDA unavailable: PyTorch has CUDA support, but torch.cuda.is_available() is False."
