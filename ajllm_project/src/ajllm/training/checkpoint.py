"""Atomic checkpoint serialization and restoration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metadata: dict[str, Any],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "metadata": metadata,
        },
        temporary,
    )
    os.replace(temporary, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
