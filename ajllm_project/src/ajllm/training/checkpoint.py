"""Portable checkpoints for ordinary and educational-FSDP pre-training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from ajllm.training.distributed import FullyShardedDataParallel


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _distributed() else 0


def _barrier() -> None:
    if _distributed():
        dist.barrier()


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metadata: dict[str, Any],
    training_state: dict[str, int] | None = None,
) -> Path:
    """Save an unwrapped portable model state and rank-local optimizer state."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sharded = isinstance(model, FullyShardedDataParallel)
    model_state = model.full_state_dict() if sharded else model.state_dict()
    if _rank() == 0:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(
            {
                "model_state_dict": model_state,
                "step": step,
                "metadata": metadata,
                "training_state": training_state or {},
                "world_size": dist.get_world_size() if _distributed() else 1,
                "sharded": sharded,
            },
            temporary,
        )
        os.replace(temporary, destination)
    if sharded:
        torch.save(optimizer.state_dict(), destination.with_suffix(destination.suffix + f".rank{_rank()}.optim"))
    elif _rank() == 0:
        torch.save(optimizer.state_dict(), destination.with_suffix(destination.suffix + ".optim"))
    _barrier()
    return destination


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore a checkpoint. FSDP resumes require the same world size."""
    source = Path(path)
    checkpoint = torch.load(source, map_location=map_location, weights_only=False)
    is_sharded = bool(checkpoint.get("sharded", False))
    if is_sharded:
        if not isinstance(model, FullyShardedDataParallel):
            raise ValueError("This checkpoint requires FullyShardedDataParallel")
        expected, actual = checkpoint["world_size"], dist.get_world_size() if _distributed() else 1
        if expected != actual:
            raise ValueError(f"FSDP resume requires world_size={expected}, got {actual}")
        model.load_full_state_dict(checkpoint["model_state_dict"])
        optimizer_path = source.with_suffix(source.suffix + f".rank{_rank()}.optim")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_path = source.with_suffix(source.suffix + ".optim")
    if optimizer is not None:
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=map_location, weights_only=False))
    _barrier()
    return checkpoint
