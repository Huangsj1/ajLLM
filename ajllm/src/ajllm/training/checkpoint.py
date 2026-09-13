"""Portable checkpoints for ordinary and educational-FSDP pre-training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from ajllm.training.parallel.expert_parallel import ExpertParallel
from ajllm.training.parallel.fsdp import FullyShardedDataParallel
from ajllm.training.parallel.tensor_expert_parallel import TensorExpertParallel
from ajllm.training.parallel.tensor_parallel import TensorParallel


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _distributed() else 0


def _barrier() -> None:
    if _distributed():
        dist.barrier()


def _parallel_topology(model: torch.nn.Module) -> dict[str, int] | None:
    if isinstance(model, TensorExpertParallel):
        return {"tp_size": model.context.tp_size, "ep_size": model.context.ep_size}
    if isinstance(model, TensorParallel):
        return {"tp_size": model.world_size, "ep_size": 1}
    if isinstance(model, ExpertParallel):
        return {"tp_size": 1, "ep_size": model.world_size}
    return None


def _check_topology(checkpoint: dict[str, Any], model: torch.nn.Module) -> None:
    saved = checkpoint.get("parallel_topology")
    actual = _parallel_topology(model)
    if saved is not None and saved != actual:
        raise ValueError(f"Parallel checkpoint topology {saved} does not match current topology {actual}")


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
    fsdp = isinstance(model, FullyShardedDataParallel)
    tensor_parallel = isinstance(model, TensorParallel)
    expert_parallel = isinstance(model, ExpertParallel)
    tensor_expert_parallel = isinstance(model, TensorExpertParallel)
    uses_parallel_state_dict = fsdp or tensor_parallel or expert_parallel or tensor_expert_parallel
    model_state = model.full_state_dict() if uses_parallel_state_dict else model.state_dict()
    if _rank() == 0:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(
            {
                "model_state_dict": model_state,
                "step": step,
                "metadata": metadata,
                "training_state": training_state or {},
                "world_size": dist.get_world_size() if _distributed() else 1,
                "sharded": fsdp,
                "tensor_parallel": tensor_parallel,
                "expert_parallel": expert_parallel,
                "tensor_expert_parallel": tensor_expert_parallel,
                "parallel_topology": _parallel_topology(model),
            },
            temporary,
        )
        os.replace(temporary, destination)
    if fsdp or tensor_parallel or expert_parallel or tensor_expert_parallel:
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
    is_tensor_parallel = bool(checkpoint.get("tensor_parallel", False))
    is_expert_parallel = bool(checkpoint.get("expert_parallel", False))
    is_tensor_expert_parallel = bool(checkpoint.get("tensor_expert_parallel", False))
    if is_sharded:
        if not isinstance(model, FullyShardedDataParallel):
            raise ValueError("This checkpoint requires FullyShardedDataParallel")
        expected, actual = checkpoint["world_size"], dist.get_world_size() if _distributed() else 1
        if expected != actual:
            raise ValueError(f"FSDP resume requires world_size={expected}, got {actual}")
        model.load_full_state_dict(checkpoint["model_state_dict"])
        optimizer_path = source.with_suffix(source.suffix + f".rank{_rank()}.optim")
    elif is_tensor_parallel:
        if not isinstance(model, TensorParallel):
            raise ValueError("This checkpoint requires TensorParallel")
        expected, actual = checkpoint["world_size"], dist.get_world_size() if _distributed() else 1
        if expected != actual:
            raise ValueError(f"TensorParallel resume requires world_size={expected}, got {actual}")
        _check_topology(checkpoint, model)
        model.load_full_state_dict(checkpoint["model_state_dict"])
        optimizer_path = source.with_suffix(source.suffix + f".rank{_rank()}.optim")
    elif is_expert_parallel:
        if not isinstance(model, ExpertParallel):
            raise ValueError("This checkpoint requires ExpertParallel")
        expected, actual = checkpoint["world_size"], dist.get_world_size() if _distributed() else 1
        if expected != actual:
            raise ValueError(f"ExpertParallel resume requires world_size={expected}, got {actual}")
        _check_topology(checkpoint, model)
        model.load_full_state_dict(checkpoint["model_state_dict"])
        optimizer_path = source.with_suffix(source.suffix + f".rank{_rank()}.optim")
    elif is_tensor_expert_parallel:
        if not isinstance(model, TensorExpertParallel):
            raise ValueError("This checkpoint requires TensorExpertParallel")
        expected, actual = checkpoint["world_size"], dist.get_world_size() if _distributed() else 1
        if expected != actual:
            raise ValueError(f"TensorExpertParallel resume requires world_size={expected}, got {actual}")
        _check_topology(checkpoint, model)
        model.load_full_state_dict(checkpoint["model_state_dict"])
        optimizer_path = source.with_suffix(source.suffix + f".rank{_rank()}.optim")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_path = source.with_suffix(source.suffix + ".optim")
    if optimizer is not None:
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=map_location, weights_only=False))
    _barrier()
    return checkpoint
