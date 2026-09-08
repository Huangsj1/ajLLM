"""Shared process-group, collective, and topology helpers for model parallelism."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn


def _dist_is_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _world_size(group: dist.ProcessGroup | None = None) -> int:
    return dist.get_world_size(group) if _dist_is_ready() else 1


def _rank(group: dist.ProcessGroup | None = None) -> int:
    return dist.get_rank(group) if _dist_is_ready() else 0


def _broadcast_tensor_(tensor: torch.Tensor, src: int = 0, group: dist.ProcessGroup | None = None) -> None:
    if _world_size(group) > 1:
        dist.broadcast(tensor, src=src, group=group)


def _all_reduce_average_(tensor: torch.Tensor, group: dist.ProcessGroup | None = None) -> None:
    world_size = _world_size(group)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        tensor.div_(world_size)


def all_reduce_sum(tensor: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
    """Return a summed copy without modifying an autograd-saved tensor."""
    if _world_size(group) == 1:
        return tensor
    result = tensor.contiguous().clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
    return result


class _CopyToTensorParallelRegion(torch.autograd.Function):
    """Replicate activations forward and sum their TP gradients in backward."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
        ctx.group = group
        return inputs

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return all_reduce_sum(gradient, ctx.group), None


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    """Sum TP partial outputs forward and replicate output gradients in backward."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
        return all_reduce_sum(inputs, group)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return gradient, None


def copy_to_tensor_parallel(inputs: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
    return _CopyToTensorParallelRegion.apply(inputs, group)


def reduce_from_tensor_parallel(inputs: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
    return _ReduceFromTensorParallelRegion.apply(inputs, group)


def broadcast_module_(module: nn.Module, group: dist.ProcessGroup | None = None) -> None:
    """Synchronize parameters and buffers from group rank zero before sharding."""
    if _world_size(group) == 1:
        return
    with torch.no_grad():
        for tensor in list(module.parameters()) + list(module.buffers()):
            dist.broadcast(tensor, src=0, group=group)


def all_gather_cat(tensor: torch.Tensor, axis: int, group: dist.ProcessGroup | None) -> torch.Tensor:
    """Gather equal-shaped shards and concatenate them along ``axis``."""
    world_size = _world_size(group)
    if world_size == 1:
        return tensor.detach().clone()
    shards = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(shards, tensor.contiguous(), group=group)
    return torch.cat(shards, dim=axis)


@dataclass(frozen=True)
class ParallelContext:
    """A deterministic 2-D EP×TP process topology.

    Ranks use ``global_rank = ep_rank * tp_size + tp_rank``.  TP groups hold
    one EP shard fixed; EP groups hold one TP shard fixed.  Groups are created
    even for a unit dimension so that ``None`` retains PyTorch's standard
    meaning of the default global process group.
    """

    tp_size: int
    ep_size: int
    tp_group: dist.ProcessGroup | None  # current rank's TP group
    ep_group: dist.ProcessGroup | None  # current rank's EP group
    tp_rank: int
    ep_rank: int

    @property
    def world_size(self) -> int:
        return self.tp_size * self.ep_size

    @classmethod
    def from_distributed(cls, tp_size: int = 1, ep_size: int = 1) -> ParallelContext:
        if tp_size < 1 or ep_size < 1:
            raise ValueError("tp_size and ep_size must be positive")
        if tp_size * ep_size != _world_size():
            raise ValueError(
                f"tp_size * ep_size must equal WORLD_SIZE; got {tp_size} * {ep_size} != {_world_size()}"
            )
        if _world_size() == 1:
            return cls(1, 1, None, None, 0, 0)
        global_rank = _rank()
        ep_rank, tp_rank = divmod(global_rank, tp_size)
        tp_group = None
        ep_group = None
        # Every process must create every group in the same order.
        for ep_index in range(ep_size):
            group = dist.new_group(list(range(ep_index * tp_size, (ep_index + 1) * tp_size)))
            if ep_index == ep_rank:
                tp_group = group
        for tp_index in range(tp_size):
            group = dist.new_group([ep_index * tp_size + tp_index for ep_index in range(ep_size)])
            if tp_index == tp_rank:
                ep_group = group
        return cls(tp_size, ep_size, tp_group, ep_group, tp_rank, ep_rank)
