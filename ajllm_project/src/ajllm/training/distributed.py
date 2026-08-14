"""Fully Sharded Data Parallel with Activation Checkpointing for multi-GPU training.

FSDP shards model parameters, gradients, and optimizer states across GPUs,
enabling training of models larger than single-GPU memory. Activation checkpointing
trades compute for memory by recomputing activations during backward pass.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn


# ========== Helper Functions ==========

def _dist_is_ready() -> bool:
    """Check if distributed process group is initialized."""
    return dist.is_available() and dist.is_initialized()


def _world_size() -> int:
    """Get number of processes in distributed group."""
    return dist.get_world_size() if _dist_is_ready() else 1


def _rank() -> int:
    """Get current process rank."""
    return dist.get_rank() if _dist_is_ready() else 0


def _broadcast_tensor_(tensor: torch.Tensor, src: int = 0) -> None:
    """Broadcast tensor from src rank to all ranks."""
    if _world_size() > 1:
        dist.broadcast(tensor, src=src)


def _all_reduce_average_(tensor: torch.Tensor) -> None:
    """Average tensor across all ranks via all-reduce."""
    world_size = _world_size()
    if world_size == 1:
        return
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(world_size)


# ========== Shard Management ==========

@dataclass
class _ShardInfo:
    """Metadata for a sharded parameter.

    The full parameter is replaced by a flat 1-D shard containing only
    this rank's slice. We need shape info to reconstruct the full tensor.
    """
    full_name: str          # Original parameter name
    attr_name: str          # Attribute name on module (e.g., "weight")
    module: nn.Module       # Owner module
    shard_param: nn.Parameter  # Flat 1-D shard (this rank's slice)
    shape: torch.Size       # Original full shape
    numel: int              # Full element count (before padding)
    shard_numel: int        # Elements per rank (after padding for even split)


@dataclass
class _AsyncAllGather:
    """One in-flight all-gather operation."""
    input_shard: torch.Tensor
    full_flat: torch.Tensor
    work: dist.Work | None


def _all_gather_flat(shard: torch.Tensor, numel: int, shape: torch.Size) -> torch.Tensor:
    """Gather all shards, drop padding, reshape to original."""
    world_size = _world_size()
    if world_size == 1:
        return shard[:numel].reshape(shape)
    gathered = torch.empty(shard.numel() * world_size, device=shard.device, dtype=shard.dtype)
    dist.all_gather_into_tensor(gathered, shard.contiguous())
    return gathered[:numel].reshape(shape)


def _start_all_gather_flat(shard: torch.Tensor, compute_dtype: torch.dtype | None = None) -> _AsyncAllGather:
    """Start async all-gather without autograd attachment."""
    world_size = _world_size()
    input_shard = shard.detach().contiguous()
    if compute_dtype is not None:
        # Communicate in compute dtype to save bandwidth
        input_shard = input_shard.to(compute_dtype)
    if world_size == 1:
        return _AsyncAllGather(input_shard, input_shard.clone(), None)
    full_flat = torch.empty(input_shard.numel() * world_size, device=shard.device, dtype=input_shard.dtype)
    work = dist.all_gather_into_tensor(full_flat, input_shard, async_op=True)
    return _AsyncAllGather(input_shard, full_flat, work)


def _finish_all_gather_flat(prefetch: _AsyncAllGather, numel: int, shape: torch.Size) -> torch.Tensor:
    """Wait for async all-gather and reshape result."""
    if prefetch.work is not None:
        prefetch.work.wait()
    return prefetch.full_flat[:numel].reshape(shape).clone()


def _reduce_scatter_average(full_grad: torch.Tensor, shard_numel: int) -> torch.Tensor:
    """Average full gradient across ranks and keep only this rank's slice.

    This is the key FSDP backward optimization: each rank only keeps
    the gradient slice matching its parameter shard.
    """
    world_size = _world_size()
    flat = full_grad.reshape(-1)
    padded_numel = shard_numel * world_size
    if flat.numel() != padded_numel:
        padded = torch.zeros(padded_numel, device=flat.device, dtype=flat.dtype)
        padded[: flat.numel()].copy_(flat)
        flat = padded
    if world_size == 1:
        return flat[:shard_numel].clone()
    out = torch.empty(shard_numel, device=flat.device, dtype=flat.dtype)
    dist.reduce_scatter_tensor(out, flat.contiguous(), op=dist.ReduceOp.SUM)
    return out.div_(world_size)


# ========== Autograd Functions ==========

class _AllGatherWeight(torch.autograd.Function):
    """All-gather forward, reduce-scatter backward (synchronous fallback)."""

    @staticmethod
    def forward(ctx, shard: torch.Tensor, numel: int, shape: torch.Size, compute_dtype: torch.dtype | None):
        ctx.shard_numel = shard.numel()
        ctx.shard_dtype = shard.dtype
        # Cast to compute_dtype before all-gather to save bandwidth
        comm_shard = shard if compute_dtype is None else shard.to(compute_dtype)
        full = _all_gather_flat(comm_shard, numel, shape)
        return full

    @staticmethod
    def backward(ctx, grad_full: torch.Tensor):
        grad_shard = _reduce_scatter_average(grad_full.to(ctx.shard_dtype), ctx.shard_numel)
        return grad_shard, None, None, None


class _PrefetchedAllGatherWeight(torch.autograd.Function):
    """Attach prefetched weight to autograd graph with reduce-scatter backward."""

    @staticmethod
    def forward(ctx, shard: torch.Tensor, prefetched_full: torch.Tensor):
        ctx.shard_numel = shard.numel()
        ctx.shard_dtype = shard.dtype
        return prefetched_full

    @staticmethod
    def backward(ctx, grad_full: torch.Tensor):
        grad_shard = _reduce_scatter_average(grad_full.to(ctx.shard_dtype), ctx.shard_numel)
        return grad_shard, None


# ========== Main FSDP Module ==========

class FullyShardedDataParallel(nn.Module):
    """FSDP with Activation Checkpointing enabled by default.

    Shards Linear and Embedding layers across GPUs. Each rank stores only 1/world_size
    of each weight. Full weights are reconstructed via all-gather during forward,
    then dropped after the layer completes. Gradients are reduced via reduce-scatter
    during backward so each rank only keeps its shard's gradient.

    Activation Checkpointing wraps each sharded module: during forward, activations
    are NOT saved; during backward, the module's forward is recomputed from the shard.
    This trades ~33% extra compute for 2-4x memory savings.
    """

    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
        activation_checkpointing: bool = True,
    ):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.activation_checkpointing = activation_checkpointing
        self._shards: list[_ShardInfo] = []

        # Step 1: Sync initial weights from rank 0
        self._sync_initial_state()

        # Step 2: Shard parameters across ranks
        self._shard_parameters()

        self._module_to_shard_index = {id(info.module): idx for idx, info in enumerate(self._shards)}
        self._prefetches: dict[int, _AsyncAllGather] = {}
        self._prefetch_distance = 2  # Prefetch 2 layers ahead

        # Step 3: Register hooks to all-gather before forward
        self._register_gather_hooks()

        # Step 4: Apply activation checkpointing if enabled
        if activation_checkpointing:
            self._apply_activation_checkpointing()

    def _sync_initial_state(self) -> None:
        """Broadcast all parameters and buffers from rank 0."""
        with torch.no_grad():
            for tensor in itertools.chain(self.module.parameters(), self.module.buffers()):
                _broadcast_tensor_(tensor.data, src=0)

    def _is_sharded_module(self, module: nn.Module) -> bool:
        """Check if module should be sharded (Linear or Embedding)."""
        # Import locally to avoid circular dependency
        from ajllm.modeling.layers import Linear, Embedding
        return isinstance(module, (Linear, Embedding))

    def _shard_parameters(self) -> None:
        """Replace full parameters with rank-local shards."""
        world_size = _world_size()
        rank = _rank()
        seen: set[int] = set()

        for module_name, module in self.module.named_modules():
            if not self._is_sharded_module(module):
                continue
            param = module._parameters.get("weight")
            if not isinstance(param, nn.Parameter):
                continue
            if id(param) in seen:  # Handle tied weights
                continue
            seen.add(id(param))

            # Flatten and convert to fp32 master copy
            full = param.detach().reshape(-1).float()
            numel = full.numel()

            # Pad to make evenly divisible
            shard_numel = (numel + world_size - 1) // world_size
            padded = torch.zeros(shard_numel * world_size, device=param.device, dtype=torch.float32)
            padded[:numel].copy_(full)
            shard = padded[rank * shard_numel : (rank + 1) * shard_numel].clone()

            # Replace full parameter with shard
            del module._parameters["weight"]
            shard_param = nn.Parameter(shard, requires_grad=param.requires_grad)
            module.register_parameter("weight_shard", shard_param)
            module.weight = None  # Placeholder, filled by forward hook

            self._shards.append(
                _ShardInfo(
                    full_name=f"{module_name}.weight" if module_name else "weight",
                    attr_name="weight",
                    module=module,
                    shard_param=shard_param,
                    shape=param.shape,
                    numel=numel,
                    shard_numel=shard_numel,
                )
            )

    def _apply_activation_checkpointing(self) -> None:
        """Wrap each sharded module's forward with torch.utils.checkpoint.

        The all-gather is placed INSIDE the checkpointed function so during
        recompute: (1) shard is all-gathered, (2) forward runs, (3) weight is dropped.
        This way autograd never saves the full weight for backward.
        """
        from torch.utils.checkpoint import checkpoint

        for info in self._shards:
            mod = info.module
            orig_fwd = type(mod).forward

            def _make_recompute_fn(mod_ref, sp, n, sh, cd, of):
                def _recompute(*args, **kwargs):
                    full_weight = _AllGatherWeight.apply(sp, n, sh, cd)
                    mod_ref.weight = full_weight
                    try:
                        return of(mod_ref, *args, **kwargs)
                    finally:
                        mod_ref.weight = None
                return _recompute

            recompute_fn = _make_recompute_fn(
                mod, info.shard_param, info.numel, info.shape, self.compute_dtype, orig_fwd
            )

            def _make_forward_method(rfn):
                def _forward(self_unused, *args, **kwargs):
                    return checkpoint(rfn, *args, use_reentrant=False, **kwargs)
                return _forward

            mod.forward = _make_forward_method(recompute_fn).__get__(mod, type(mod))

    def forward(self, *args, **kwargs):
        # Clear any stale prefetches from previous forward
        self._finish_and_clear_prefetches()
        return self.module(*args, **kwargs)

    def _start_prefetch(self, index: int) -> None:
        """Start async all-gather for a layer ahead of current."""
        if index < 0 or index >= len(self._shards) or index in self._prefetches:
            return
        info = self._shards[index]
        self._prefetches[index] = _start_all_gather_flat(info.shard_param, self.compute_dtype)

    def _finish_and_clear_prefetches(self) -> None:
        """Wait for all pending prefetches and clear."""
        for prefetch in self._prefetches.values():
            if prefetch.work is not None:
                prefetch.work.wait()
        self._prefetches.clear()

    def _prefetch_ahead(self, current_index: int) -> None:
        """Prefetch next N layers."""
        if self._prefetch_distance is None:
            return
        for offset in range(1, self._prefetch_distance + 1):
            self._start_prefetch(current_index + offset)

    def _take_prefetched_weight(self, index: int) -> torch.Tensor:
        """Get prefetched weight or fall back to synchronous gather."""
        info = self._shards[index]
        prefetch = self._prefetches.pop(index, None)
        if prefetch is None:
            # Synchronous fallback
            return _AllGatherWeight.apply(info.shard_param, info.numel, info.shape, self.compute_dtype)
        full = _finish_all_gather_flat(prefetch, info.numel, info.shape)
        # Note: full is already in compute_dtype from _start_all_gather_flat
        return _PrefetchedAllGatherWeight.apply(info.shard_param, full)

    def _register_gather_hooks(self) -> None:
        """Register pre/post hooks to gather and release weights."""
        def pre_hook(module: nn.Module, _inputs):
            index = self._module_to_shard_index[id(module)]
            module.weight = self._take_prefetched_weight(index)
            self._prefetch_ahead(index)

        def post_hook(module: nn.Module, _inputs_unused, output):
            module.weight = None  # Release reference (autograd still holds it)
            return output

        for info in self._shards:
            if self.activation_checkpointing:
                # AC mode has gather inside checkpointed forward, skip hooks
                continue
            info.module.register_forward_pre_hook(pre_hook)
            info.module.register_forward_hook(post_hook)

    def finish_gradient_synchronization(self) -> None:
        """Synchronize gradients for non-sharded parameters.

        Sharded parameters are already synchronized via reduce-scatter in
        _AllGatherWeight.backward. Only small replicated params (RMSNorm, etc.)
        need all-reduce here.
        """
        sharded_ids = {id(info.shard_param) for info in self._shards}
        for param in self.module.parameters():
            if not param.requires_grad or param.grad is None:
                continue
            if id(param) in sharded_ids:
                continue
            # All-reduce replicated parameter gradients
            param.grad.data = param.grad.data.float()
            _all_reduce_average_(param.grad.data)

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        """Gather full parameters from all shards (for checkpointing)."""
        full_params: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for info in self._shards:
                full = _all_gather_flat(info.shard_param.detach(), info.numel, info.shape)
                full_params[info.full_name] = full.float().clone()
            for name, param in self.module.named_parameters():
                if name.endswith("weight_shard"):
                    continue
                full_params[name] = param.detach().clone()
        return full_params
