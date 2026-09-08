"""Megatron-style tensor parallel adapters for the project's dense Transformer.

The public entry point wraps an already constructed :class:`TransformerLM`.
It replaces only the linear paths whose input/output layouts are known, so the
model's public forward contract stays ``input_ids -> full vocabulary logits``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
from torch import nn

from ajllm.modeling.activations import silu
from ajllm.modeling.attention import RotaryPositionalEmbedding, repeat_kv
from ajllm.modeling.cuda_kernels import swiglu_gate
from ajllm.modeling.layers import Linear, RMSNorm
from ajllm.training.parallel.common import (
    _dist_is_ready,
    _rank,
    _world_size,
    all_gather_cat,
    all_reduce_sum,
    broadcast_module_,
    copy_to_tensor_parallel,
    reduce_from_tensor_parallel,
)


@dataclass(frozen=True)
class _ShardLayout:
    """How a local parameter maps back to its unwrapped checkpoint tensor."""

    axis: int
    full_shape: torch.Size


class TensorParallelLinear(nn.Module):
    """A bias-free project ``Linear`` sharded along its output or input axis."""

    def __init__(
        self,
        linear: Linear,
        style: Literal["column", "row"],
        group: dist.ProcessGroup | None,
    ) -> None:
        super().__init__()
        if not isinstance(linear, Linear):
            raise TypeError("TensorParallelLinear can only convert ajllm.modeling.layers.Linear")
        self.style = style
        self.group = group
        self.world_size = _world_size(group)
        self.rank = _rank(group)
        self.global_in_features = linear.in_features
        self.global_out_features = linear.out_features
        axis = 0 if style == "column" else 1
        partitioned = linear.weight.shape[axis]
        if partitioned % self.world_size:
            dimension = "out_features" if style == "column" else "in_features"
            raise ValueError(f"{dimension}={partitioned} is not divisible by tp_size={self.world_size}")
        shard_size = partitioned // self.world_size
        start = self.rank * shard_size
        local_weight = linear.weight.detach().narrow(axis, start, shard_size).contiguous().clone()
        self.weight = nn.Parameter(local_weight, requires_grad=linear.weight.requires_grad)
        self.in_features = self.global_in_features if style == "column" else shard_size
        self.out_features = shard_size if style == "column" else self.global_out_features
        self.shard_layout = _ShardLayout(axis=axis, full_shape=linear.weight.shape)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.style == "column":
            # forward: directly use the full input
            # backward: reduce sum gradients of d_input across shards
            inputs = copy_to_tensor_parallel(inputs, self.group)
            return inputs @ self.weight.transpose(-2, -1)
        # the input is already partially sharded, so we must reduce sum the partial outputs across shards
        partial = inputs @ self.weight.transpose(-2, -1)
        return reduce_from_tensor_parallel(partial, self.group)


class TensorParallelSwiGLU(nn.Module):
    """Column-parallel gate/up projections followed by a row-parallel down projection."""

    def __init__(self, module: nn.Module, group: dist.ProcessGroup | None) -> None:
        super().__init__()
        # Deliberately structural: this keeps the adapter usable for the exact
        # project SwiGLU class without adding a second public model definition.
        if not all(hasattr(module, name) for name in ("gate_proj", "up_proj", "down_proj", "use_cuda_kernels")):
            raise TypeError("TensorParallelSwiGLU requires the project's SwiGLU module")
        self.gate_proj = TensorParallelLinear(module.gate_proj, "column", group)
        self.up_proj = TensorParallelLinear(module.up_proj, "column", group)
        self.down_proj = TensorParallelLinear(module.down_proj, "row", group)
        self.use_cuda_kernels = module.use_cuda_kernels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_proj(inputs), self.up_proj(inputs)
        hidden = swiglu_gate(gate, up) if self.use_cuda_kernels else silu(gate) * up
        return self.down_proj(hidden)


class TensorParallelAttention(nn.Module):
    """GQA with local heads and a row-parallel output projection."""

    def __init__(self, attention: nn.Module, group: dist.ProcessGroup | None) -> None:
        super().__init__()
        required = (
            "d_model",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "num_kv_groups",
            "q_proj",
            "k_proj",
            "v_proj",
            "output_proj",
            "rope",
            "qk_norm",
            "dropout",
            "use_flash_attention",
        )
        if not all(hasattr(attention, name) for name in required):
            raise TypeError("TensorParallelAttention requires GroupedQueryAttention")
        self.group = group
        self.world_size = _world_size(group)
        self.rank = _rank(group)
        if attention.num_heads % self.world_size or attention.num_kv_heads % self.world_size:
            raise ValueError(
                f"num_heads={attention.num_heads} and num_kv_heads={attention.num_kv_heads} "
                f"must both divide by tp_size={self.world_size}"
            )
        self.d_model = attention.d_model
        self.global_num_heads = attention.num_heads
        self.global_num_kv_heads = attention.num_kv_heads
        self.num_heads = attention.num_heads // self.world_size
        self.num_kv_heads = attention.num_kv_heads // self.world_size
        self.head_dim = attention.head_dim
        self.num_kv_groups = attention.num_kv_groups
        self.dropout = attention.dropout
        self.use_flash_attention = attention.use_flash_attention
        self.q_proj = TensorParallelLinear(attention.q_proj, "column", group)
        self.k_proj = TensorParallelLinear(attention.k_proj, "column", group)
        self.v_proj = TensorParallelLinear(attention.v_proj, "column", group)
        self.output_proj = TensorParallelLinear(attention.output_proj, "row", group)
        self.rope: RotaryPositionalEmbedding = attention.rope
        self.qk_norm = attention.qk_norm
        if self.qk_norm:
            self.q_norm: RMSNorm = attention.q_norm
            self.k_norm: RMSNorm = attention.k_norm
            self.q_norm.weight.register_hook(self._sum_replicated_gradient)
            self.k_norm.weight.register_hook(self._sum_replicated_gradient)

    def _sum_replicated_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        return all_reduce_sum(gradient, self.group)

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = inputs.shape
        queries = self.q_proj(inputs).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.k_proj(inputs).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        values = self.v_proj(inputs).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)
        queries = self.rope(queries, positions[:, None, :].expand(batch_size, self.num_heads, seq_len))
        keys = self.rope(keys, positions[:, None, :].expand(batch_size, self.num_kv_heads, seq_len))
        keys = repeat_kv(keys, self.num_kv_groups)
        values = repeat_kv(values, self.num_kv_groups)

        from ajllm.modeling.flash_attention import flash_attention, flash_attention_pytorch

        sequence = queries.shape[-2]
        padded_sequence = max(16, 1 << (sequence - 1).bit_length())
        if self.use_flash_attention and padded_sequence != sequence:
            padding_shape = (*queries.shape[:-2], padded_sequence - sequence, queries.shape[-1])
            queries = torch.cat((queries, queries.new_zeros(padding_shape)), dim=-2)
            keys = torch.cat((keys, keys.new_zeros(padding_shape)), dim=-2)
            values = torch.cat((values, values.new_zeros(padding_shape)), dim=-2)
        attended = (
            flash_attention(queries, keys, values, True)[..., :sequence, :]
            if self.use_flash_attention
            else flash_attention_pytorch(queries, keys, values, True)
        )
        output = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.output_proj(output)


class TensorParallel(nn.Module):
    """Wrap a dense ``TransformerLM`` with project-specific tensor parallelism.

    The wrapper requires an initialized process group when ``tp_size > 1``.
    Every TP rank must receive identical token ids and labels.  It is currently
    intentionally separate from FSDP and expert parallelism; ``wrap_parallel``
    will combine these strategies after their independent test suites exist.
    """

    is_tensor_parallel = True

    def __init__(self, module: nn.Module, group: dist.ProcessGroup | None = None) -> None:
        super().__init__()
        self.group = group
        self.world_size = _world_size(group)
        self.rank = _rank(group)
        if self.world_size > 1 and not _dist_is_ready():
            raise RuntimeError("TensorParallel requires torch.distributed to be initialized")
        if not hasattr(module, "config") or not hasattr(module, "layers"):
            raise TypeError("TensorParallel requires the project's TransformerLM")
        if module.config.model_type != "dense":
            raise ValueError("TensorParallel currently supports dense models only; use the TP×EP wrapper for MoE")
        self._broadcast_initial_state(module)
        self.module = module
        self._layouts: dict[str, _ShardLayout] = {}
        self._convert_dense_transformer()

    @property
    def config(self):
        return self.module.config

    def _broadcast_initial_state(self, module: nn.Module) -> None:
        if self.world_size == 1:
            return
        broadcast_module_(module, self.group)

    def _convert_dense_transformer(self) -> None:
        for layer_index, layer in enumerate(self.module.layers):
            layer.attention = TensorParallelAttention(layer.attention, self.group)
            layer.feed_forward = TensorParallelSwiGLU(layer.feed_forward, self.group)
            attention_prefix = f"layers.{layer_index}.attention"
            ffn_prefix = f"layers.{layer_index}.feed_forward"
            self._register_linear_layouts(attention_prefix, layer.attention)
            self._register_linear_layouts(ffn_prefix, layer.feed_forward)

    def _register_linear_layouts(self, prefix: str, module: nn.Module) -> None:
        for name, child in module.named_modules():
            if isinstance(child, TensorParallelLinear):
                full_name = f"{prefix}.{name}.weight" if name else f"{prefix}.weight"
                self._layouts[full_name] = child.shard_layout

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def auxiliary_loss(self) -> torch.Tensor:
        return self.module.auxiliary_loss()

    def parameter_count(self) -> int:
        ''' get the full parameter count'''
        count = 0
        for name, parameter in self.module.named_parameters():
            layout = self._layouts.get(name)
            count += int(torch.tensor(layout.full_shape).prod()) if layout else parameter.numel()
        return count

    def finish_gradient_synchronization(self) -> None:
        """Compatibility hook; TP collectives run in autograd at their exact boundaries."""

    def clip_grad_norm_(self, maximum_norm: float, epsilon: float = 1e-6) -> float:
        """Clip the norm of logical parameters, counting TP replicas once."""
        squared_norm = torch.zeros((), device=next(self.parameters()).device, dtype=torch.float32)
        for name, parameter in self.module.named_parameters():
            # rank0 counts all parameters, other ranks only count those that are sharded
            if parameter.grad is None:
                continue
            if name not in self._layouts and self.rank != 0:
                continue
            squared_norm.add_(parameter.grad.detach().float().square().sum())
        if self.world_size > 1:
            dist.all_reduce(squared_norm, op=dist.ReduceOp.SUM, group=self.group)
        norm = torch.sqrt(squared_norm)
        norm_value = float(norm.item())
        if norm_value > maximum_norm:
            scale = maximum_norm / (norm_value + epsilon)
            for parameter in self.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
        return norm_value

    @torch.no_grad()
    def full_state_dict(self) -> dict[str, torch.Tensor]:
        """Materialize the original unwrapped parameter keys and shapes on every TP rank."""
        state = self.module.state_dict()
        full_state: dict[str, torch.Tensor] = {}
        for name, value in state.items():
            layout = self._layouts.get(name)
            if layout is None:
                full_state[name] = value.detach().clone()
                continue
            full_state[name] = all_gather_cat(value, layout.axis, self.group).reshape(layout.full_shape).clone()
        return full_state

    @torch.no_grad()
    def load_full_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load an ordinary model state by slicing each TP-local parameter."""
        local_state = self.module.state_dict()
        missing = set(local_state) - set(state_dict)
        unexpected = set(state_dict) - set(local_state)
        if missing or unexpected:
            raise ValueError(f"Checkpoint keys differ; missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        for name, destination in local_state.items():
            source = state_dict[name].to(destination.device, dtype=destination.dtype)
            layout = self._layouts.get(name)
            if layout is not None:
                shard_size = layout.full_shape[layout.axis] // self.world_size
                source = source.narrow(layout.axis, self.rank * shard_size, shard_size)
            if source.shape != destination.shape:
                raise ValueError(f"Checkpoint tensor {name} has shape {source.shape}, expected {destination.shape}")
            destination.copy_(source)


def tensor_parallelize(module: nn.Module, group: dist.ProcessGroup | None = None) -> TensorParallel:
    """Construct the tensor-parallel wrapper around an existing dense model."""
    return TensorParallel(module, group)
