"""A unified TP×EP wrapper that converts each MoE feed-forward layer once."""

from __future__ import annotations

import torch
from torch import nn

from ajllm.modeling.moe import MoELayer
from ajllm.training.parallel.common import (
    ParallelContext,
    _world_size,
    all_gather_cat,
    all_reduce_sum,
    broadcast_module_,
)
from ajllm.training.parallel.expert_parallel import ExpertParallelMoE
from ajllm.training.parallel.tensor_parallel import TensorParallelAttention, TensorParallelLinear


class TensorExpertParallel(nn.Module):
    """Wrap a Top-1 MoE Transformer with 2-D expert and tensor parallelism.

    Attention is TP-sharded in every EP replica.  Each MoE layer is replaced
    once with :class:`ExpertParallelMoE`, which owns an EP expert range and TP
    slices the gate, up, and down intermediate channels inside those experts.
    """

    is_tensor_expert_parallel = True

    def __init__(self, module: nn.Module, context: ParallelContext, expert_backend: str = "torch") -> None:
        super().__init__()
        if not hasattr(module, "config") or not hasattr(module, "layers"):
            raise TypeError("TensorExpertParallel requires the project's TransformerLM")
        if module.config.model_type != "moe":
            raise ValueError("TensorExpertParallel requires model_type='moe'")
        self.context = context
        self.rank = context.tp_rank + context.ep_rank * context.tp_size
        self._broadcast_initial_state(module)
        self.module = module
        self._tp_layouts: dict[str, tuple[int, torch.Size]] = {}
        self._expert_layouts: dict[str, tuple[ExpertParallelMoE, str, torch.Size]] = {}
        self._convert_transformer(expert_backend)

    @property
    def config(self):
        return self.module.config

    def _broadcast_initial_state(self, module: nn.Module) -> None:
        broadcast_module_(module)

    def _convert_transformer(self, expert_backend: str) -> None:
        for layer_index, layer in enumerate(self.module.layers):
            # 1. Replace attention with TP-sharded version
            layer.attention = TensorParallelAttention(layer.attention, self.context.tp_group)
            attention_prefix = f"layers.{layer_index}.attention"
            for name, child in layer.attention.named_modules():
                if isinstance(child, TensorParallelLinear):
                    key = f"{attention_prefix}.{name}.weight" if name else f"{attention_prefix}.weight"
                    self._tp_layouts[key] = (child.shard_layout.axis, child.shard_layout.full_shape)

            # 2. Replace MoE feed-forward with EP+TP version
            original_ffn = layer.feed_forward
            if not isinstance(original_ffn, MoELayer) or original_ffn.grouped_experts is None:
                raise TypeError(f"layers.{layer_index}.feed_forward must be the Top-1 grouped MoELayer")
            gate_shape = original_ffn.grouped_experts.gate_up_proj.weight.shape
            down_shape = original_ffn.grouped_experts.down_proj.weight.shape
            layer.feed_forward = ExpertParallelMoE(
                original_ffn,
                self.context.ep_group,
                expert_backend,
                tp_group=self.context.tp_group,
            )
            prefix = f"layers.{layer_index}.feed_forward.grouped_experts"
            adapter = layer.feed_forward
            self._expert_layouts[f"{prefix}.gate_up_proj.weight"] = (adapter, "gate_up", gate_shape)
            self._expert_layouts[f"{prefix}.down_proj.weight"] = (adapter, "down", down_shape)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def auxiliary_loss(self) -> torch.Tensor:
        return self.module.auxiliary_loss()

    def parameter_count(self) -> int:
        count = 0
        for name, parameter in self.module.named_parameters():
            if name in self._tp_layouts:
                count += int(torch.tensor(self._tp_layouts[name][1]).prod())
            elif name in self._expert_layouts:
                count += int(torch.tensor(self._expert_layouts[name][2]).prod())
            else:
                count += parameter.numel()
        return count

    def finish_gradient_synchronization(self) -> None:
        """Average data/EP replicas while retaining TP and EP parameter ownership."""
        for name, parameter in self.module.named_parameters():
            if parameter.grad is None:
                continue
            if name in self._expert_layouts:
                # The EP all-to-all backward already accumulated every source
                # batch for this owner; no EP all-reduce is valid here.
                parameter.grad.div_(self.context.ep_size)
            elif self.context.ep_size > 1:
                parameter.grad.copy_(all_reduce_sum(parameter.grad, self.context.ep_group))
                parameter.grad.div_(self.context.ep_size)

    def clip_grad_norm_(self, maximum_norm: float, epsilon: float = 1e-6) -> float:
        """Compute a logical norm without counting EP/TP replicas twice."""
        squared_norm = torch.zeros((), device=next(self.parameters()).device, dtype=torch.float32)
        for name, parameter in self.module.named_parameters():
            if parameter.grad is None:
                continue
            if name in self._expert_layouts:
                include = True
            elif name in self._tp_layouts:
                include = self.context.ep_rank == 0
            else:
                include = self.context.ep_rank == 0 and self.context.tp_rank == 0
            if include:
                squared_norm.add_(parameter.grad.detach().float().square().sum())
        if _world_size() > 1:
            torch.distributed.all_reduce(squared_norm, op=torch.distributed.ReduceOp.SUM)
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
        state = self.module.state_dict()
        full_state: dict[str, torch.Tensor] = {}
        expert_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for name, value in state.items():
            if name in self._expert_layouts:
                adapter, kind, _ = self._expert_layouts[name]
                cache_key = id(adapter)
                if cache_key not in expert_cache:
                    expert_cache[cache_key] = adapter.full_expert_weights()
                full_state[name] = expert_cache[cache_key][0 if kind == "gate_up" else 1]
            elif name in self._tp_layouts:
                axis, full_shape = self._tp_layouts[name]
                full_state[name] = all_gather_cat(value, axis, self.context.tp_group).reshape(full_shape).clone()
            else:
                full_state[name] = value.detach().clone()
        return full_state

    @torch.no_grad()
    def load_full_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        local_state = self.module.state_dict()
        missing = set(local_state) - set(state_dict)
        unexpected = set(state_dict) - set(local_state)
        if missing or unexpected:
            raise ValueError(f"Checkpoint keys differ; missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        loaded_experts: set[int] = set()
        for name, destination in local_state.items():
            if name in self._expert_layouts:
                adapter, kind, _ = self._expert_layouts[name]
                if id(adapter) not in loaded_experts:
                    prefix = name.rsplit(".", 2)[0]
                    adapter.load_full_expert_weights(
                        state_dict[f"{prefix}.gate_up_proj.weight"].to(destination.device, dtype=destination.dtype),
                        state_dict[f"{prefix}.down_proj.weight"].to(destination.device, dtype=destination.dtype),
                    )
                    loaded_experts.add(id(adapter))
                continue
            source = state_dict[name].to(destination.device, dtype=destination.dtype)
            if name in self._tp_layouts:
                axis, full_shape = self._tp_layouts[name]
                shard_size = full_shape[axis] // self.context.tp_size
                source = source.narrow(axis, self.context.tp_rank * shard_size, shard_size)
            if source.shape != destination.shape:
                raise ValueError(f"Checkpoint tensor {name} has shape {source.shape}, expected {destination.shape}")
            destination.copy_(source)


def tensor_expert_parallelize(
    module: nn.Module,
    context: ParallelContext,
    expert_backend: str = "torch",
) -> TensorExpertParallel:
    return TensorExpertParallel(module, context, expert_backend)
