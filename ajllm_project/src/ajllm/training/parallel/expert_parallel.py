"""Expert parallel wrapper for the project's Top-1 grouped MoE implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from ajllm.modeling.activations import VariableGroupedSwiGLUExperts
from ajllm.modeling.cuda_kernels import build_variable_m_schedule, moe_top1_combine, moe_top1_dispatch
from ajllm.modeling.moe import MoELayer
from ajllm.training.parallel.common import (
    _dist_is_ready,
    _rank,
    _world_size,
    all_gather_cat,
    broadcast_module_,
    copy_to_tensor_parallel,
    reduce_from_tensor_parallel,
)


@dataclass(frozen=True)
class _ExpertShardLayout:
    full_shape: torch.Size


class _AllToAll(torch.autograd.Function):
    """Variable-split all-to-all whose backward performs the inverse exchange."""

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        input_splits: tuple[int, ...],
        output_splits: tuple[int, ...],
        group: dist.ProcessGroup | None,
    ) -> torch.Tensor:
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.group = group
        output = inputs.new_empty((sum(output_splits), *inputs.shape[1:]))
        if _world_size(group) == 1:
            output.copy_(inputs)
        else:
            dist.all_to_all_single(
                output,
                inputs.contiguous(),
                output_split_sizes=list(output_splits),
                input_split_sizes=list(input_splits),
                group=group,
            )
        return output

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        inputs_gradient = gradient.new_empty((sum(ctx.input_splits), *gradient.shape[1:]))
        if _world_size(ctx.group) == 1:
            inputs_gradient.copy_(gradient)
        else:
            dist.all_to_all_single(
                inputs_gradient,
                gradient.contiguous(),
                output_split_sizes=list(ctx.input_splits),
                input_split_sizes=list(ctx.output_splits),
                group=ctx.group,
            )
        return inputs_gradient, None, None, None


def _exchange_counts(send_counts: torch.Tensor, group: dist.ProcessGroup | None) -> tuple[int, ...]:
    """Return the rows this rank receives from every source rank."""
    world_size = _world_size(group)
    if world_size == 1:
        return (int(send_counts.item()),)
    gathered = [torch.empty_like(send_counts) for _ in range(world_size)]
    dist.all_gather(gathered, send_counts, group=group)
    rank = _rank(group)
    return tuple(int(counts[rank].item()) for counts in gathered)


def _all_to_all_metadata(
    inputs: torch.Tensor,
    input_splits: tuple[int, ...],
    output_splits: tuple[int, ...],
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """Non-differentiable counterpart used for expert-local integer IDs."""
    output = inputs.new_empty((sum(output_splits), *inputs.shape[1:]))
    if _world_size(group) == 1:
        output.copy_(inputs)
    else:
        dist.all_to_all_single(
            output,
            inputs.contiguous(),
            output_split_sizes=list(output_splits),
            input_split_sizes=list(input_splits),
            group=group,
        )
    return output


class ExpertParallelMoE(MoELayer):
    """Top-1 MoE with EP ownership and optional TP inside every local expert."""

    def __init__(
        self,
        module: MoELayer,
        group: dist.ProcessGroup | None,
        expert_backend: str,
        tp_group: dist.ProcessGroup | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.group = group
        self.world_size = _world_size(group)
        self.rank = _rank(group)
        self.tp_group = tp_group
        # ``None`` is the default global group in torch.distributed.  For the
        # standalone EP adapter it instead means that no TP dimension exists.
        self.tp_size = _world_size(tp_group) if tp_group is not None else 1
        self.tp_rank = _rank(tp_group) if tp_group is not None else 0
        if module.num_experts_per_token != 1:
            raise ValueError("ExpertParallel currently supports the project's Top-1 grouped MoE path only")
        if module.num_experts % self.world_size:
            raise ValueError(f"num_experts={module.num_experts} must divide ep_size={self.world_size}")
        if expert_backend not in {"torch", "triton"}:
            raise ValueError("parallel.expert_backend must be 'torch' or 'triton'")
        if expert_backend == "triton" and not module.use_cuda_kernels:
            raise ValueError("expert_backend='triton' requires model use_cuda_kernels: true")
        assert module.grouped_experts is not None
        if module.grouped_experts.d_ff % self.tp_size:
            raise ValueError(f"d_ff={module.grouped_experts.d_ff} must divide tp_size={self.tp_size}")
        self.d_model = module.d_model
        self.num_experts = module.num_experts
        self.num_experts_per_token = module.num_experts_per_token
        self.router_aux_loss_coef = module.router_aux_loss_coef
        self.use_cuda_kernels = expert_backend == "triton"
        self.expert_backend = expert_backend
        # 1. EP: split experts
        self.local_expert_count = module.num_experts // self.world_size
        self.expert_start = self.rank * self.local_expert_count
        self.global_d_ff = module.grouped_experts.d_ff
        # 2. TP: split d_ff chanel
        self.local_d_ff = self.global_d_ff // self.tp_size
        self.ff_start = self.tp_rank * self.local_d_ff
        self.router = module.router
        expert_weight = module.grouped_experts.gate_up_proj.weight
        self.grouped_experts = VariableGroupedSwiGLUExperts(
            self.local_expert_count,
            module.d_model,
            self.local_d_ff,
            use_cuda_kernels=self.use_cuda_kernels,
        ).to(device=expert_weight.device, dtype=expert_weight.dtype)
        with torch.no_grad():
            expert_slice = slice(self.expert_start, self.expert_start + self.local_expert_count)
            source_gate_up = module.grouped_experts.gate_up_proj.weight[expert_slice]
            local_gate_up = torch.cat(
                (
                    # gate part
                    source_gate_up[:, self.ff_start : self.ff_start + self.local_d_ff],
                    # up part
                    source_gate_up[
                        :, self.global_d_ff + self.ff_start : self.global_d_ff + self.ff_start + self.local_d_ff
                    ],
                ),
                dim=1,
            )
            self.grouped_experts.gate_up_proj.weight.copy_(local_gate_up)
            self.grouped_experts.down_proj.weight.copy_(
                module.grouped_experts.down_proj.weight[
                    expert_slice, :, self.ff_start : self.ff_start + self.local_d_ff
                ]
            )
        self.experts = None
        self.aux_loss: torch.Tensor | None = None

    def _local_expert_forward(self, received: torch.Tensor, local_routes: torch.Tensor) -> torch.Tensor:
        """Evaluate local experts after the all-to-all, restoring receive order. 
            almost same as MoELayer._forward_grouped_top1"""
        if received.shape[0] == 0:
            return received
        # sort according to local expert routes
        order = torch.argsort(local_routes)
        sorted_tokens = torch.arange(received.shape[0], device=received.device).index_select(0, order)
        counts = torch.bincount(local_routes, minlength=self.local_expert_count)
        offsets = torch.cat((counts.new_zeros(1), torch.cumsum(counts, dim=0)))
        tile_experts, tile_rows = build_variable_m_schedule(offsets)
        compact_inputs = (
            moe_top1_dispatch(received, sorted_tokens)
            if self.use_cuda_kernels
            else received.index_select(0, sorted_tokens)
        )
        if self.tp_size > 1:
            compact_inputs = copy_to_tensor_parallel(compact_inputs, self.tp_group)
        compact_outputs = self.grouped_experts(compact_inputs, offsets, tile_experts, tile_rows)
        if self.tp_size > 1:
            compact_outputs = reduce_from_tensor_parallel(compact_outputs, self.tp_group)
        if self.use_cuda_kernels:
            return moe_top1_combine(compact_outputs, sorted_tokens)
        return compact_outputs[torch.argsort(sorted_tokens)]

    @torch.no_grad()
    def full_expert_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather local TP channels and then EP expert ranges into original tensors."""
        local_gate_up = self.grouped_experts.gate_up_proj.weight
        gate = local_gate_up[:, : self.local_d_ff]
        up = local_gate_up[:, self.local_d_ff :]
        if self.tp_size > 1:
            gate = all_gather_cat(gate, 1, self.tp_group)
            up = all_gather_cat(up, 1, self.tp_group)
        gate_up = all_gather_cat(torch.cat((gate, up), dim=1), 0, self.group)
        down = self.grouped_experts.down_proj.weight
        if self.tp_size > 1:
            down = all_gather_cat(down, 2, self.tp_group)
        down = all_gather_cat(down, 0, self.group)
        return gate_up, down

    @torch.no_grad()
    def load_full_expert_weights(self, gate_up: torch.Tensor, down: torch.Tensor) -> None:
        expert_slice = slice(self.expert_start, self.expert_start + self.local_expert_count)
        local_gate_up = torch.cat(
            (
                gate_up[expert_slice, self.ff_start : self.ff_start + self.local_d_ff],
                gate_up[
                    expert_slice,
                    self.global_d_ff + self.ff_start : self.global_d_ff + self.ff_start + self.local_d_ff,
                ],
            ),
            dim=1,
        )
        self.grouped_experts.gate_up_proj.weight.copy_(local_gate_up)
        self.grouped_experts.down_proj.weight.copy_(
            down[expert_slice, :, self.ff_start : self.ff_start + self.local_d_ff]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, d_model = inputs.shape
        inputs_flat = inputs.reshape(-1, d_model)
        router_logits = self.router(inputs_flat)
        # global expert routes 
        global_routes = torch.argmax(router_logits, dim=-1)
        if self.training:
            router_probs = torch.softmax(router_logits, dim=-1)
            self.aux_loss = self._compute_aux_loss(router_probs, global_routes[:, None])
        else:
            self.aux_loss = inputs.new_zeros(())

        # get the owner rank for each token
        owners = torch.div(global_routes, self.local_expert_count, rounding_mode="floor")
        source_order = torch.argsort(owners)
        # Count rows sent to each owner and rows received from each source.
        send_counts = torch.bincount(owners, minlength=self.world_size)
        input_splits = tuple(int(value) for value in send_counts.tolist())
        # get current rank's all input counts from all ranks, e.g. [3,2,4,1] for 4 ranks with 3,2,4,1 rows
        output_splits = _exchange_counts(send_counts, self.group)
        # sort the inputs according to the owner rank
        packed_inputs = inputs_flat.index_select(0, source_order)
        # sorted local expert routes according to the owner rank
        packed_local_routes = (global_routes - owners * self.local_expert_count).index_select(0, source_order)
        # 1. get the current rank's inputs from all ranks
        received_inputs = _AllToAll.apply(packed_inputs, input_splits, output_splits, self.group)
        # 2. get the current rank's local expert routes from all ranks
        received_local_routes = _all_to_all_metadata(packed_local_routes, input_splits, output_splits, self.group)
        # 3. compute the current rank's local expert outputs
        received_outputs = self._local_expert_forward(received_inputs, received_local_routes)
        # 4. send the current rank's local expert outputs back to the original owner ranks
        returned_outputs = _AllToAll.apply(received_outputs, output_splits, input_splits, self.group)
        output_flat = returned_outputs[torch.argsort(source_order)]
        return output_flat.view(batch_size, sequence_length, d_model)


class ExpertParallel(nn.Module):
    """Wrap an existing Top-1 MoE ``TransformerLM`` with expert parallelism."""

    is_expert_parallel = True

    def __init__(
        self,
        module: nn.Module,
        group: dist.ProcessGroup | None = None,
        expert_backend: str = "triton",
    ) -> None:
        super().__init__()
        self.group = group
        self.world_size = _world_size(group)
        self.rank = _rank(group)
        if self.world_size > 1 and not _dist_is_ready():
            raise RuntimeError("ExpertParallel requires torch.distributed to be initialized")
        if not hasattr(module, "config") or not hasattr(module, "layers"):
            raise TypeError("ExpertParallel requires the project's TransformerLM")
        if module.config.model_type != "moe":
            raise ValueError("ExpertParallel requires a model_type='moe' model")
        self._broadcast_initial_state(module)
        self.module = module
        self._layouts: dict[str, _ExpertShardLayout] = {}
        self._convert_moe_layers(expert_backend)

    @property
    def config(self):
        return self.module.config

    def _broadcast_initial_state(self, module: nn.Module) -> None:
        if self.world_size == 1:
            return
        broadcast_module_(module, self.group)

    def _convert_moe_layers(self, expert_backend: str) -> None:
        for layer_index, layer in enumerate(self.module.layers):
            original = layer.feed_forward
            if not isinstance(original, MoELayer):
                raise TypeError(f"layers.{layer_index}.feed_forward is not MoELayer")
            full_gate_up_shape = original.grouped_experts.gate_up_proj.weight.shape  # type: ignore[union-attr]
            full_down_shape = original.grouped_experts.down_proj.weight.shape  # type: ignore[union-attr]
            layer.feed_forward = ExpertParallelMoE(original, self.group, expert_backend)
            prefix = f"layers.{layer_index}.feed_forward.grouped_experts"
            self._layouts[f"{prefix}.gate_up_proj.weight"] = _ExpertShardLayout(full_gate_up_shape)
            self._layouts[f"{prefix}.down_proj.weight"] = _ExpertShardLayout(full_down_shape)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def auxiliary_loss(self) -> torch.Tensor:
        return self.module.auxiliary_loss()

    def parameter_count(self) -> int:
        count = 0
        for name, parameter in self.module.named_parameters():
            layout = self._layouts.get(name)
            count += int(torch.tensor(layout.full_shape).prod()) if layout else parameter.numel()
        return count

    def finish_gradient_synchronization(self) -> None:
        """Average replicated gradients and scale already aggregated local expert gradients."""
        for name, parameter in self.module.named_parameters():
            if parameter.grad is None:
                continue
            if name in self._layouts:
                parameter.grad.div_(self.world_size)
            elif self.world_size > 1:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=self.group)
                parameter.grad.div_(self.world_size)

    def clip_grad_norm_(self, maximum_norm: float, epsilon: float = 1e-6) -> float:
        """Clip one logical MoE model norm without double-counting replicated tensors."""
        squared_norm = torch.zeros((), device=next(self.parameters()).device, dtype=torch.float32)
        for name, parameter in self.module.named_parameters():
            if parameter.grad is None or (name not in self._layouts and self.rank != 0):
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
        """Reconstruct the standard MoE state, including every expert, on all ranks."""
        state = self.module.state_dict()
        full_state: dict[str, torch.Tensor] = {}
        for name, value in state.items():
            layout = self._layouts.get(name)
            if layout is None:
                full_state[name] = value.detach().clone()
                continue
            full_state[name] = all_gather_cat(value, 0, self.group).reshape(layout.full_shape).clone()
        return full_state

    @torch.no_grad()
    def load_full_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        local_state = self.module.state_dict()
        missing = set(local_state) - set(state_dict)
        unexpected = set(state_dict) - set(local_state)
        if missing or unexpected:
            raise ValueError(f"Checkpoint keys differ; missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        for name, destination in local_state.items():
            source = state_dict[name].to(destination.device, dtype=destination.dtype)
            layout = self._layouts.get(name)
            if layout is not None:
                shard_size = layout.full_shape[0] // self.world_size
                source = source.narrow(0, self.rank * shard_size, shard_size)
            if source.shape != destination.shape:
                raise ValueError(f"Checkpoint tensor {name} has shape {source.shape}, expected {destination.shape}")
            destination.copy_(source)


def expert_parallelize(
    module: nn.Module,
    group: dist.ProcessGroup | None = None,
    expert_backend: str = "torch",
) -> ExpertParallel:
    """Construct the expert-parallel wrapper around a Top-1 MoE model."""
    return ExpertParallel(module, group, expert_backend)
