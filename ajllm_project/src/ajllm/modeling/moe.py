"""Top-K MoE with compact Variable-M grouped GEMMs for the default Top-1 path."""

from __future__ import annotations

import torch
from torch import nn

from ajllm.modeling.activations import SwiGLU, VariableGroupedSwiGLUExperts
from ajllm.modeling.cuda_kernels import (
    build_variable_m_schedule,
    moe_top1_combine,
    moe_top1_dispatch,
)
from ajllm.modeling.layers import Linear


class MoELayer(nn.Module):
    """Route tokens to compact Top-1 ranges and Variable-M grouped GEMMs."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 4,
        num_experts_per_token: int = 1,
        router_aux_loss_coef: float = 5e-4,
        use_cuda_kernels: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.router_aux_loss_coef = router_aux_loss_coef
        self.use_cuda_kernels = use_cuda_kernels
        if not 1 <= num_experts_per_token <= num_experts:
            raise ValueError("num_experts_per_token must be in [1, num_experts]")
        self.router = Linear(d_model, num_experts)
        if num_experts_per_token == 1:
            self.grouped_experts: VariableGroupedSwiGLUExperts | None = VariableGroupedSwiGLUExperts(
                num_experts, d_model, d_ff, use_cuda_kernels
            )
            self.experts: nn.ModuleList | None = None
        else:
            # Top-K>1 retains the readable weighted/atomic-add reference path.
            self.grouped_experts = None
            self.experts = nn.ModuleList([SwiGLU(d_model, d_ff, use_cuda_kernels) for _ in range(num_experts)])

        self.aux_loss: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Route ``[batch, sequence, hidden]`` inputs and restore token order."""
        batch_size, sequence_length, d_model = inputs.shape
        inputs_flat = inputs.reshape(-1, d_model)
        router_logits = self.router(inputs_flat)
        if self.num_experts_per_token == 1:
            # Normalized Top-1 routing weight is exactly one, so expert output
            # depends only on argmax(logits). Softmax is needed only for training
            # load-balancing gradients and deliberately stays in PyTorch for E=4.
            routes = torch.argmax(router_logits, dim=-1)
            if self.training:
                router_probs = torch.softmax(router_logits, dim=-1)
                self.aux_loss = self._compute_aux_loss(router_probs, routes[:, None])
            else:
                self.aux_loss = inputs.new_zeros(())
            output = self._forward_grouped_top1(inputs_flat, routes)
        else:
            output = self._forward_topk_reference(inputs_flat, router_logits)
        return output.view(batch_size, sequence_length, d_model)

    def _forward_grouped_top1(self, inputs: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        """Sort once, execute true Variable-M GEMMs, then restore token order."""
        assert self.grouped_experts is not None
        tokens = inputs.shape[0]
        token_indices = torch.arange(tokens, device=inputs.device)
        order = torch.argsort(routes)
        # sorted indices
        sorted_tokens = token_indices.index_select(0, order)
        # each expert's token counts, like [3,2,4,1]
        counts = torch.bincount(routes, minlength=self.num_experts)
        # each expert's token offsets, like [0,3,5,9,10]
        expert_offsets = torch.cat((counts.new_zeros(1), torch.cumsum(counts, dim=0)))
        # Group input tokens into Variable-M tiles.
        #  `tile_experts` (an array indicating the expert assigned to each tile) e.g. [0,0,1,2,2,3] for 4 experts with 3,2,4,1 rows and M=2
        #  `tile_rows` (the starting position of each tile within the input token sequence) e.g. [0,2,3,5,7,8]
        tile_experts, tile_rows = build_variable_m_schedule(expert_offsets)
        compact_inputs = moe_top1_dispatch(inputs, sorted_tokens) if self.use_cuda_kernels else inputs.index_select(0, sorted_tokens)
        compact_outputs = self.grouped_experts(compact_inputs, expert_offsets, tile_experts, tile_rows)
        if self.use_cuda_kernels:
            return moe_top1_combine(compact_outputs, sorted_tokens)
        output = torch.empty_like(compact_outputs)
        return output.index_copy_(0, sorted_tokens, compact_outputs)

    def _forward_topk_reference(self, inputs: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        """Keep the prior general Top-K path until capacity/expert-parallel policy exists."""
        assert self.experts is not None
        router_probs = torch.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, self.num_experts_per_token, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        if self.training:
            self.aux_loss = self._compute_aux_loss(router_probs, topk_indices)
        else:
            self.aux_loss = inputs.new_zeros(())

        output = torch.zeros_like(inputs)
        for expert_index in range(self.num_experts):
            token_indices, route_indices = torch.where(topk_indices == expert_index)
            if token_indices.numel() == 0:
                continue
            expert_outputs = self.experts[expert_index](inputs[token_indices])
            output.index_add_(0, token_indices, expert_outputs * topk_probs[token_indices, route_indices, None])
        return output

    def _compute_aux_loss(self, router_probs: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary load balancing loss from the unchanged router probabilities."""
        mean_probs = router_probs.mean(dim=0)
        assignment = torch.zeros_like(router_probs)
        assignment.scatter_add_(1, topk_indices, torch.ones_like(topk_indices, dtype=router_probs.dtype))
        load = assignment.mean(dim=0) / self.num_experts_per_token
        return self.num_experts * self.router_aux_loss_coef * (mean_probs * load).sum()
