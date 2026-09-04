"""Educational, vectorized Top-K MoE feed-forward layer."""

from __future__ import annotations

import torch
from torch import nn

from ajllm.modeling.activations import SwiGLU
from ajllm.modeling.layers import Linear


class MoELayer(nn.Module):
    """Mixture of Experts layer with Top-K routing and load balancing."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 4,
        num_experts_per_token: int = 1,
        router_aux_loss_coef: float = 5e-4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.router_aux_loss_coef = router_aux_loss_coef
        if not 1 <= num_experts_per_token <= num_experts:
            raise ValueError("num_experts_per_token must be in [1, num_experts]")

        # Router network
        self.router = Linear(d_model, num_experts)

        # Experts
        self.experts = nn.ModuleList([SwiGLU(d_model, d_ff) for _ in range(num_experts)])

        # For tracking auxiliary loss
        self.aux_loss: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Route tokens to experts and aggregate results."""
        batch_size, seq_len, d_model = inputs.shape
        inputs_flat = inputs.reshape(-1, d_model)

        router_logits = self.router(inputs_flat)
        router_probs = torch.softmax(router_logits, dim=-1)

        topk_probs, topk_indices = torch.topk(router_probs, self.num_experts_per_token, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        if self.training:
            self.aux_loss = self._compute_aux_loss(router_probs, topk_indices)
        else:
            self.aux_loss = inputs.new_zeros(())

        output = torch.zeros_like(inputs_flat)
        for expert_idx in range(self.num_experts):
            token_indices, route_indices = torch.where(topk_indices == expert_idx)
            if token_indices.numel() == 0:
                continue
            expert_outputs = self.experts[expert_idx](inputs_flat[token_indices])
            output.index_add_(0, token_indices, expert_outputs * topk_probs[token_indices, route_indices, None])

        return output.view(batch_size, seq_len, d_model)

    def _compute_aux_loss(self, router_probs: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary load balancing loss."""
        mean_probs = router_probs.mean(dim=0)
        assignment = torch.zeros_like(router_probs)
        assignment.scatter_add_(1, topk_indices, torch.ones_like(topk_indices, dtype=router_probs.dtype))
        load = assignment.mean(dim=0) / self.num_experts_per_token
        return self.num_experts * self.router_aux_loss_coef * (mean_probs * load).sum()
