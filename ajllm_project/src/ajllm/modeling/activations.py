"""Feed-forward building blocks used by both dense and MoE blocks."""

from __future__ import annotations

import torch
from torch import nn

from ajllm.modeling.cuda_kernels import swiglu_gate, variable_grouped_gemm
from ajllm.modeling.layers import Linear


def silu(inputs: torch.Tensor) -> torch.Tensor:
    """Elementwise SiLU written from its defining equation."""
    return inputs * torch.sigmoid(inputs)


class SwiGLU(nn.Module):
    """Bias-free SwiGLU: ``down(silu(gate) * up)``."""

    def __init__(self, d_model: int, d_ff: int, use_cuda_kernels: bool = True) -> None:
        super().__init__()
        self.gate_proj = Linear(d_model, d_ff)
        self.up_proj = Linear(d_model, d_ff)
        self.down_proj = Linear(d_ff, d_model)
        self.use_cuda_kernels = use_cuda_kernels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_proj(inputs), self.up_proj(inputs)
        hidden = swiglu_gate(gate, up) if self.use_cuda_kernels else silu(gate) * up
        return self.down_proj(hidden)


class VariableGroupedLinear(Linear):
    """Bias-free expert weights evaluated from compact Variable-M token ranges."""

    def __init__(
        self, num_experts: int, in_features: int, out_features: int, use_cuda_kernels: bool = True
    ) -> None:
        super().__init__(in_features, out_features)
        self.num_experts = num_experts
        self.use_cuda_kernels = use_cuda_kernels
        # One contiguous tensor lets each Triton tile select its expert by offset.
        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(
        self,
        inputs: torch.Tensor,
        expert_offsets: torch.Tensor,
        tile_experts: torch.Tensor,
        tile_rows: torch.Tensor,
    ) -> torch.Tensor:
        """Project compact ``[tokens, in]`` rows using their expert intervals."""
        if inputs.ndim != 2 or inputs.shape[-1] != self.in_features:
            raise ValueError("VariableGroupedLinear expects [tokens, in_features]")
        # Custom Functions are outside autocast's built-in operator list, so
        # explicitly use the active CUDA compute dtype for Tensor Core GEMM.
        if torch.is_autocast_enabled("cuda"):
            compute_dtype = torch.get_autocast_dtype("cuda")
            inputs = inputs.to(compute_dtype)
            weight = self.weight.to(compute_dtype)
        else:
            weight = self.weight
        if self.use_cuda_kernels:
            return variable_grouped_gemm(inputs, weight, expert_offsets, tile_experts, tile_rows)
        offsets = expert_offsets.detach().cpu()
        outputs = [
            inputs[int(offsets[expert]) : int(offsets[expert + 1])] @ weight[expert].transpose(0, 1)
            for expert in range(self.num_experts)
        ]
        return torch.cat(outputs, dim=0)


class VariableGroupedSwiGLUExperts(nn.Module):
    """SwiGLU experts over a compact token matrix and Variable-M offsets."""

    def __init__(self, num_experts: int, d_model: int, d_ff: int, use_cuda_kernels: bool = True) -> None:
        super().__init__()
        # Gate and up share one grouped GEMM and one contiguous weight tensor.
        self.gate_up_proj = VariableGroupedLinear(num_experts, d_model, 2 * d_ff, use_cuda_kernels)
        self.down_proj = VariableGroupedLinear(num_experts, d_ff, d_model, use_cuda_kernels)
        self.d_ff = d_ff
        self.use_cuda_kernels = use_cuda_kernels

    def forward(
        self,
        inputs: torch.Tensor,
        expert_offsets: torch.Tensor,
        tile_experts: torch.Tensor,
        tile_rows: torch.Tensor,
    ) -> torch.Tensor:
        """Run fused gate/up and down Variable-M grouped GEMMs."""
        gate_up = self.gate_up_proj(inputs, expert_offsets, tile_experts, tile_rows)
        gate, up = gate_up[..., : self.d_ff], gate_up[..., self.d_ff :]
        hidden = swiglu_gate(gate, up) if self.use_cuda_kernels else silu(gate) * up
        return self.down_proj(hidden, expert_offsets, tile_experts, tile_rows)
