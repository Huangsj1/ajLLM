"""Feed-forward building blocks used by both dense and MoE blocks."""

from __future__ import annotations

import torch
from torch import nn

from ajllm.modeling.layers import Linear


def silu(inputs: torch.Tensor) -> torch.Tensor:
    """Elementwise SiLU written from its defining equation."""
    return inputs * torch.sigmoid(inputs)


class SwiGLU(nn.Module):
    """Bias-free SwiGLU: ``down(silu(gate) * up)``."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = Linear(d_model, d_ff)
        self.up_proj = Linear(d_model, d_ff)
        self.down_proj = Linear(d_ff, d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down_proj(silu(self.gate_proj(inputs)) * self.up_proj(inputs))
