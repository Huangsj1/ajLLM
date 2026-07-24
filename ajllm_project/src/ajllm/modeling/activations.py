"""Feed-forward activations and gated networks."""

from __future__ import annotations

import torch
from torch import nn

from ajllm.modeling.layers import Linear


def silu(inputs: torch.Tensor) -> torch.Tensor:
    return inputs * torch.sigmoid(inputs)


class SwiGLU(nn.Module):
    """SwiGLU(x) = W2(SiLU(W1x) * W3x)."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(inputs)) * self.w3(inputs))


class SiLUFeedForward(nn.Module):
    """Two-projection feed-forward network with SiLU activation."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(inputs)))


def build_feed_forward(kind: str, d_model: int, d_ff: int) -> nn.Module:
    if kind == "swiglu":
        return SwiGLU(d_model, d_ff)
    if kind == "silu":
        return SiLUFeedForward(d_model, d_ff)
    raise ValueError(f"Unsupported feed-forward type: {kind}")
