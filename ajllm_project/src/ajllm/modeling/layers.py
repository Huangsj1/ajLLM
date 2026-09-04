"""Small bias-free layers. Explicit weights keep the teaching FSDP wrapper simple."""

from __future__ import annotations

import torch
from torch import nn


class Linear(nn.Module):
    """Bias-free linear projection."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs @ self.weight.transpose(-2, -1)


class Embedding(nn.Module):
    """Map integer IDs to trainable dense vectors."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned gain."""

    def __init__(
        self,
        d_model: int,
        epsilon: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_dtype = inputs.dtype
        float_inputs = inputs.float()
        rms = torch.sqrt(torch.mean(float_inputs.square(), dim=-1, keepdim=True) + self.epsilon)
        return ((float_inputs / rms) * self.weight.float()).to(input_dtype)
