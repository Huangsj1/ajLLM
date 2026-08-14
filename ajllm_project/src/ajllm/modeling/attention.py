"""Rotary position embeddings and causal self-attention."""

from __future__ import annotations

import math

import torch
from torch import nn

from ajllm.modeling.layers import Linear


def softmax(inputs: torch.Tensor, dimension: int) -> torch.Tensor:
    """Numerically stable softmax that preserves input dtype."""
    shifted = inputs - torch.amax(inputs, dim=dimension, keepdim=True)
    exponentials = torch.exp(shifted)
    return exponentials / torch.sum(exponentials, dim=dimension, keepdim=True)


def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention with optional masking."""
    d_k = queries.shape[-1]
    scores = queries @ keys.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probabilities = softmax(scores, -1)
    probabilities = probabilities.nan_to_num_(nan=0.0)
    return probabilities @ values


class RotaryPositionalEmbedding(nn.Module):
    """Precomputed rotary position embedding tables."""

    def __init__(self, theta: float, head_dimension: int, max_sequence_length: int) -> None:
        super().__init__()
        if head_dimension % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        positions = torch.arange(max_sequence_length).unsqueeze(1)
        dimension_indices = torch.arange(0, head_dimension, 2).float()
        frequencies = theta ** (-dimension_indices / head_dimension)
        angles = positions * frequencies
        self.register_buffer("cosine", torch.cos(angles), persistent=False)
        self.register_buffer("sine", torch.sin(angles), persistent=False)

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cosine = self.cosine[positions]
        sine = self.sine[positions]
        # Cast to input dtype to handle mixed precision
        if cosine.dtype != inputs.dtype:
            cosine = cosine.to(inputs.dtype)
            sine = sine.to(inputs.dtype)
        even = inputs[..., ::2]
        odd = inputs[..., 1::2]
        rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1)
        return rotated.flatten(-2)


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional RoPE and FlashAttention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        context_length: int,
        position_encoding: str,
        rope_theta: float,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dimension = d_model // num_heads
        self.use_flash_attention = use_flash_attention
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.rope = (
            RotaryPositionalEmbedding(rope_theta, self.head_dimension, context_length)
            if position_encoding == "rope"
            else None
        )

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = inputs.shape

        def split_heads(projected: torch.Tensor) -> torch.Tensor:
            return projected.view(batch_size, sequence_length, self.num_heads, self.head_dimension).transpose(1, 2)

        queries = split_heads(self.q_proj(inputs))
        keys = split_heads(self.k_proj(inputs))
        values = split_heads(self.v_proj(inputs))
        if self.rope is not None:
            expanded_positions = positions[:, None, :].expand(batch_size, self.num_heads, sequence_length)
            queries = self.rope(queries, expanded_positions)
            keys = self.rope(keys, expanded_positions)

        # Use FlashAttention if enabled and on CUDA
        if self.use_flash_attention and inputs.device.type == "cuda":
            try:
                from ajllm.modeling.flash_attention import flash_attention
                attended = flash_attention(queries, keys, values, is_causal=True, use_triton=True)
            except Exception:
                # Fallback to standard attention on error
                mask = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=inputs.device))
                attended = scaled_dot_product_attention(queries, keys, values, mask)
        else:
            mask = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=inputs.device))
            attended = scaled_dot_product_attention(queries, keys, values, mask)

        joined = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.d_model)
        return self.output_proj(joined)
