"""Grouped-query causal attention using PyTorch SDPA."""

from __future__ import annotations

import torch
from torch import nn

from ajllm.modeling.cuda_kernels import rope
from ajllm.modeling.layers import Linear, RMSNorm


class RotaryPositionalEmbedding(nn.Module):
    """RoPE cache, indexed by the requested absolute positions."""

    def __init__(self, theta: float, head_dimension: int, max_sequence_length: int, use_cuda_kernels: bool) -> None:
        super().__init__()
        if head_dimension % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension")
        positions = torch.arange(max_sequence_length).unsqueeze(1)
        dimension_indices = torch.arange(0, head_dimension, 2).float()
        frequencies = theta ** (-dimension_indices / head_dimension)
        angles = positions * frequencies
        self.register_buffer("cosine", torch.cos(angles), persistent=False)
        self.register_buffer("sine", torch.sin(angles), persistent=False)
        self.use_cuda_kernels = use_cuda_kernels

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.use_cuda_kernels:
            return rope(inputs, self.cosine, self.sine, positions)
        cosine, sine = self.cosine[positions].to(inputs.dtype), self.sine[positions].to(inputs.dtype)
        even, odd = inputs[..., ::2], inputs[..., 1::2]
        return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value tensors to match query head count for GQA."""
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        .reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    """GQA with the project's FlashAttention implementation and SDPA fallback."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        rope_theta: float = 10_000.0,
        qk_norm: bool = True,
        dropout: float = 0.0,
        use_flash_attention: bool = True,
        use_cuda_kernels: bool = True,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention

        # Q/K/V projections
        self.q_proj = Linear(d_model, num_heads * self.head_dim)
        self.k_proj = Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = Linear(d_model, num_kv_heads * self.head_dim)
        self.output_proj = Linear(d_model, d_model)

        # Q/K RMSNorm (applied per head)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, use_cuda_kernels=use_cuda_kernels)
            self.k_norm = RMSNorm(self.head_dim, use_cuda_kernels=use_cuda_kernels)

        # RoPE
        self.rope = RotaryPositionalEmbedding(
            rope_theta, self.head_dim, max_position_embeddings, use_cuda_kernels
        )

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Forward pass with GQA."""
        batch_size, seq_len, _ = inputs.shape

        queries = self.q_proj(inputs).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.k_proj(inputs).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        values = self.v_proj(inputs).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)

        expanded_positions = positions[:, None, :].expand(batch_size, self.num_heads, seq_len)
        queries = self.rope(queries, expanded_positions)

        expanded_kv_positions = positions[:, None, :].expand(batch_size, self.num_kv_heads, seq_len)
        keys = self.rope(keys, expanded_kv_positions)

        keys = repeat_kv(keys, self.num_kv_groups)
        values = repeat_kv(values, self.num_kv_groups)

        from ajllm.modeling.flash_attention import flash_attention, flash_attention_pytorch

        sequence = queries.shape[-2]
        # The tiled FlashAttention kernel uses Tensor Core-compatible sequence
        # tiles. Generation may start from a prompt shorter than one tile, so
        # append causal-invisible zero rows and discard them after attention.
        padded_sequence = max(16, 1 << (sequence - 1).bit_length())
        if self.use_flash_attention and padded_sequence != sequence:
            padding_shape = (*queries.shape[:-2], padded_sequence - sequence, queries.shape[-1])
            zero_queries = queries.new_zeros(padding_shape)
            zero_keys = keys.new_zeros(padding_shape)
            zero_values = values.new_zeros(padding_shape)
            queries = torch.cat((queries, zero_queries), dim=-2)
            keys = torch.cat((keys, zero_keys), dim=-2)
            values = torch.cat((values, zero_values), dim=-2)
        if self.use_flash_attention:
            attended = flash_attention(queries, keys, values, True)[..., :sequence, :]
        else:
            attended = flash_attention_pytorch(queries, keys, values, True)
        output = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.output_proj(output)
