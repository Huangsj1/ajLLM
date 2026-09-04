"""Grouped-query causal attention using PyTorch SDPA."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from ajllm.modeling.layers import Linear, RMSNorm


class RotaryPositionalEmbedding(nn.Module):
    """RoPE cache, indexed by the requested absolute positions."""

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
        cosine, sine = cosine.to(inputs.dtype), sine.to(inputs.dtype)
        even = inputs[..., ::2]
        odd = inputs[..., 1::2]
        rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1)
        return rotated.flatten(-2)


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
        use_flash_attention: bool = True,
        dropout: float = 0.0,
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
        self.use_flash_attention = use_flash_attention
        self._custom_flash_failed = False
        self.dropout = dropout

        # Q/K/V projections
        self.q_proj = Linear(d_model, num_heads * self.head_dim)
        self.k_proj = Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = Linear(d_model, num_kv_heads * self.head_dim)
        self.output_proj = Linear(d_model, d_model)

        # Q/K RMSNorm (applied per head)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # RoPE
        self.rope = RotaryPositionalEmbedding(rope_theta, self.head_dim, max_position_embeddings)

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

        if self.use_flash_attention and not self._custom_flash_failed:
            try:
                # Prefer this project's Torch/Triton FlashAttention code.
                # Triton's block-pointer path requires a power-of-two head
                # width. The default 64-wide configuration uses it on CUDA.
                from ajllm.modeling.flash_attention import flash_attention

                use_triton = inputs.is_cuda and self.head_dim > 0 and (self.head_dim & (self.head_dim - 1)) == 0
                attended = flash_attention(queries, keys, values, is_causal=True, use_triton=use_triton)
            except Exception:
                # A custom backend must never make basic pre-training fail.
                # Do not retry a known-broken kernel on every subsequent step.
                self._custom_flash_failed = True
                attended = self._sdpa_fallback(queries, keys, values)
        else:
            attended = self._sdpa_fallback(queries, keys, values)
        output = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.output_proj(output)

    def _sdpa_fallback(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Use SDPA only when custom FlashAttention is disabled or fails."""
        context = nullcontext()
        if queries.is_cuda:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            context = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
        with context:
            return F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
