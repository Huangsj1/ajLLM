"""Configurable decoder-only Transformer language model."""

from __future__ import annotations

import torch
from torch import nn

from ajllm.modeling.activations import build_feed_forward
from ajllm.modeling.attention import MultiHeadSelfAttention
from ajllm.modeling.layers import Embedding, Linear, build_norm


class TransformerBlock(nn.Module):
    """Transformer block supporting pre-norm, post-norm, and no-norm variants."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        context_length: int,
        position_encoding: str,
        rope_theta: float,
        normalization: str,
        norm_placement: str,
        feed_forward: str,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        if norm_placement not in {"pre", "post"}:
            raise ValueError("normalization.placement must be 'pre' or 'post'")
        self.norm_placement = norm_placement
        self.norm1 = build_norm(normalization, d_model)
        self.norm2 = build_norm(normalization, d_model)
        self.attention = MultiHeadSelfAttention(
            d_model,
            num_heads,
            context_length,
            position_encoding,
            rope_theta,
            use_flash_attention,
        )
        self.feed_forward = build_feed_forward(feed_forward, d_model, d_ff)

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.norm_placement == "pre":
            hidden = inputs + self.attention(self.norm1(inputs), positions)
            return hidden + self.feed_forward(self.norm2(hidden))
        hidden = self.norm1(inputs + self.attention(inputs, positions))
        return self.norm2(hidden + self.feed_forward(hidden))


class TransformerLM(nn.Module):
    """Decoder-only Transformer with a configurable block architecture."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        position_encoding: str = "rope",
        rope_theta: float = 10_000.0,
        normalization: str = "rmsnorm",
        norm_placement: str = "pre",
        feed_forward: str = "swiglu",
        tie_embeddings: bool = False,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        if position_encoding not in {"rope", "none", "learned"}:
            raise ValueError("position encoding must be rope, none, or learned")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.position_encoding = position_encoding
        self.normalization = normalization
        self.norm_placement = norm_placement
        self.feed_forward_type = feed_forward
        self.tie_embeddings = tie_embeddings

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.position_embeddings = Embedding(context_length, d_model) if position_encoding == "learned" else None
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    position_encoding,
                    rope_theta,
                    normalization,
                    norm_placement,
                    feed_forward,
                    use_flash_attention,
                )
                for _ in range(num_layers)
            ]
        )
        final_norm_type = normalization if norm_placement == "pre" else "none"
        self.final_norm = build_norm(final_norm_type, d_model)
        self.lm_head = None if tie_embeddings else Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = token_ids.shape
        if sequence_length > self.context_length:
            raise ValueError(f"Input length {sequence_length} exceeds context length {self.context_length}")
        positions = torch.arange(sequence_length, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.token_embeddings(token_ids)
        # if use learned position embeddings, add them to the token embeddings
        if self.position_embeddings is not None:
            hidden = hidden + self.position_embeddings(positions)
        for layer in self.layers:
            hidden = layer(hidden, positions)
        hidden = self.final_norm(hidden)
        # if input embeddings are tied to the output embeddings, use the same token embedding matrix
        if self.tie_embeddings:
            return hidden @ self.token_embeddings.weight.transpose(0, 1)
        assert self.lm_head is not None
        return self.lm_head(hidden)
