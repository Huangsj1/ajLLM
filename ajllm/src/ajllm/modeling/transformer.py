"""The single decoder-only language model implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from ajllm.modeling.activations import SwiGLU
from ajllm.modeling.attention import GroupedQueryAttention
from ajllm.modeling.layers import Embedding, Linear, RMSNorm
from ajllm.modeling.moe import MoELayer


@dataclass(frozen=True)
class ModelConfig:
    """Validated configuration for a compact decoder-only model."""

    vocab_size: int
    context_length: int = 512
    max_position_embeddings: int = 32_768
    d_model: int = 768
    num_layers: int = 8
    num_heads: int = 12
    num_kv_heads: int = 4
    d_ff: int | None = None
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    qk_norm: bool = True
    tie_embeddings: bool = True
    model_type: str = "dense"
    num_experts: int = 4
    num_experts_per_token: int = 1
    router_aux_loss_coef: float = 5e-4
    use_flash_attention: bool = True
    use_cuda_kernels: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.context_length <= 0 or self.num_layers <= 0:
            raise ValueError("vocab_size, context_length, and num_layers must be positive")
        if self.context_length > self.max_position_embeddings:
            raise ValueError("context_length cannot exceed max_position_embeddings")
        if self.d_model % self.num_heads or self.num_heads % self.num_kv_heads:
            raise ValueError("d_model must divide by num_heads and num_heads by num_kv_heads")
        if (self.d_model // self.num_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.model_type not in {"dense", "moe"}:
            raise ValueError("model_type must be 'dense' or 'moe'")

    @property
    def intermediate_size(self) -> int:
        """Use the SwiGLU-equivalent width, rounded for efficient matmuls."""
        return self.d_ff or math.ceil(self.d_model * math.pi / 64) * 64


class TransformerBlock(nn.Module):
    """Pre-norm attention and feed-forward residual block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rms_norm_eps, use_cuda_kernels=config.use_cuda_kernels)
        self.attention = GroupedQueryAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            qk_norm=config.qk_norm,
            dropout=config.dropout,
            use_flash_attention=config.use_flash_attention,
            use_cuda_kernels=config.use_cuda_kernels,
        )
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps, use_cuda_kernels=config.use_cuda_kernels)
        self.feed_forward: nn.Module
        if config.model_type == "moe":
            self.feed_forward = MoELayer(
                config.d_model,
                config.intermediate_size,
                config.num_experts,
                config.num_experts_per_token,
                config.router_aux_loss_coef,
                use_cuda_kernels=config.use_cuda_kernels,
            )
        else:
            self.feed_forward = SwiGLU(config.d_model, config.intermediate_size, config.use_cuda_kernels)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.dropout(self.attention(self.attn_norm(hidden_states), positions))
        return hidden_states + self.dropout(self.feed_forward(self.ffn_norm(hidden_states)))


class TransformerLM(nn.Module):
    """Compact causal language model with optional sparse MoE FFNs."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps, use_cuda_kernels=config.use_cuda_kernels)
        self.lm_head = None if config.tie_embeddings else Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return vocabulary logits for ``[batch, sequence]`` token IDs."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.context_length:
            raise ValueError(f"Input length {sequence_length} exceeds context_length")
        positions = torch.arange(sequence_length, device=input_ids.device).expand(input_ids.shape[0], -1)
        hidden_states = self.token_embeddings(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions)
        hidden_states = self.final_norm(hidden_states)
        if self.lm_head is not None:
            return self.lm_head(hidden_states)
        return hidden_states @ self.token_embeddings.weight.transpose(0, 1)

    def auxiliary_loss(self) -> torch.Tensor:
        """Sum current MoE router losses; dense models return a scalar zero."""
        losses = [layer.feed_forward.aux_loss for layer in self.layers if isinstance(layer.feed_forward, MoELayer)]
        losses = [loss for loss in losses if loss is not None]
        if not losses:
            return self.token_embeddings.weight.new_zeros(())
        return torch.stack(losses).sum()

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
