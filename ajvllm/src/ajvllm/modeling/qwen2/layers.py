"""Qwen2 layers with eager reference math and native Triton inference fusions."""

import torch
from torch import nn
from torch.nn import functional as F

from ajvllm.execution.batch import ModelBatch
from ajvllm.kernels.attention import paged_attention
from ajvllm.kernels.elementwise import rms_norm, rope_and_cache, swiglu
from ajvllm.memory.contiguous import update_layer
from ajvllm.memory.types import LayerKV
from ajvllm.modeling.qwen2.config import Qwen2Config


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor, *, optimized=False) -> torch.Tensor:
        if optimized:
            return rms_norm(x, self.weight, self.eps)
        normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def rotary_factors(
    positions: torch.Tensor, config: Qwen2Config, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    # shape = (head_dim/2,)
    frequencies = config.rope_theta ** (
        -torch.arange(0, config.head_dim, 2, dtype=torch.float32, device=positions.device) / config.head_dim
    )
    # shape = (num_positions, head_dim/2), each position's angle for each frequency
    angles = positions.float()[:, None] * frequencies[None, :]
    # shape = (num_positions, head_dim), each position's cos/sin for each frequency, repeated for the two halves of
    # the head dimension
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-second, first), dim=-1) * sin


class Attention(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=True)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self, x: torch.Tensor, factors: tuple[torch.Tensor, torch.Tensor], batch: ModelBatch, layer_index: int
    ) -> tuple[torch.Tensor, tuple[LayerKV, ...]]:
        config = self.config
        # x.shape = (sum(lengths), hidden_size)
        count = x.shape[0]
        # Projections and RoPE operate on ALL packed tokens in one operation.
        q = self.q_proj(x).view(count, config.num_attention_heads, config.head_dim)
        k = self.k_proj(x).view(count, config.num_key_value_heads, config.head_dim)
        v = self.v_proj(x).view(count, config.num_key_value_heads, config.head_dim)
        if batch.attention_metadata is not None:
            # rope qk, store kv to kv cache
            q = rope_and_cache(q, k, v, factors, batch.paged, layer_index)
            packed = paged_attention(q, batch.paged, layer_index, batch.attention_metadata)
            return self.o_proj(packed.reshape(count, config.hidden_size)), ()
        factors = tuple(factor[:, None, :] for factor in factors)
        q, k = apply_rotary(q, *factors), apply_rotary(k, *factors)
        queries = x.new_zeros(batch.num_requests, config.num_attention_heads, batch.max_query_len, config.head_dim)
        queries[batch.sequence_ids, :, batch.query_offsets] = q
        if batch.paged is None:
            keys, values, present = update_layer(batch, layer_index, k, v)
        else:
            keys, values = batch.paged.update(layer_index, k, v)
            present = ()
        groups = config.num_attention_heads // config.num_key_value_heads
        # Put each KV head's query-head group into the GEMM query dimension.
        # This shares K/V directly instead of materializing one copy per query head.
        grouped_queries = queries.view(
            batch.num_requests, config.num_key_value_heads, groups * batch.max_query_len, config.head_dim
        )
        scores = torch.matmul(grouped_queries, keys.transpose(-1, -2)) * config.head_dim**-0.5
        scores = scores.view(batch.num_requests, config.num_attention_heads, batch.max_query_len, batch.max_context_len)
        scores.masked_fill_(batch.causal_mask, torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        grouped_probabilities = probabilities.view(
            batch.num_requests, config.num_key_value_heads, groups * batch.max_query_len, batch.max_context_len
        )
        attended = torch.matmul(grouped_probabilities, values).view(
            batch.num_requests, config.num_attention_heads, batch.max_query_len, config.head_dim
        )
        packed = attended[batch.sequence_ids, :, batch.query_offsets].reshape(count, config.hidden_size)
        return self.o_proj(packed), present


class MLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, *, optimized=False) -> torch.Tensor:
        gate, up = self.gate_proj(x), self.up_proj(x)
        return self.down_proj(swiglu(gate, up) if optimized else F.silu(gate) * up)


class DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(
        self, x: torch.Tensor, factors: tuple[torch.Tensor, torch.Tensor], batch: ModelBatch, layer_index: int
    ) -> tuple[torch.Tensor, tuple[LayerKV, ...]]:
        optimized = batch.attention_metadata is not None
        attention, present = self.self_attn(self.input_layernorm(x, optimized=optimized), factors, batch, layer_index)
        if optimized:
            # calculate residual and norm in one kernel to avoid extra memory allocation and copyS
            norm, residual = rms_norm(
                attention, self.post_attention_layernorm.weight, self.post_attention_layernorm.eps, x
            )
            return residual + self.mlp(norm, optimized=True), present
        x = x + attention
        # return AttentionLayerOutput, current full kv cache
        return x + self.mlp(self.post_attention_layernorm(x)), present
