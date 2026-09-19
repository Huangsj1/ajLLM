"""Packed mixed-batch forward with precomputed rotary tables."""

from dataclasses import dataclass

import torch
from torch import nn

from ajvllm.execution.batch import KVCache, ModelBatch
from ajvllm.modeling.qwen2.config import Qwen2Config
from ajvllm.modeling.qwen2.layers import DecoderLayer, RMSNorm, rotary_factors


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor | None
    cache: KVCache


@dataclass(frozen=True)
class BatchOutput:
    logits: torch.Tensor | None
    caches: tuple[KVCache, ...]


class Decoder(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.model = Decoder(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_weights()
        self.register_buffer("rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)
        self._init_rope()

    def _init_rope(self) -> None:
        positions = torch.arange(self.config.max_position_embeddings, device=self.device)
        self.rope_cos, self.rope_sin = rotary_factors(positions, self.config, self.dtype)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse)
        # Rebuild once after dtype/device changes, including meta -> CUDA loading.
        # This also avoids converting already-rounded BF16 factors back to FP32.
        if self.device.type != "meta":
            self._init_rope()
        return result

    def tie_weights(self) -> None:
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @property
    def device(self) -> torch.device:
        return self.model.embed_tokens.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.model.embed_tokens.weight.dtype

    @torch.inference_mode()
    def forward(
        self, input_ids: torch.Tensor | ModelBatch, cache: KVCache | None = None, *, logits_to_keep: int | None = 0
    ) -> ModelOutput | BatchOutput:
        """The runner passes ModelBatch; a 1D tensor remains a numerical-reference convenience."""
        single = isinstance(input_ids, torch.Tensor)
        if single:
            if input_ids.ndim != 1 or not input_ids.numel():
                raise ValueError("input_ids must be a nonempty 1D tensor")
            batch = ModelBatch.build([input_ids], [cache], self.device, [])
            if batch.max_context_len > self.config.max_position_embeddings:
                raise ValueError("input exceeds checkpoint context length")
            if logits_to_keep is not None:
                start = 0 if logits_to_keep == 0 else max(0, input_ids.numel() - logits_to_keep)
                batch.sample_indices = torch.arange(start, input_ids.numel(), device=self.device)
        else:
            batch = input_ids
        x = self.model.embed_tokens(batch.token_ids)
        factors = self.rope_cos[batch.positions], self.rope_sin[batch.positions]
        layers = []
        for index, layer in enumerate(self.model.layers):
            x, present = layer(x, factors, batch, index)
            layers.append(present)
        logits = None
        if batch.sample_indices.numel():
            logits = self.lm_head(self.model.norm(x[batch.sample_indices])).float()
        caches = tuple(tuple(layer[row] for layer in layers) for row in range(batch.num_requests))
        return ModelOutput(logits, caches[0]) if single else BatchOutput(logits, caches)
