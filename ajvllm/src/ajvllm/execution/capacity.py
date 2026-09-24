"""Workspace estimates for eager and Triton Qwen2 backends, excluding loaded model weights."""

from dataclasses import dataclass

from ajvllm.config import EngineConfig, MemoryConfig
from ajvllm.modeling.qwen2.config import Qwen2Config


@dataclass(frozen=True)
class Qwen2MemoryEstimate:
    model_config: Qwen2Config
    element_size: int
    engine_config: EngineConfig
    memory_config: MemoryConfig
    compute_backend: str = "eager"
    graph_reserve_bytes: int = 0

    def __call__(self, slots: int, tokens: int) -> int:
        cfg = self.model_config
        size = self.element_size
        context = self.engine_config.max_model_len
        prefill_tokens = min(tokens, self.engine_config.max_prefill_tokens_per_step or tokens)
        query = min(prefill_tokens, self.engine_config.max_prefill_chunk_size or prefill_tokens)
        kv_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * size
        # Old + replacement caches plus packing; reserve all admitted requests to their context limit.
        if self.memory_config.backend == "paged":
            blocks = self.memory_config.num_blocks or slots * (
                (context + self.memory_config.block_size - 1) // self.memory_config.block_size
            )
            pool = blocks * self.memory_config.block_size * kv_per_token
            # Only one layer's dense gather is temporary; all persistent KV lives in the pool.
            kv = pool + slots * context * kv_per_token // cfg.num_hidden_layers
        else:
            kv = 3 * slots * context * kv_per_token
        # Retain the earlier expanded-KV-sized reserve for backend workspace.
        # Grouped eager GQA no longer allocates repeated K/V, but removing this
        # safety margin should follow capacity profiling on longer workloads.
        workspace_reserve = 2 * slots * context * cfg.hidden_size * size
        # Mixed batches pad every request to the largest query/context dimensions,
        # including decode rows. Reserve that workspace even when prefill is capped.
        attention = min(slots, tokens) * cfg.num_attention_heads * query * context * (3 * size + 4)
        if self.compute_backend == "triton":
            kv = pool
            workspace_reserve = 0
            # FP32 split outputs/LSE, bounded by one decode row per sequence.
            splits = (context + 255) // 256
            attention = slots * cfg.num_attention_heads * splits * (cfg.head_dim + 1) * 4
        activations = tokens * (8 * cfg.hidden_size + 4 * cfg.intermediate_size) * size
        # General CUDA sampling keeps FP32 scores/probabilities/CDFs, history
        # buffers, int64 sorted IDs and sorting workspace on device.
        logits = slots * cfg.vocab_size * 64
        return kv + workspace_reserve + attention + activations + logits + self.graph_reserve_bytes
