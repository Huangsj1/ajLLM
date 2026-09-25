"""Conservative workspace bounds supplement measured startup peaks."""

from dataclasses import dataclass

from ajvllm.config import EngineConfig, MemoryConfig
from ajvllm.modeling.qwen2.config import Qwen2Config


@dataclass
class Qwen2MemoryEstimate:
    model_config: Qwen2Config
    element_size: int
    engine_config: EngineConfig
    memory_config: MemoryConfig
    compute_backend: str = "eager"
    graph_reserve_bytes: int = 0

    @property
    def kv_per_token(self):
        cfg = self.model_config
        return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * self.element_size

    def workspace(self, slots: int, tokens: int) -> int:
        """Exclude a paged pool: its size is determined after profiling."""
        cfg, size, context = self.model_config, self.element_size, self.engine_config.max_model_len
        prefill_tokens = min(tokens, self.engine_config.max_prefill_tokens_per_step or tokens)
        query = min(prefill_tokens, self.engine_config.max_prefill_chunk_size or prefill_tokens)
        if self.compute_backend == "triton":
            splits = (context + 255) // 256
            attention = slots * cfg.num_attention_heads * splits * (cfg.head_dim + 1) * 4
            kv_workspace = 0
        else:
            attention = min(slots, tokens) * cfg.num_attention_heads * query * context * (3 * size + 4)
            # Eager comparison gathers one layer; contiguous storage also retains
            # old/replacement histories. These paths need full-context headroom.
            kv_workspace = slots * context * self.kv_per_token // cfg.num_hidden_layers
            if self.memory_config.backend == "contiguous":
                kv_workspace = 3 * slots * context * self.kv_per_token
            kv_workspace += 2 * slots * context * cfg.hidden_size * size
        activations = tokens * (8 * cfg.hidden_size + 4 * cfg.intermediate_size) * size
        # Live scores/sorted IDs/CDF, CUDA sort scratch and transient histories.
        sampling = slots * cfg.vocab_size * 64
        sampling += max(0, self.engine_config.max_num_seqs - slots) * cfg.vocab_size * 12
        return kv_workspace + attention + activations + sampling

    def __call__(self, slots: int, tokens: int) -> int:
        pool = 0
        if self.memory_config.backend == "paged":
            block = self.memory_config.block_size
            blocks = self.memory_config.num_blocks or (self.engine_config.max_model_len + block - 1) // block
            pool = blocks * block * self.kv_per_token
        return pool + self.workspace(slots, tokens) + self.graph_reserve_bytes
