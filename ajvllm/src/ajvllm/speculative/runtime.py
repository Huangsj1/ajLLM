"""Joint model/workspace profiling and paired KV-pool allocation."""

from dataclasses import replace
from pathlib import Path

from tokenizers import Tokenizer

from ajvllm.attention.backends.triton import resolve_backend
from ajvllm.config import ComputeConfig, GraphConfig, MemoryConfig, QuantizationConfig
from ajvllm.engine.core import Engine
from ajvllm.execution.qwen2 import Qwen2Runner
from ajvllm.modeling.qwen2.projections import pack_projections
from ajvllm.modeling.qwen2.weights import load_qwen2
from ajvllm.quantization.linear import quantize_model
from ajvllm.runtime.budget import MemoryBudget
from ajvllm.runtime.profiling import profile_memory
from ajvllm.speculative.decoder import SpeculativeDecoder


def check_tokenizers(target, draft):
    # Equal vocabulary size alone does not establish equal token semantics.
    a = Tokenizer.from_file(str(Path(target) / "tokenizer.json"))
    b = Tokenizer.from_file(str(Path(draft) / "tokenizer.json"))
    if a.get_vocab() != b.get_vocab():
        raise ValueError("speculative decoding requires identical target/draft token-ID vocabularies")


def create_runtime(
    runtime_cls,
    model,
    draft,
    config,
    speculative,
    *,
    memory_config=None,
    compute_config=None,
    graph_config=None,
    quantization_config=None,
    eos_token_ids=(),
    **budget_options,
):
    memory = memory_config or MemoryConfig()
    graphs = graph_config or GraphConfig()
    if memory.backend != "paged":
        raise ValueError("speculative decoding requires paged KV")
    if model.config.vocab_size != draft.config.vocab_size:
        raise ValueError("target and draft vocabularies must match")
    if config.max_model_len > min(model.config.max_position_embeddings, draft.config.max_position_embeddings):
        raise ValueError("context limit exceeds target or draft model capacity")
    quantize_model(model, quantization_config or QuantizationConfig())
    quantize_model(draft, quantization_config or QuantizationConfig())
    backends = [
        ComputeConfig(backend=resolve_backend(compute_config or ComputeConfig(), m, memory)) for m in (model, draft)
    ]
    for m, backend in zip((model, draft), backends, strict=True):
        if backend.backend == "triton":
            pack_projections(m)
    planner = MemoryBudget(model, config, graph_reserve_bytes=2 * graphs.reserve_bytes, **budget_options)
    stats = planner.stats
    stats.draft_model_bytes = sum(t.numel() * t.element_size() for t in (*draft.parameters(), *draft.buffers()))
    stats.model_bytes += stats.draft_model_bytes
    stats.weight_bytes += sum(t.numel() * t.element_size() for t in draft.parameters())
    stats.buffer_bytes += sum(t.numel() * t.element_size() for t in draft.buffers())
    profiles = [
        profile_memory(m, config, memory, backend, planner) for m, backend in zip((model, draft), backends, strict=True)
    ]
    peak, temporary, _, _ = max(profiles, key=lambda p: p[0] - p[1])
    slots = min(config.max_num_seqs, config.max_num_batched_tokens)
    # Retained p/q/verification logits, speculative histories, FP64 CDF scratch.
    extra = slots * model.config.vocab_size * (4 * speculative.num_draft_tokens + 16) * 4
    stats.speculative_workspace_bytes = extra
    target_block_bytes, draft_block_bytes = (profile[3] for profile in profiles)
    # Each logical block needs storage in both models. Divide the remaining bytes
    # by their combined cost before allocating either pool. Equal block counts
    # give equal token capacity and a byte split proportional to KV size per token.
    blocks = planner.resolve(
        profile_peak=peak,
        temporary_pool_bytes=temporary,
        workspace_reserve=max(p[2] for p in profiles) + extra,
        block_bytes=target_block_bytes + draft_block_bytes,
        minimum_blocks=(config.max_model_len + memory.block_size - 1) // memory.block_size,
        explicit_blocks=memory.num_blocks,
    )
    resolved = replace(memory, num_blocks=blocks)
    target = Qwen2Runner(
        model,
        eos_token_ids,
        memory_config=resolved,
        engine_config=config,
        compute_config=backends[0],
        graph_config=graphs,
    )
    cfg = draft.config
    peer = target.kv_cache.create_peer(
        layers=cfg.num_hidden_layers, kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim
    )
    draft_runner = Qwen2Runner(
        draft,
        eos_token_ids,
        memory_config=resolved,
        engine_config=config,
        compute_config=backends[1],
        graph_config=graphs,
        kv_cache=peer,
    )
    stats.target_pool_bytes = target.kv_cache.storage.nbytes
    stats.draft_pool_bytes = peer.storage.nbytes
    decoder = SpeculativeDecoder(target, draft_runner, speculative.num_draft_tokens)
    return runtime_cls(Engine(target, config, decoder=decoder), planner)


def from_directory(runtime_cls, directory, config, speculative, *, device, dtype, **options):
    from ajvllm.execution.qwen2 import read_eos_token_ids

    check_tokenizers(directory, speculative.draft_model)
    model = load_qwen2(directory, device=device, dtype=dtype)
    draft = load_qwen2(speculative.draft_model, device=device, dtype=dtype)
    return create_runtime(
        runtime_cls,
        model,
        draft,
        config,
        speculative,
        eos_token_ids=read_eos_token_ids(directory, model.config.vocab_size),
        **options,
    )
