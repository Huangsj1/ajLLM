"""Bounded production-forward startup probes with a disposable KV pool."""

from dataclasses import replace

import torch

from ajvllm.config import GraphConfig
from ajvllm.execution.capacity import Qwen2MemoryEstimate
from ajvllm.execution.qwen2 import Qwen2Runner
from ajvllm.requests import Request
from ajvllm.sampling.params import SamplingParams
from ajvllm.sampling.sampler import Sampler


def profile_memory(model, config, memory_config, compute_config, planner):
    cfg = model.config
    block_bytes = (
        memory_config.block_size
        * 2
        * cfg.num_hidden_layers
        * cfg.num_key_value_heads
        * cfg.head_dim
        * model.dtype.itemsize
    )
    slots = min(config.max_num_seqs, config.max_num_batched_tokens)
    chunk = min(config.max_model_len, config.max_prefill_chunk_size or config.max_num_batched_tokens)
    total = min(
        config.max_num_batched_tokens,
        config.max_prefill_tokens_per_step or config.max_num_batched_tokens,
        slots * chunk,
    )
    rows = min(slots, total)
    # 2 types of prefill shapes: balanced (evenly distributed) and concentrated (maximally packed)
    balanced = [total // rows + (i < total % rows) for i in range(rows)]
    concentrated, left = [], total
    while left:
        length = min(chunk, left)
        concentrated.append(length)
        left -= length
    # Also exercise the largest sampling/decode row count independently of the prefill cap.
    shapes = [balanced, concentrated, [1] * slots]
    probe_context = min(config.max_model_len, max(max(s) for s in shapes) + 1)
    # Compute the number of blocks needed to cover the largest shape, and ensure at least one block for the probe context.
    blocks = max(sum((n + 1 + memory_config.block_size - 1) // memory_config.block_size for n in s) for s in shapes)
    blocks = max(blocks, (probe_context + memory_config.block_size - 1) // memory_config.block_size)
    scratch = replace(memory_config, num_blocks=blocks, enable_prefix_cache=False)
    # Bound unobserved long-context attention, sorting/history and allocator workspace.
    # Exclude persistent KV: its final capacity is the result of profiling, not an input.
    estimate = Qwen2MemoryEstimate(cfg, model.dtype.itemsize, config, memory_config, compute_config.backend)
    reserve = estimate.workspace(slots, config.max_num_batched_tokens)
    planner.check_probe(blocks * block_bytes if memory_config.backend == "paged" else 0, reserve)
    runner = Qwen2Runner(
        model,
        memory_config=scratch,
        compute_config=compute_config,
        graph_config=GraphConfig(),
        engine_config=replace(config, max_model_len=probe_context),
    )
    pool_bytes = runner.kv_cache.storage.nbytes if runner.kv_cache else 0
    peak = 0
    try:
        for shape in shapes:
            sequences = [[0] * n for n in shape]
            sampler = Sampler(max_histories=slots)
            requests = [
                Request(
                    f"warmup-{i}",
                    tuple(tokens),
                    SamplingParams(
                        max_tokens=2, min_tokens=1, stop_token_ids=(0,), repetition_penalty=1.1, frequency_penalty=0.1
                    ),
                    0.0,
                )
                for i, tokens in enumerate(sequences)
            ]
            torch.cuda.reset_peak_memory_stats(model.device)
            runner.probe(sequences, sample=lambda logits: sampler.sample(logits, requests))
            torch.cuda.synchronize(model.device)
            peak = max(peak, torch.cuda.max_memory_allocated(model.device))
            sampler.histories.clear()
    finally:
        torch.cuda.synchronize(model.device)
        del runner
        torch.cuda.empty_cache()
    return peak, pool_bytes, reserve, block_bytes
