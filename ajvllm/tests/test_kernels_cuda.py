"""Bounded CUDA oracles for paged prefill/decode and inference fusions."""

import pytest
import torch
from test_qwen2_cuda import pair, tiny_config

from ajvllm.config import EngineConfig, MemoryConfig
from ajvllm.config.compute import ComputeConfig
from ajvllm.execution.qwen2 import Qwen2Runner
from ajvllm.kernels.elementwise import rms_norm, swiglu
from ajvllm.scheduling.batch import Phase, ScheduledRequest, SchedulerOutput

pytestmark = pytest.mark.cuda


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_elementwise(dtype):
    torch.manual_seed(11)
    x = torch.randn(7, 70, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    w = torch.randn(70, device="cuda", dtype=dtype)
    eps = 1e-6

    def reference(value):
        return (value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + eps)).to(dtype) * w

    torch.testing.assert_close(
        rms_norm(x, w, eps), reference(x), atol=2e-6 if dtype == torch.float32 else 0.02, rtol=1e-4
    )
    y, summed = rms_norm(x, w, eps, residual)
    torch.testing.assert_close(summed, x + residual, atol=0, rtol=0)
    torch.testing.assert_close(y, reference(summed), atol=2e-6 if dtype == torch.float32 else 0.02, rtol=1e-4)
    torch.testing.assert_close(
        swiglu(x, residual),
        torch.nn.functional.silu(x) * residual,
        atol=2e-6 if dtype == torch.float32 else 0.02,
        rtol=1e-4,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_model_mixed_paged_chunks(dtype):
    model, _ = pair(tiny_config(max_position_embeddings=160), dtype)
    cfg = EngineConfig(max_model_len=160, max_num_seqs=3)
    runners = [
        Qwen2Runner(
            model,
            memory_config=MemoryConfig(block_size=7),
            engine_config=cfg,
            compute_config=ComputeConfig(backend=backend),
        )
        for backend in ("eager", "triton")
    ]
    plans = [
        [("a", (1,) * 35, 0), ("b", (2,) * 9, 0)],
        [("a", (3,), 35), ("b", (4,) * 37, 9), ("c", (5,) * 3, 0)],
        [("a", (6,), 36), ("b", (7,), 46), ("c", (8,), 3)],
    ]
    for plan in plans:
        batch = SchedulerOutput(
            tuple(
                ScheduledRequest(
                    rid, tokens, start, Phase.DECODE if start and len(tokens) == 1 else Phase.PREFILL, True
                )
                for rid, tokens, start in plan
            )
        )
        expected, actual = [runner.execute(batch) for runner in runners]
        for rid in expected:
            torch.testing.assert_close(
                actual[rid], expected[rid], atol=2e-5 if dtype == torch.float32 else 0.025, rtol=2e-4
            )
    for runner in runners:
        for rid in ("a", "b", "c"):
            runner.release(rid)
        assert runner.cache_bytes == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim,kv_heads,groups,scale", [(48, 2, 7, 1), (64, 1, 1, 12)])
@torch.inference_mode()
def test_fragmented_pages_ragged_split_decode(dtype, head_dim, kv_heads, groups, scale):
    from ajvllm.attention.backends.triton import AttentionMetadata
    from ajvllm.kernels.attention import paged_attention
    from ajvllm.memory.storage import PagedBatch, PagedKVStorage

    torch.manual_seed(5)
    lengths, contexts = (33, 1, 1, 2), (1103, 1097, 31, 17)
    block_size = 7
    counts = [(n + block_size - 1) // block_size for n in contexts]
    storage = PagedKVStorage(1, sum(counts), block_size, kv_heads, head_dim, device="cuda", dtype=dtype)
    storage.tensor.normal_()
    pages = torch.randperm(sum(counts), device="cuda")
    tables = torch.zeros((4, max(counts)), device="cuda", dtype=torch.long)
    offset = 0
    for row, count in enumerate(counts):
        tables[row, :count] = pages[offset : offset + count]
        offset += count
    batch = PagedBatch(storage, tables, torch.empty(0, device="cuda", dtype=torch.long), None, None)
    q = torch.randn(sum(lengths), kv_heads * groups, head_dim, device="cuda", dtype=dtype) * scale
    metadata = AttentionMetadata.build(lengths, contexts, "cuda")
    actual = paged_attention(q, batch, 0, metadata)
    k, v = storage.layer(0)
    begin = 0
    for row, (length, context) in enumerate(zip(lengths, contexts, strict=True)):
        positions = torch.arange(context, device="cuda")
        slots = tables[row, positions // block_size] * block_size + positions % block_size
        keys = k[slots].repeat_interleave(groups, dim=1).transpose(0, 1).float()
        values = v[slots].repeat_interleave(groups, dim=1).transpose(0, 1).float()
        query = q[begin : begin + length].transpose(0, 1).float()
        scores = query @ keys.transpose(-1, -2) * head_dim**-0.5
        causal = positions[None, :] > (context - length + torch.arange(length, device="cuda"))[:, None]
        scores.masked_fill_(causal, float("-inf"))
        expected = (scores.softmax(-1) @ values).transpose(0, 1).to(dtype)
        torch.testing.assert_close(
            actual[begin : begin + length], expected, atol=3e-5 if dtype == torch.float32 else 0.025, rtol=2e-3
        )
        begin += length
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("prefix", [False, True])
@torch.inference_mode()
def test_triton_pressure_replay_and_prefix_hits(prefix):
    from ajvllm import Engine, SamplingParams

    model, _ = pair(tiny_config())
    cfg = EngineConfig(max_model_len=24, max_num_seqs=3, max_num_batched_tokens=7, max_prefill_chunk_size=3)
    results = []
    for backend in ("eager", "triton"):
        runner = Qwen2Runner(
            model,
            engine_config=cfg,
            memory_config=MemoryConfig(block_size=4, num_blocks=6, enable_prefix_cache=prefix),
            compute_config=ComputeConfig(backend=backend),
        )
        engine = Engine(runner, cfg)
        for i, n in enumerate((13, 11, 15)):
            engine.add_request(str(i), [i + 1] * n, SamplingParams(max_tokens=4, temperature=0.8, seed=42))
        finished = {}
        for _ in range(150):
            for output in engine.step():
                if output.finished:
                    finished[output.request_id] = output.output_token_ids
            if not engine.has_unfinished_requests:
                break
        assert len(finished) == 3 and runner.kv_cache.preemptions > 0
        assert runner.num_active_states == 0
        results.append(finished)
        engine.add_request("cold", [9] * 13, SamplingParams(max_tokens=1, temperature=0))
        first = list(engine.run())[-1]
        before = runner.kv_cache.hit_tokens
        engine.add_request("hot", [9] * 13, SamplingParams(max_tokens=1, temperature=0))
        assert list(engine.run())[-1].output_token_ids == first.output_token_ids
        assert runner.kv_cache.hit_tokens - before == (12 if prefix else 0)
    assert results[0] == results[1]


@pytest.mark.parametrize("head_dim", [8, 128, 256])
@torch.inference_mode()
def test_supported_head_sizes_and_no_dense_workspace(head_dim):
    from unittest.mock import patch

    from ajvllm.memory.storage import PagedBatch

    model, _ = pair(tiny_config(hidden_size=head_dim * 2, num_attention_heads=2, num_key_value_heads=1))
    config = EngineConfig(max_model_len=128, max_num_seqs=2)
    reference = Qwen2Runner(model, engine_config=config, memory_config=MemoryConfig())
    runner = Qwen2Runner(
        model, engine_config=config, memory_config=MemoryConfig(), compute_config=ComputeConfig(backend="triton")
    )
    batch = SchedulerOutput(
        (ScheduledRequest("a", (1,) * 17, 0, Phase.PREFILL, True), ScheduledRequest("b", (2,), 0, Phase.PREFILL, True))
    )
    expected = reference.execute(batch)
    seen = []

    def check(_, args):
        inputs = args[0]
        assert inputs.causal_mask is None
        assert inputs.paged.read_slots is None and inputs.paged.valid is None
        seen.append(inputs)

    hook = model.register_forward_pre_hook(check)
    try:
        with patch.object(PagedBatch, "update", side_effect=AssertionError("dense gather not allowed")):
            actual = runner.execute(batch)
    finally:
        hook.remove()
    assert len(seen) == 1
    for rid in expected:
        torch.testing.assert_close(actual[rid], expected[rid], atol=3e-5, rtol=2e-4)


@torch.inference_mode()
def test_runtime_auto_selection_and_workspace_estimate():
    from ajvllm.execution.capacity import Qwen2MemoryEstimate
    from ajvllm.runtime.inference import InferenceRuntime

    model, _ = pair(tiny_config(), torch.bfloat16)
    cfg = EngineConfig(max_model_len=128, max_num_seqs=2, max_num_batched_tokens=8, max_prefill_chunk_size=4)
    runtime = InferenceRuntime.from_model(model, cfg, memory_config=MemoryConfig(block_size=4))
    assert runtime.engine.runner.compute_backend == "triton"
    assert runtime.engine.runner.num_active_states == 0
    assert runtime.budget.stats.profile_peak_bytes > 0
    assert runtime.budget._estimate.compute_backend == "triton"
    eager = Qwen2MemoryEstimate(model.config, 2, cfg, MemoryConfig(block_size=4), "eager")
    assert runtime.budget._estimate(2, 8) < eager(2, 8)
    contiguous = Qwen2Runner(model, compute_config=ComputeConfig(), engine_config=cfg)
    assert contiguous.compute_backend == "eager"
    with pytest.raises(ValueError, match="paged KV"):
        Qwen2Runner(model, compute_config=ComputeConfig(backend="triton"), engine_config=cfg)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_packed_projections_preserve_logits_and_storage(dtype):
    from copy import deepcopy

    from ajvllm.execution.batch import ModelBatch
    from ajvllm.modeling.qwen2.projections import pack_projections

    model, _ = pair(tiny_config(), dtype)
    packed = deepcopy(model)
    original_bytes = sum(p.numel() * p.element_size() for p in packed.parameters())
    pack_projections(packed)
    pack_projections(packed)  # Preparing a shared runner twice must be harmless.
    assert sum(p.numel() * p.element_size() for p in packed.parameters()) == original_bytes
    assert not hasattr(packed.model.layers[0].self_attn, "q_proj")
    assert not hasattr(packed.model.layers[0].mlp, "gate_proj")
    batch = ModelBatch.build([[1, 2, 3], [4, 5]], [None, None], model.device, [0, 1])
    tolerance = {torch.float32: 2e-6, torch.float16: 0.002, torch.bfloat16: 0.025}[dtype]
    torch.testing.assert_close(packed(batch).logits, model(batch).logits, atol=tolerance, rtol=1e-4)
    # Split gate/up views are non-contiguous for more than one row.
    x = torch.randn(17, model.config.hidden_size, device=model.device, dtype=dtype)
    expected = model.model.layers[0].mlp(x)
    actual = packed.model.layers[0].mlp(x, optimized=True)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=1e-3)
