"""Bounded CUDA graph replay and weight-only quantization correctness."""

import pytest
import torch
from test_qwen2_cuda import pair, tiny_config
from torch import nn

from ajvllm import EngineConfig
from ajvllm.config import ComputeConfig, GraphConfig, MemoryConfig, QuantizationConfig
from ajvllm.execution.qwen2 import Qwen2Runner
from ajvllm.quantization.linear import Int8Linear, quantization_stats, quantize_model
from ajvllm.scheduling.batch import Phase, ScheduledRequest, SchedulerOutput

pytestmark = pytest.mark.cuda


def execute(runner, rows):
    return runner.execute(
        SchedulerOutput(
            tuple(
                ScheduledRequest(
                    rid, tuple(tokens), start, Phase.PREFILL if start == 0 or len(tokens) > 1 else Phase.DECODE, True
                )
                for rid, tokens, start in rows
            )
        )
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 4, 17, 65])
@pytest.mark.parametrize("bias", [False, True])
@torch.inference_mode()
def test_int8_linear_reference(dtype, rows, bias):
    torch.manual_seed(19)
    linear = nn.Linear(70, 39, bias=bias, device="cuda", dtype=dtype)
    linear.weight[0].zero_()
    quant = Int8Linear.from_linear(linear)
    x = torch.randn(rows, 70, device="cuda", dtype=dtype)
    assert quant.qweight.dtype == torch.int8 and quant.scales.dtype == torch.float32
    assert quant.scales[0] == 1 and quant.qweight[0].count_nonzero() == 0
    torch.testing.assert_close(quant(x), quant.reference(x), atol=0.005 if dtype == torch.float16 else 0.02, rtol=0.004)
    restored = quant.qweight.float() * quant.scales[:, None]
    assert ((restored - linear.weight.float()).abs() <= quant.scales[:, None] * 0.501 + 1e-6).all()


@pytest.mark.parametrize("quantized", [False, True])
@torch.inference_mode()
def test_graph_replay_padding_context_changes_and_quantization(quantized):
    model, _ = pair(tiny_config(max_position_embeddings=320), torch.bfloat16)
    if quantized:
        tied = model.model.embed_tokens.weight
        quantize_model(model, QuantizationConfig(mode="w8a16"))
        assert model.lm_head.weight is tied
        assert quantization_stats(model)["linear_layers"] == 14
    cfg = EngineConfig(max_model_len=320, max_num_seqs=4)
    options = dict(
        engine_config=cfg, memory_config=MemoryConfig(block_size=7), compute_config=ComputeConfig(backend="triton")
    )
    eager = Qwen2Runner(model, **options)
    graphed = Qwen2Runner(model, **options, graph_config=GraphConfig(enabled=True, batch_sizes=(2, 4)))
    # Only 3 real rows occupy a bucket of 4. Padding must not overwrite block 0.
    rows = [("a", [1] * 127, 0), ("b", [2] * 3, 0), ("c", [3] * 9, 0)]
    for runner in (eager, graphed):
        execute(runner, rows)
    old_output = None
    saved = None
    for step in range(3):
        rows = [("a", [4], 127 + step), ("b", [5], 3 + step), ("c", [6], 9 + step)]
        expected = execute(eager, rows)
        actual = execute(graphed, rows)
        for rid in actual:
            torch.testing.assert_close(actual[rid], expected[rid], atol=0.03, rtol=0.004)
        if old_output is not None:
            torch.testing.assert_close(old_output, saved, atol=0, rtol=0)
        old_output = actual["a"]
        saved = old_output.clone()
    assert graphed.graphs.captures == 2 and graphed.graphs.replays == 3
    # Different row order and page assignments replay the same graph safely.
    for runner in (eager, graphed):
        runner.release("b")
        execute(runner, [("new", [7] * 4, 0)])
    rows = [("new", [8], 4), ("c", [9], 12)]
    expected, actual = [execute(r, rows) for r in (eager, graphed)]
    for rid in actual:
        torch.testing.assert_close(actual[rid], expected[rid], atol=0.03, rtol=0.004)
    for runner in (eager, graphed):
        for rid in ("a", "new", "c"):
            runner.release(rid)
        assert runner.num_active_states == 0


@torch.inference_mode()
def test_graph_cache_bound_and_memory_fallback():
    model, _ = pair(tiny_config(), torch.float16)
    cfg = EngineConfig(max_model_len=128, max_num_seqs=1)
    runner = Qwen2Runner(
        model,
        engine_config=cfg,
        memory_config=MemoryConfig(),
        compute_config=ComputeConfig(backend="triton"),
        graph_config=GraphConfig(enabled=True, memory_limit_mb=1),
    )
    execute(runner, [("a", [1, 2], 0)])
    actual = execute(runner, [("a", [3], 2)])
    assert torch.isfinite(actual["a"]).all()
    assert runner.graphs.captures == 0 and runner.graphs.fallbacks == 2
    runner.release("a")


@torch.inference_mode()
def test_graph_split_decode_padding_cross_stream_and_cache_limit():
    model, _ = pair(tiny_config(max_position_embeddings=1300), torch.float16)
    cfg = EngineConfig(max_model_len=1300, max_num_seqs=2)
    options = dict(
        engine_config=cfg, memory_config=MemoryConfig(block_size=7), compute_config=ComputeConfig(backend="triton")
    )
    reference = Qwen2Runner(model, **options)
    runner = Qwen2Runner(model, **options, graph_config=GraphConfig(enabled=True, batch_sizes=(2,), max_graphs=1))
    for target in (reference, runner):
        target.kv_cache.storage.tensor.zero_()
        execute(target, [("a", [1] * 1024, 0)])
    expected = execute(reference, [("a", [2], 1024)])
    actual = execute(runner, [("a", [2], 1024)])
    torch.testing.assert_close(actual["a"], expected["a"], atol=0.005, rtol=0.004)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = execute(runner, [("a", [3], 1025)])
    stream.synchronize()
    expected = execute(reference, [("a", [3], 1025)])
    torch.testing.assert_close(actual["a"], expected["a"], atol=0.005, rtol=0.004)
    torch.testing.assert_close(
        runner.kv_cache.storage.tensor, reference.kv_cache.storage.tensor, atol=0.005, rtol=0.004
    )
    runner.release("a")
    execute(runner, [("short", [4, 5], 0)])
    execute(runner, [("short", [6], 2)])
    assert runner.graphs.captures == 1 and len(runner.graphs.entries) == 1
    assert (2, 128) in runner.graphs.rejected
    runner.release("short")


@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty:UserWarning")
@torch.inference_mode()
def test_graph_capture_failure_releases_engine_pages_and_recovers():
    from ajvllm import Engine, EngineExecutionError, SamplingParams

    model, _ = pair(tiny_config(), torch.float16)
    cfg = EngineConfig(max_model_len=128, max_num_seqs=1, max_num_batched_tokens=4)
    runner = Qwen2Runner(
        model,
        engine_config=cfg,
        memory_config=MemoryConfig(),
        compute_config=ComputeConfig(backend="triton"),
        graph_config=GraphConfig(enabled=True),
    )
    engine = Engine(runner, cfg)
    engine.add_request("fail", [1, 2], SamplingParams(max_tokens=3))
    engine.step()

    def fail(_, args):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("injected capture failure")

    hook = model.register_forward_pre_hook(fail)
    try:
        with pytest.raises(EngineExecutionError):
            engine.step()
    finally:
        hook.remove()
    assert runner.num_active_states == 0 and not runner.graphs.entries
    engine.add_request("next", [3, 4], SamplingParams(max_tokens=3))
    assert list(engine.run())[-1].finished
    assert runner.graphs.replays > 0 and runner.num_active_states == 0


@torch.inference_mode()
def test_runtime_initialization_quantizes_before_budget_and_captures():
    from ajvllm import SamplingParams
    from ajvllm.runtime.inference import InferenceRuntime

    model, _ = pair(tiny_config(), torch.bfloat16)
    before = quantization_stats(model)["model_storage_bytes"]
    runtime = InferenceRuntime.from_model(
        model,
        EngineConfig(max_model_len=64, max_num_seqs=2, max_num_batched_tokens=8, max_prefill_chunk_size=4),
        graph_config=GraphConfig(enabled=True, memory_limit_mb=64, batch_sizes=(1, 2)),
        quantization_config=QuantizationConfig(mode="w8a16"),
        memory_config=MemoryConfig(num_blocks=16),
    )
    runner = runtime.engine.runner
    assert runner.quantization["model_storage_bytes"] < before
    assert runtime.budget.stats.graph_reserve_bytes == 64 * 1024**2
    assert runner.graphs.captures == 0 and runner.num_active_states == 0
    for i in range(2):
        runtime.engine.add_request(str(i), [i + 1] * 9, SamplingParams(max_tokens=4, temperature=0.8, seed=42))
    result = [out for out in runtime.run() if out.finished]
    assert len(result) == 2 and runner.num_active_states == 0
    assert runner.graphs.replays > 1


@torch.inference_mode()
def test_graph_replay_with_prefixes_and_preemption_preserves_rng():
    from ajvllm import Engine, SamplingParams

    model, _ = pair(tiny_config(), torch.bfloat16)
    quantize_model(model, QuantizationConfig(mode="w8a16"))
    cfg = EngineConfig(max_model_len=24, max_num_seqs=3, max_num_batched_tokens=7, max_prefill_chunk_size=3)
    results = []
    for enabled in (False, True):
        runner = Qwen2Runner(
            model,
            engine_config=cfg,
            memory_config=MemoryConfig(block_size=4, num_blocks=6),
            compute_config=ComputeConfig(backend="triton"),
            graph_config=GraphConfig(enabled=enabled, batch_sizes=(1, 2, 3)),
        )
        engine = Engine(runner, cfg)
        for i, length in enumerate((3, 3, 3)):
            engine.add_request(str(i), [i + 1] * length, SamplingParams(max_tokens=12, temperature=0.8, seed=42))
        finished = {}
        for _ in range(150):
            for out in engine.step():
                if out.finished:
                    finished[out.request_id] = out.output_token_ids
            if not engine.has_unfinished_requests:
                break
        assert len(finished) == 3 and runner.kv_cache.preemptions > 0
        assert runner.num_active_states == 0
        if enabled:
            assert runner.graphs.replays > 0
        results.append(finished)
    assert results[0] == results[1]
