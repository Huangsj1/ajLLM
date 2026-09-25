"""Paged KV correctness, prefix ownership, pressure/replay and CUDA lifetime boundaries."""

import pytest
import torch
from test_qwen2_cuda import pair, tiny_config

from ajvllm import Engine, EngineConfig, EngineExecutionError, SamplingParams
from ajvllm.config.memory import MemoryConfig
from ajvllm.execution.qwen2 import Qwen2Runner
from ajvllm.runtime.inference import InferenceRuntime
from ajvllm.scheduling.batch import Phase, ScheduledRequest, SchedulerOutput

pytestmark = pytest.mark.cuda


@pytest.fixture
def model():
    assert torch.cuda.is_available()
    return pair(tiny_config())[0]


def runner_for(model, blocks=24, prefix=True, context=32):
    config = EngineConfig(max_model_len=context, max_num_seqs=3, max_num_batched_tokens=7, max_prefill_chunk_size=3)
    runner = Qwen2Runner(
        model,
        memory_config=MemoryConfig(block_size=4, num_blocks=blocks, enable_prefix_cache=prefix),
        engine_config=config,
    )
    return runner, config


def execute(runner, rid, tokens, start=0):
    return runner.execute(
        SchedulerOutput(
            (
                ScheduledRequest(
                    rid, tuple(tokens), start, Phase.PREFILL if len(tokens) > 1 or start == 0 else Phase.DECODE, True
                ),
            )
        )
    )[rid]


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("temperature", [0, 0.8])
@pytest.mark.parametrize("penalties", [False, True])
def test_pressure_recompute_preserves_tokens_and_rng(model, prefix, temperature, penalties):
    configs = EngineConfig(max_model_len=24, max_num_seqs=3, max_num_batched_tokens=7, max_prefill_chunk_size=3)
    outputs = []
    for paged in (False, True):
        runner = Qwen2Runner(
            model,
            engine_config=configs,
            memory_config=MemoryConfig(
                backend="paged" if paged else "contiguous", block_size=4, num_blocks=6, enable_prefix_cache=prefix
            ),
        )
        engine = Engine(runner, configs)
        for index, length in enumerate((3, 3, 3)):
            engine.add_request(
                str(index),
                [index + 1] * length,
                SamplingParams(
                    max_tokens=12,
                    temperature=temperature,
                    seed=42,
                    repetition_penalty=1.2 if penalties else 1.0,
                    presence_penalty=0.3 if penalties else 0.0,
                    frequency_penalty=0.1 if penalties else 0.0,
                ),
            )
        result = {}
        for step in range(150):
            for output in engine.step():
                if output.finished:
                    result[output.request_id] = output.output_token_ids
            if not engine.has_unfinished_requests:
                break
        assert not engine.has_unfinished_requests
        outputs.append(result)
        if paged:
            assert runner.kv_cache.preemptions > 0
            assert all(ref == 0 for ref in runner.kv_cache.blocks.ref_counts)
            assert runner.kv_cache.snapshot()["used_bytes"] == 0
        assert runner.num_active_states == 0
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("length", [3, 4, 5, 8, 9, 12])
@torch.inference_mode()
def test_prefix_hit_last_token_recompute_and_salt_isolation(model, length):
    runner, config = runner_for(model)
    engine = Engine(runner, config)
    tokens = tuple((index % 9) + 1 for index in range(length))
    params = SamplingParams(max_tokens=1, temperature=0)
    engine.add_request("first", tokens, params, cache_salt="a")
    expected = list(engine.run())[-1].output_token_ids
    before = engine.metrics.scheduled_prefill_tokens
    engine.add_request("reuse", tokens, params, cache_salt="a")
    actual = list(engine.run())[-1].output_token_ids
    reused = (length - 1) // 4 * 4
    assert actual == expected
    assert engine.metrics.scheduled_prefill_tokens - before == length - reused
    assert runner.kv_cache.hit_tokens == reused
    before = engine.metrics.scheduled_prefill_tokens
    engine.add_request("isolated", tokens, params, cache_salt="b")
    assert list(engine.run())[-1].output_token_ids == expected
    assert engine.metrics.scheduled_prefill_tokens - before == length
    assert all(ref == 0 for ref in runner.kv_cache.blocks.ref_counts)


@torch.inference_mode()
def test_copy_on_write_partial_tail_and_reference_logits(model):
    runner, _ = runner_for(model)
    execute(runner, "source", [1, 2])
    manager = runner.kv_cache
    manager.fork("source", "fork")
    original_page = manager.states["source"].blocks[0]
    before = manager.storage.tensor[:, :, original_page].clone()
    actual = execute(runner, "fork", [3], start=2)
    assert manager.states["fork"].blocks[0] != original_page
    assert manager.cow_copies == 1 and manager.blocks.ref_counts[original_page] == 1
    torch.testing.assert_close(manager.storage.tensor[:, :, original_page], before, equal_nan=True)
    reference = Qwen2Runner(model)
    expected = execute(reference, "full", [1, 2, 3])
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-4)
    for rid in ("source", "fork"):
        runner.release(rid)
        runner.release(rid)
    assert all(ref == 0 for ref in manager.blocks.ref_counts)
    assert manager.snapshot()["shared_blocks"] == 0


@torch.inference_mode()
def test_full_shared_blocks_eviction_and_atomic_reservation(model):
    runner, _ = runner_for(model, blocks=8)
    manager = runner.kv_cache
    execute(runner, "first", [1] * 8)
    runner.release("first")
    assert manager.attach("a", (1,) * 9) == 8
    assert manager.attach("b", (1,) * 9) == 8
    assert all(manager.blocks.ref_counts[block] == 2 for block in manager.states["a"].blocks)
    assert manager.snapshot()["shared_blocks"] == len(manager.states["a"].blocks)
    manager.release("b")
    refs, free = list(manager.blocks.ref_counts), list(manager.blocks.free)
    assert not manager.reserve("a", 40)
    assert manager.blocks.ref_counts == refs and list(manager.blocks.free) == free
    manager.release("a")
    # Fill the pool with a distinct prefix, evicting the old completed pages.
    execute(runner, "replacement", [2] * 32)
    runner.release("replacement")
    assert manager.blocks.evictions > 0
    assert manager.attach("old", (1,) * 9) == 0
    manager.release("old")
    manager.clear_prefix_cache()
    assert not manager.blocks.prefixes


@torch.inference_mode()
def test_copy_failure_rolls_back_and_fork_only_shares_committed_pages(model, monkeypatch):
    runner, _ = runner_for(model)
    execute(runner, "source", [1, 2])
    manager = runner.kv_cache
    assert manager.reserve("source", 8)
    manager.fork("source", "fork")
    assert len(manager.states["fork"].blocks) == 1
    refs = list(manager.blocks.ref_counts)
    pages = list(manager.states["fork"].blocks)

    def fail(*_):
        raise RuntimeError("injected copy failure")

    with monkeypatch.context() as patch:
        patch.setattr(manager.storage, "copy_block", fail)
        with pytest.raises(RuntimeError, match="copy failure"):
            manager.reserve("fork", 8)
    assert manager.blocks.ref_counts == refs
    assert manager.states["fork"].blocks == pages
    # A source with a reserved future page still COWs its committed partial tail.
    assert manager.reserve("source", 3)
    assert manager.states["source"].blocks[0] != pages[0]
    for rid in ("source", "fork"):
        manager.release(rid)
    assert all(ref == 0 for ref in manager.blocks.ref_counts)
    assert manager.snapshot()["shared_blocks"] == 0


def test_forward_failure_and_cancel_release_pages(model):
    runner, config = runner_for(model)
    engine = Engine(runner, config)
    engine.add_request("a", [1] * 9, SamplingParams(max_tokens=3))
    engine.step()
    assert runner.cache_bytes > 0
    engine.cancel_request("a")
    assert runner.cache_bytes == 0
    for rid in ("b", "c"):
        engine.add_request(rid, [2] * 9, SamplingParams(max_tokens=3))
    hook = model.model.layers[0].register_forward_hook(
        lambda *_: (_ for _ in ()).throw(RuntimeError("failed after KV write"))
    )
    try:
        with pytest.raises(EngineExecutionError):
            engine.step()
    finally:
        hook.remove()
    assert runner.num_active_states == 0 and runner.cache_bytes == 0
    assert all(ref == 0 for ref in runner.kv_cache.blocks.ref_counts)
    # The failed partial blocks must not be published for reuse.
    assert runner.kv_cache.attach("fresh", (2,) * 9) == 0
    runner.release("fresh")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_paged_mixed_batch_and_cross_stream_reuse(dtype):
    model, _ = pair(tiny_config(), dtype)
    runner, _ = runner_for(model, prefix=False)
    reference = Qwen2Runner(model)
    for target in (runner, reference):
        execute(target, "a", [1, 2, 3])
        execute(target, "b", [4, 5])
    plan = SchedulerOutput(
        (
            ScheduledRequest("a", (6,), 3, Phase.DECODE, True),
            ScheduledRequest("b", (7, 8, 9), 2, Phase.PREFILL, True),
            ScheduledRequest("c", (10, 11), 0, Phase.PREFILL, True),
        )
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = runner.execute(plan)
    stream.synchronize()
    expected = reference.execute(plan)
    for rid in actual:
        torch.testing.assert_close(actual[rid], expected[rid], atol=2e-6 if dtype == torch.float32 else 0.02, rtol=1e-4)
    for rid in ("a", "b", "c"):
        runner.release(rid)
    # Poison free storage: padding must be masked before matmul.
    runner.kv_cache.storage.tensor.fill_(float("nan"))
    result = runner.execute(
        SchedulerOutput(
            (
                ScheduledRequest("x", (1,), 0, Phase.PREFILL, True),
                ScheduledRequest("y", (2, 3, 4), 0, Phase.PREFILL, True),
            )
        )
    )
    assert all(torch.isfinite(row).all() for row in result.values())


def test_memory_budget_profiles_selected_paged_backend(model):
    config = EngineConfig(max_model_len=32, max_num_seqs=2, max_num_batched_tokens=8, max_prefill_chunk_size=4)
    runtime = InferenceRuntime.from_model(model, config, memory_config=MemoryConfig(block_size=4, num_blocks=8))
    runner, budget = runtime.engine.runner, runtime.budget
    assert budget.stats.profile_peak_bytes > 0
    assert runner.kv_cache is not None and runner.num_active_states == 0
    assert runner.kv_cache.snapshot()["pool_bytes"] > 0
    with pytest.raises(ValueError, match="maximum-context"):
        Qwen2Runner(model, memory_config=MemoryConfig(block_size=4, num_blocks=1), engine_config=config)


@pytest.mark.parametrize("backend", ["contiguous", "paged"])
def test_runtime_probe_and_offline_budget_share_execution_path(model, backend, monkeypatch):
    from ajvllm.runtime.budget import MemoryBudget

    calls = []
    execute_original = Qwen2Runner.execute

    def execute_recorded(self, batch):
        calls.append(batch)
        return execute_original(self, batch)

    monkeypatch.setattr(Qwen2Runner, "execute", execute_recorded)
    config = EngineConfig(max_model_len=32, max_num_seqs=2, max_num_batched_tokens=8, max_prefill_chunk_size=4)
    runtime = InferenceRuntime.from_model(
        model, config, memory_config=MemoryConfig(backend=backend, block_size=4, num_blocks=16)
    )
    runner = runtime.engine.runner
    assert len(calls) == 6
    assert {item.phase for item in calls[0].requests} == {Phase.PREFILL}
    assert {item.phase for item in calls[1].requests} == {Phase.DECODE}
    assert runner.num_active_states == 0
    if runner.kv_cache:
        assert runner.kv_cache.enable_prefix_cache and not runner.kv_cache.blocks.prefixes
        # Weight baseline excludes the pool, which is accounted for exactly once.
        assert runtime.budget.stats.pool_bytes == runner.kv_cache.storage.nbytes
    runtime.engine.add_request("live", [1] * 9, SamplingParams(max_tokens=2, temperature=0))
    assert list(runtime.run())[-1].finished
    assert runtime.engine.token_budget == 8
    assert runner.num_active_states == 0

    # The controller only plans: constructing it neither executes nor allocates KV.
    calls.clear()
    MemoryBudget(model, config)
    assert not calls and runner.num_active_states == 0


@pytest.mark.parametrize("backend", ["contiguous", "paged"])
def test_probe_failure_releases_state_and_restores_prefix_policy(model, backend):
    config = EngineConfig(max_model_len=32, max_num_seqs=2)
    runner = Qwen2Runner(model, memory_config=MemoryConfig(backend=backend, block_size=4), engine_config=config)

    def fail(*_):
        raise RuntimeError("warmup failure")

    hook = model.model.layers[0].register_forward_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="warmup failure"):
            runner.probe([[1, 2], [3]])
    finally:
        hook.remove()
    assert runner.num_active_states == 0
    if runner.kv_cache:
        assert runner.kv_cache.enable_prefix_cache
        assert not runner.kv_cache.blocks.prefixes
        assert all(ref == 0 for ref in runner.kv_cache.blocks.ref_counts)
    runner.probe([[1], [2]])
    assert runner.num_active_states == 0


def test_runtime_oom_keeps_fixed_budget_without_retrying_failed_requests(model):
    config = EngineConfig(max_model_len=32, max_num_seqs=2, max_num_batched_tokens=8, max_prefill_chunk_size=4)
    runtime = InferenceRuntime.from_model(model, config, memory_config=MemoryConfig(block_size=4, num_blocks=16))
    runtime.engine.add_request("failed", [1] * 9, SamplingParams(max_tokens=2))

    def fail(*_):
        raise torch.cuda.OutOfMemoryError("injected OOM; no oversized allocation")

    hook = model.model.layers[0].register_forward_hook(fail)
    try:
        with pytest.raises(EngineExecutionError) as error:
            runtime.step()
    finally:
        hook.remove()
    assert error.value.outputs[0].finish_reason == "error"
    assert runtime.budget.stats.token_budget == 8
    assert runtime.engine.runner.num_active_states == 0
    runtime.engine.add_request("next", [2], SamplingParams(max_tokens=1))
    assert list(runtime.run())[-1].finished
    assert runtime.engine.token_budget == 8


def test_new_prompt_waits_without_evicting_running_request(model):
    cfg = EngineConfig(max_model_len=24, max_num_seqs=3, max_num_batched_tokens=7, max_prefill_chunk_size=3)
    runner = Qwen2Runner(model, engine_config=cfg, memory_config=MemoryConfig(block_size=4, num_blocks=6))
    engine = Engine(runner, cfg)
    engine.add_request("first", [1] * 13, SamplingParams(max_tokens=4, temperature=0))
    engine.step()
    engine.add_request("second", [2] * 11, SamplingParams(max_tokens=4, temperature=0))
    while "first" in engine._scheduler.requests:
        engine.step()
        assert "second" not in runner.kv_cache.states
        assert runner.kv_cache.preemptions == 0
    assert runner.kv_cache.admission_waits > 0
    assert list(engine.run())[-1].request_id == "second"
    assert not runner.kv_cache.states


@pytest.mark.parametrize("enabled", [False, True])
def test_short_request_switch_is_bounded_and_preserves_long_request(model, enabled):
    cfg = EngineConfig(max_model_len=128, max_num_seqs=1, max_num_batched_tokens=16, short_request_preemption=enabled)
    runner = Qwen2Runner(
        model, engine_config=cfg, memory_config=MemoryConfig(block_size=4, num_blocks=32, enable_prefix_cache=False)
    )
    engine = Engine(runner, cfg)
    params = SamplingParams(max_tokens=90, temperature=0.8, seed=7)
    engine.add_request("long", [1] * 16, params)
    for _ in range(3):
        engine.step()
    engine.add_request("short", [2, 3], SamplingParams(max_tokens=2, temperature=0))
    finished = [out for out in engine.run() if out.finished]
    assert finished[0].request_id == ("short" if enabled else "long")
    assert runner.kv_cache.policy_preemptions == int(enabled)
    assert runner.kv_cache.pressure_preemptions == 0
    reference = Engine(Qwen2Runner(model), cfg)
    reference.add_request("long", [1] * 16, params)
    expected = list(reference.run())[-1].output_token_ids
    assert next(out for out in finished if out.request_id == "long").output_token_ids == expected
    assert not runner.kv_cache.states


def test_profiled_pool_not_proportional_to_sequence_context_capacity(model):
    import gc
    from dataclasses import asdict

    pools = []
    for slots in (2, 64):
        cfg = EngineConfig(max_model_len=32, max_num_seqs=slots, max_num_batched_tokens=8)
        free, total = torch.cuda.mem_get_info(model.device)
        target = torch.cuda.memory_allocated(model.device) + 32 * 1024**2
        runtime = InferenceRuntime.from_model(
            model, cfg, gpu_memory_utilization=target / total, safety_bytes=2 * 1024**2
        )
        stats = runtime.budget.stats
        assert runtime.engine.config == cfg
        assert stats.pool_bytes + stats.non_kv_peak_bytes + stats.safety_bytes <= stats.target_bytes
        assert stats.temporary_pool_bytes < stats.pool_bytes
        assert stats.num_blocks == runtime.engine.runner.kv_cache.snapshot()["num_blocks"]
        pools.append(asdict(stats))
        runtime.engine.close()
        del runtime
        gc.collect()
        torch.cuda.empty_cache()
    assert 0.7 < pools[1]["pool_bytes"] / pools[0]["pool_bytes"] < 1.3


def test_waiting_prefixes_do_not_pin_pages_ahead_of_fifo_admission(model):
    cfg = EngineConfig(max_model_len=24, max_num_seqs=3, max_num_batched_tokens=7, max_prefill_chunk_size=3)
    runner = Qwen2Runner(model, engine_config=cfg, memory_config=MemoryConfig(block_size=4, num_blocks=6))
    execute(runner, "cached", [1] * 20)
    runner.release("cached")
    engine = Engine(runner, cfg)
    for rid, token in (("first", 2), ("later", 1)):
        engine.add_request(rid, [token] * 20, SamplingParams(max_tokens=2, temperature=0))
    finished = []
    for _ in range(40):
        finished.extend(out.request_id for out in engine.step() if out.finished)
        if not engine.has_unfinished_requests:
            break
    assert finished == ["first", "later"]
    assert runner.kv_cache.preemptions == 0
    assert all(ref == 0 for ref in runner.kv_cache.blocks.ref_counts)
