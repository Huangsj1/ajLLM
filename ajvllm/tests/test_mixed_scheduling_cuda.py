"""Mixed execution, shared budget, fair chunking, and failure semantics on CUDA."""

import pytest
import torch
from test_qwen2_cuda import pair, tiny_config

from ajvllm import Engine, EngineConfig, EngineExecutionError, SamplingParams
from ajvllm.execution.qwen2 import Qwen2Runner
from ajvllm.scheduling.batch import Phase, SchedulerOutput

pytestmark = pytest.mark.cuda


@pytest.fixture
def runner():
    assert torch.cuda.is_available()
    model, _ = pair(tiny_config())
    return Qwen2Runner(model)


def test_shared_budget_decode_reservation_and_fair_prefill(runner):
    engine = Engine(
        runner,
        EngineConfig(
            max_model_len=128,
            max_num_batched_tokens=12,
            max_prefill_chunk_size=8,
            max_prefill_tokens_per_step=8,
            max_num_seqs=4,
        ),
    )
    params = SamplingParams(temperature=0, max_tokens=5)
    engine.add_request("decode", [1], params)
    engine.step()
    for rid, length in [("long-a", 40), ("short", 1), ("long-b", 40)]:
        engine.add_request(rid, [2] * length, params)
    engine.step()
    plan = engine.last_batch
    prefill = [item for item in plan.requests if item.phase == Phase.PREFILL]
    decode = [item for item in plan.requests if item.phase == Phase.DECODE]
    assert sum(item.num_tokens for item in decode) == 1
    assert sum(item.num_tokens for item in prefill) == 8
    assert sorted(item.num_tokens for item in prefill) == [1, 3, 4]
    assert plan.num_scheduled_tokens == 9 <= engine.token_budget
    assert next(item for item in prefill if item.request_id == "short").phase == Phase.PREFILL
    assert not next(item for item in prefill if item.request_id == "long-a").do_sample
    list(engine.run())
    assert runner.cache_bytes == 0


def test_active_prefills_rotate_when_budget_shrinks(runner):
    engine = Engine(
        runner, EngineConfig(max_model_len=128, max_num_batched_tokens=3, max_num_seqs=3, max_prefill_chunk_size=1)
    )
    for rid in "abc":
        engine.add_request(rid, [1] * 8, SamplingParams(max_tokens=1, temperature=0))
    engine.step()
    assert len(engine.last_batch.requests) == 3
    engine.set_token_budget(1)
    selected = []
    for _ in range(6):
        engine.step()
        selected.append(engine.last_batch.requests[0].request_id)
    assert selected == list("abcabc")
    list(engine.run())
    assert runner.cache_bytes == 0


@pytest.mark.parametrize("cap", [1, 2, 7, 32])
def test_prefill_cap_and_chunk_limit_during_dynamic_budgets(runner, cap):
    engine = Engine(
        runner,
        EngineConfig(
            max_model_len=128,
            max_num_batched_tokens=8,
            max_num_seqs=3,
            max_prefill_chunk_size=3,
            max_prefill_tokens_per_step=cap,
        ),
    )
    for rid in "abc":
        engine.add_request(rid, [1] * 17, SamplingParams(max_tokens=3, temperature=0))
    index = 0
    while engine.has_unfinished_requests:
        budget = (1, 8, 3)[index % 3]
        engine.set_token_budget(budget)
        engine.step()
        plan = engine.last_batch
        assert plan.num_scheduled_tokens <= budget
        assert sum(item.num_tokens for item in plan.requests if item.phase == Phase.PREFILL) <= cap
        assert all(item.num_tokens <= 3 for item in plan.requests if item.phase == Phase.PREFILL)
        index += 1
        assert index < 100
    assert runner.cache_bytes == 0


def test_empty_schedule_skips_forward_and_single_token_batches(runner):
    calls = []
    hook = runner.model.register_forward_pre_hook(lambda _, args: calls.append(args[0].query_lengths))
    try:
        assert runner.execute(SchedulerOutput()) == {}
        engine = Engine(runner, EngineConfig(max_model_len=128))
        for rid in "abc":
            engine.add_request(rid, [1], SamplingParams(max_tokens=2, temperature=0))
        list(engine.run())
    finally:
        hook.remove()
    assert calls == [(1, 1, 1), (1, 1, 1)]


def test_mixed_forward_failure_cleans_batch_without_emitting_partial_results(runner):
    engine = Engine(runner, EngineConfig(max_model_len=128, max_num_seqs=2, max_num_batched_tokens=8))
    engine.add_request("decode", [1], SamplingParams(max_tokens=5, temperature=0))
    engine.step()
    engine.add_request("prefill", [2] * 3, SamplingParams(max_tokens=2, temperature=0))
    engine.add_request("waiting", [3], SamplingParams(max_tokens=1, temperature=0))
    completed_batches = []

    def hook(_, args, output):
        completed_batches.append(args[0].query_lengths)
        raise RuntimeError("mixed forward failed")

    handle = runner.model.register_forward_hook(hook)
    try:
        with pytest.raises(EngineExecutionError) as error:
            engine.step()
        assert completed_batches == [(1, 3)]
        assert {out.request_id for out in error.value.outputs} == {"prefill", "decode"}
        assert all(not out.new_token_ids for out in error.value.outputs)
        assert runner.cache_bytes == 0
    finally:
        handle.remove()
    assert list(engine.run())[-1].request_id == "waiting"


def test_unchunked_cap_must_admit_a_full_prompt(runner):
    with pytest.raises(ValueError, match="prefill budget"):
        EngineConfig(max_model_len=128, enable_chunked_prefill=False, max_prefill_tokens_per_step=1)
    engine = Engine(
        runner,
        EngineConfig(
            max_model_len=128, enable_chunked_prefill=False, max_prefill_tokens_per_step=128, max_num_batched_tokens=128
        ),
    )
    engine.add_request("a", [1] * 100, SamplingParams(max_tokens=1, temperature=0))
    engine.add_request("b", [2] * 100, SamplingParams(max_tokens=1, temperature=0))
    engine.step()
    assert len(engine.last_batch.requests) == 1 and engine.last_batch.num_scheduled_tokens == 100
    engine.step()
    assert engine.last_batch.requests[0].request_id == "b"
