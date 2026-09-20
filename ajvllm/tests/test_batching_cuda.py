"""Packed computation, scheduler boundaries, and memory policy on a real CUDA model."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from model_inputs import forward_tokens
from test_qwen2_cuda import pair, tiny_config

from ajvllm import Engine, EngineConfig, EngineExecutionError, SamplingParams
from ajvllm.execution.batch import ModelBatch
from ajvllm.execution.qwen2 import Qwen2Runner
from ajvllm.modeling.qwen2.layers import rotary_factors
from ajvllm.runtime.budget import MemoryBudget
from ajvllm.scheduling.batch import Phase, ScheduledRequest, SchedulerOutput

pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module")
def models():
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    return pair(tiny_config())


@torch.inference_mode()
def test_ragged_mixed_batch_is_one_forward(models):
    native, oracle = models
    sequences = [[1, 2, 3, 4, 5, 6, 7, 8], [10, 11, 12, 13], [20, 21, 22, 23, 24]]
    starts = [0, 3, 2]
    caches = [
        None if start == 0 else forward_tokens(native, torch.tensor(seq[:start], device="cuda")).caches[0]
        for seq, start in zip(sequences, starts, strict=True)
    ]
    runner = Qwen2Runner(native)
    runner._caches.update({str(row): cache for row, cache in enumerate(caches) if cache is not None})
    batch = SchedulerOutput(
        tuple(
            ScheduledRequest(str(row), tuple(seq[start:]), start, Phase.DECODE if row == 1 else Phase.PREFILL, True)
            for row, (seq, start) in enumerate(zip(sequences, starts, strict=True))
        )
    )
    calls = []
    projections = []
    hook = native.register_forward_pre_hook(lambda _, args: calls.append(args[0]))
    proj_hook = native.model.layers[0].self_attn.q_proj.register_forward_pre_hook(
        lambda _, args: projections.append(args[0].shape[0])
    )
    try:
        logits = runner.execute(batch)
    finally:
        hook.remove()
        proj_hook.remove()
    assert len(calls) == 1 and isinstance(calls[0], ModelBatch)
    assert calls[0].num_requests == 3 and projections == [12]
    for row, seq in enumerate(sequences):
        expected = oracle(torch.tensor([seq], device="cuda"), use_cache=False).logits[0, -1].float()
        torch.testing.assert_close(logits[str(row)], expected, atol=2e-6, rtol=1e-4)
        assert runner.cached_tokens(str(row)) == len(seq)
    runner.release("0")
    # Releasing another request must not retain its cache through a batch-wide tensor view.
    for cache in runner._caches.values():
        for layer in cache:
            for tensor in layer:
                assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    for rid in ("1", "2"):
        runner.release(rid)


@torch.inference_mode()
def test_rope_is_cached_and_rebuilt_only_on_conversion(models):
    model, _ = models
    pointers = (model.rope_cos.data_ptr(), model.rope_sin.data_ptr())
    with patch("ajvllm.modeling.qwen2.model.rotary_factors", side_effect=AssertionError("RoPE recomputed in forward")):
        out = forward_tokens(model, torch.tensor([1, 2, 3], device="cuda"))
        forward_tokens(model, torch.tensor([4], device="cuda"), out.caches[0])
    assert pointers == (model.rope_cos.data_ptr(), model.rope_sin.data_ptr())
    assert not any("rope_" in key for key in model.state_dict())
    model.to(dtype=torch.bfloat16)
    model.to(dtype=torch.float32)
    cos, sin = rotary_factors(
        torch.arange(model.config.max_position_embeddings, device="cuda"), model.config, model.dtype
    )
    torch.testing.assert_close(model.rope_cos, cos, atol=0, rtol=0)
    torch.testing.assert_close(model.rope_sin, sin, atol=0, rtol=0)
    # Restore the original FP32 weights after this conversion-specific test.
    model.load_state_dict(models[1].state_dict())


@pytest.mark.parametrize(
    "budget,chunk,slots",
    [(1, None, 1), (1, 1, 8), (2, 9, 3), (3, 1, 8), (4, 2, 3), (8, None, 1), (8, 3, 8), (32, 1, 2)],
)
@torch.inference_mode()
def test_scheduler_boundaries_with_real_batched_execution(models, budget, chunk, slots):
    model, oracle = models
    prompts = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [5, 4], [1, 3, 5, 7, 9]]
    expected = {}
    for index, prompt in enumerate(prompts):
        tokens = torch.tensor([prompt], device="cuda")
        outputs = []
        for _ in range(4):
            token = oracle(tokens, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
            tokens = torch.cat((tokens, token), dim=1)
            outputs.append(token.item())
        expected[str(index)] = tuple(outputs)
    runner = Qwen2Runner(model)
    engine = Engine(
        runner,
        EngineConfig(
            max_model_len=128, max_num_batched_tokens=budget, max_prefill_chunk_size=chunk, max_num_seqs=slots
        ),
    )
    for index, prompt in enumerate(prompts):
        engine.add_request(str(index), prompt, SamplingParams(temperature=0, max_tokens=4))
    computed = {str(i): 0 for i in range(3)}
    finals = {}
    for _ in range(100):
        outputs = engine.step()
        batch = engine.last_batch
        assert batch.num_scheduled_tokens <= budget and len(batch.requests) <= slots
        for item in batch.requests:
            assert item.start_pos == computed[item.request_id]
            computed[item.request_id] += item.num_tokens
            if item.phase == Phase.PREFILL:
                assert item.num_tokens <= (chunk or budget)
                if computed[item.request_id] < len(prompts[int(item.request_id)]):
                    assert not item.do_sample
                    assert not any(out.request_id == item.request_id for out in outputs)
            else:
                assert item.num_tokens == 1
        finals.update({out.request_id: out.output_token_ids for out in outputs if out.finished})
        if not engine.has_unfinished_requests:
            break
    assert not engine.has_unfinished_requests and finals == expected
    assert runner.cache_bytes == 0


def test_dynamic_budget_rotation_unchunked_and_terminal_boundaries(models):
    model, _ = models
    runner = Qwen2Runner(model)
    engine = Engine(runner, EngineConfig(max_num_batched_tokens=8, max_model_len=128))
    for rid in "abcd":
        engine.add_request(rid, [1], SamplingParams(temperature=0, max_tokens=3))
    assert len(engine.step()) == 4
    engine.set_token_budget(1)
    order = []
    for _ in range(4):
        engine.step()
        order.append(engine.last_batch.requests[0].request_id)
    assert len(set(order)) == 4
    list(engine.run())
    assert runner.cache_bytes == 0
    engine = Engine(runner, EngineConfig(max_model_len=16, max_num_batched_tokens=16, enable_chunked_prefill=False))
    engine.add_request("full", [1] * 13, SamplingParams(temperature=0, max_tokens=9))
    engine.step()
    assert engine.last_batch.requests[0].num_tokens == 13
    assert len(list(engine.run())[-1].output_token_ids) == 3
    for rid, prompt, count in [("zero", [1], 0), ("context", [1] * 16, 1)]:
        engine.add_request(rid, prompt, SamplingParams(max_tokens=count))
    assert all(not output.new_token_ids and output.finished for output in engine.run())
    assert runner.cache_bytes == 0


def test_real_sampling_eos_minimum_and_bad_admission(models):
    model, _ = models
    first = forward_tokens(model, torch.tensor([1, 2], device="cuda")).logits[-1].argmax().item()
    runner = Qwen2Runner(model, (first,))
    config = EngineConfig(max_model_len=128)
    engine = Engine(runner, config)
    engine.add_request("eos", [1, 2], SamplingParams(temperature=0, max_tokens=3))
    assert list(engine.run())[-1].stop_token_id == first
    engine.add_request("min", [1, 2], SamplingParams(temperature=0, max_tokens=2, min_tokens=2))
    result = list(engine.run())[-1]
    assert len(result.output_token_ids) == 2 and first not in result.output_token_ids
    for prompt in ([], [-1], [97], [1] * 129):
        with pytest.raises(ValueError):
            engine.add_request("bad", prompt)
    assert runner.cache_bytes == 0


def test_failed_batch_releases_all_members_and_waiting_request_survives(models):
    model, _ = models
    runner = Qwen2Runner(model)
    engine = Engine(runner, EngineConfig(max_model_len=128, max_num_seqs=2, max_num_batched_tokens=4))
    for rid in "abc":
        engine.add_request(rid, [1, 2, 3], SamplingParams(max_tokens=1, temperature=0))
    hook = model.register_forward_hook(lambda *_: (_ for _ in ()).throw(RuntimeError("injected execution failure")))
    try:
        with pytest.raises(EngineExecutionError) as failure:
            engine.step()
        assert {out.request_id for out in failure.value.outputs} == {"a", "b"}
    finally:
        hook.remove()
    assert runner.cache_bytes == 0
    assert list(engine.run())[-1].request_id == "c"


def test_memory_warmup_growth_reduction_and_impossible_target(models):
    model, _ = models
    runner = Qwen2Runner(model)
    config = EngineConfig(max_model_len=128, max_num_seqs=2, max_num_batched_tokens=8)
    budget = MemoryBudget(runner, config, initial_token_budget=1, growth_interval=1)
    engine = Engine(runner, budget.config)
    engine.add_request("a", [1] * 40, SamplingParams(max_tokens=1, temperature=0))
    used = []
    while engine.has_unfinished_requests:
        budget.before_step(engine)
        used.append(engine.token_budget)
        engine.step()
        budget.after_step(engine)
    assert used[:4] == [1, 2, 4, 8] and max(used) <= 8
    assert budget.stats.profile_peak_bytes > 0 and budget.stats.observed_peak_bytes <= budget.stats.target_bytes
    budget.on_oom()
    assert budget.stats.token_budget == 4
    with pytest.raises(MemoryError):
        MemoryBudget(runner, config, gpu_memory_utilization=1e-9, safety_bytes=0)
    with pytest.raises(ValueError):
        MemoryBudget(runner, replace(config, enable_chunked_prefill=False, max_num_batched_tokens=128))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_qwen_seven_query_heads_per_kv_head_without_replication(dtype):
    config = tiny_config(hidden_size=112, num_attention_heads=14, num_key_value_heads=2)
    native, oracle = pair(config, dtype)
    sequences = [[1, 2, 3, 4, 5], [6, 7, 8]]
    with patch.object(torch.Tensor, "repeat_interleave", side_effect=AssertionError("KV replication")):
        output = native(ModelBatch.build(sequences, [None, None], native.device, [0, 1]))
    for row, tokens in enumerate(sequences):
        expected = oracle(torch.tensor([tokens], device="cuda"), use_cache=False).logits[0, -1].float()
        tolerance = 2e-6 if dtype == torch.float32 else 0.02
        torch.testing.assert_close(output.logits[row], expected, atol=tolerance, rtol=1e-4)
