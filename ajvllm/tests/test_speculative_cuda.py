"""CUDA rejection distribution, rollback, paired prefix ownership and mixed scheduling."""

import copy

import pytest
import torch
from test_qwen2_cuda import pair, tiny_config

from ajvllm import Engine, EngineConfig, SamplingParams
from ajvllm.config import ComputeConfig, MemoryConfig
from ajvllm.execution.qwen2 import Qwen2Runner
from ajvllm.speculative.decoder import SpeculativeDecoder
from ajvllm.speculative.sampling import categorical, verify

pytestmark = pytest.mark.cuda


@torch.inference_mode()
def test_rejection_preserves_target_distribution_cuda():
    torch.manual_seed(17)
    n = 50000
    p = torch.tensor([0.1, 0.3, 0.6], device="cuda")
    q = torch.tensor([0.6, 0.3, 0.1], device="cuda")
    proposals = categorical(q.expand(n, 3), torch.rand(n, device="cuda"))[:, None]
    result, _ = verify(p.expand(n, 2, 3), q.expand(n, 1, 3), proposals, torch.rand(n, 2, device="cuda"))
    frequencies = torch.bincount(result[:, 0, 0].long(), minlength=3) / n
    torch.testing.assert_close(frequencies, p, atol=0.008, rtol=0)


@torch.inference_mode()
def test_all_accepted_bonus_and_forced_rejection_cuda():
    p = torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]], device="cuda")
    ids = torch.tensor([1, 0], device="cuda")
    result, accepted = verify(p, p[:2], ids, torch.tensor([0.99, 0.99, 0.1], device="cuda"))
    assert accepted.item() == 2
    assert result[:, 0].tolist() == [1, 0, 0]
    q = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device="cuda")
    result, accepted = verify(p, q, ids, torch.tensor([0.99, 0.2, 0.5], device="cuda"))
    assert accepted.item() == 0 and result[0, 0].item() == 0


def build(model, draft, *, budget=12, blocks=24, context=48, backend="eager", prefix=True):
    config = EngineConfig(
        max_model_len=context, max_num_seqs=3, max_num_batched_tokens=budget, max_prefill_chunk_size=3
    )
    memory = MemoryConfig(block_size=4, num_blocks=blocks, enable_prefix_cache=prefix)
    runner = Qwen2Runner(
        model, engine_config=config, memory_config=memory, compute_config=ComputeConfig(backend=backend)
    )
    if draft is None:
        return Engine(runner, config)
    c = draft.config
    peer = runner.kv_cache.create_peer(layers=c.num_hidden_layers, kv_heads=c.num_key_value_heads, head_dim=c.head_dim)
    draft_runner = Qwen2Runner(
        draft, engine_config=config, memory_config=memory, kv_cache=peer, compute_config=ComputeConfig(backend=backend)
    )
    return Engine(runner, config, decoder=SpeculativeDecoder(runner, draft_runner, 3))


@pytest.mark.parametrize("budget", [1, 5, 12])
@pytest.mark.parametrize("different", [False, True])
@torch.inference_mode()
def test_greedy_matches_target_with_mixed_arrivals_and_rollbacks(budget, different):
    model, _ = pair(tiny_config())
    draft = copy.deepcopy(model)
    if different:
        torch.manual_seed(127)
        draft.lm_head = torch.nn.Linear(64, 97, bias=False, device="cuda")
    params = SamplingParams(
        temperature=0, max_tokens=11, repetition_penalty=1.1, presence_penalty=0.2, min_tokens=3, stop_token_ids=(0,)
    )
    outputs = []
    for assistant in (None, draft):
        engine = build(model, assistant, budget=budget)
        engine.add_request("a", [1, 2, 3, 4, 5, 6, 7], params)
        finals = {}
        for step in range(120):
            if step == 3:
                engine.add_request("b", [3, 4, 5], params)
            if step == 5:
                engine.add_request("cancel", [6] * 15, params)
                assert engine.cancel_request("cancel").finished
            for output in engine.step():
                if output.finished:
                    finals[output.request_id] = output.output_token_ids
            assert engine.last_batch.num_scheduled_tokens <= budget
            for rid, state in engine.runner.kv_cache.states.items():
                assert len(state.tokens) == engine._scheduler.requests[rid].num_computed_tokens
                if assistant is not None:
                    assert engine.decoder.draft.kv_cache.states[rid] is state
            if step > 5 and not engine.has_unfinished_requests:
                break
        assert set(finals) == {"a", "b"}
        assert not engine.runner.kv_cache.states
        assert all(n == 0 for n in engine.runner.kv_cache.blocks.ref_counts)
        outputs.append(finals)
    assert outputs[0] == outputs[1]


@torch.inference_mode()
def test_triton_prefix_reuse_and_small_pool_admission():
    model, _ = pair(tiny_config(), torch.bfloat16)
    engine = build(model, copy.deepcopy(model), budget=12, blocks=12, context=32, backend="triton")
    params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    for i in range(3):
        engine.add_request(str(i), [1, 2, 3, 4] * 4, params)
    first = {o.request_id: o.output_token_ids for o in engine.run() if o.finished}
    engine.add_request("repeat", [1, 2, 3, 4] * 4, params)
    repeated = [o for o in engine.run() if o.finished][0]
    assert repeated.output_token_ids == first["0"]
    assert engine.runner.kv_cache.hit_tokens > 0
    assert engine.decoder.counters["accepted_tokens"] > 0
    assert not engine.runner.kv_cache.states


@torch.inference_mode()
def test_stop_inside_accepted_chunk_and_context_limit():
    model, _ = pair(tiny_config())
    model.lm_head = torch.nn.Linear(64, 97, device="cuda")
    model.lm_head.weight.zero_()
    model.lm_head.bias.fill_(-10)
    model.lm_head.bias[0] = 10
    engine = build(model, copy.deepcopy(model))
    engine.add_request("stop", [2, 3], SamplingParams(temperature=0, max_tokens=12, min_tokens=2, stop_token_ids=(0,)))
    outputs = list(engine.run())
    assert outputs[-1].finish_reason == "stop"
    assert outputs[-1].output_token_ids == (1, 1, 0)
    assert outputs[-1].new_token_ids == (1, 0)
    assert not engine.runner.kv_cache.states
    engine = build(model, copy.deepcopy(model), context=8)
    engine.add_request("limit", [2] * 6, SamplingParams(temperature=0, max_tokens=20, ignore_eos=True))
    final = list(engine.run())[-1]
    assert final.finish_reason == "length" and len(final.output_token_ids) == 2


@torch.inference_mode()
def test_paired_copy_on_write_keeps_both_models_pages():
    model, _ = pair(tiny_config())
    engine = build(model, copy.deepcopy(model))
    cache = engine.runner.kv_cache
    peer = engine.decoder.draft.kv_cache
    cache.attach("a", ())
    assert cache.reserve("a", 2)
    cache.commit("a", (1, 2))
    block = cache.states["a"].blocks[0]
    cache.storage.tensor[:, :, block].fill_(11)
    peer.storage.tensor[:, :, block].fill_(22)
    cache.fork("a", "b")
    assert peer.reserve("b", 3)
    copied = cache.states["b"].blocks[0]
    assert copied != block
    assert cache.cow_copies == peer.snapshot()["cow_copies"] == 1
    assert (cache.storage.tensor[:, :, copied] == 11).all()
    assert (peer.storage.tensor[:, :, copied] == 22).all()
    cache.release("a")
    cache.release("b")
    assert all(count == 0 for count in cache.blocks.ref_counts)
