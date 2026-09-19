"""Integration acceptance against the downloaded checkpoint, entirely on CUDA."""

import json
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM

from ajvllm import Engine, EngineConfig, SamplingParams
from ajvllm.execution.qwen2 import Qwen2Runner, read_eos_token_ids
from ajvllm.modeling.qwen2.weights import load_qwen2
from ajvllm.tokenization.qwen2 import Qwen2Tokenizer

pytestmark = [pytest.mark.cuda, pytest.mark.model]


@pytest.fixture(scope="module")
def checkpoint():
    path = Path(os.environ.get("AJVLLM_TEST_MODEL", "model/Qwen2.5-0.5B-Instruct"))
    assert path.is_dir(), f"local checkpoint required: {path}"
    assert torch.cuda.is_available(), "CUDA is required; there is no CPU fallback"
    torch.backends.cuda.matmul.allow_tf32 = False
    return path


@pytest.fixture
def models(checkpoint):
    native = load_qwen2(checkpoint, dtype=torch.float32)
    reference = (
        AutoModelForCausalLM.from_pretrained(
            checkpoint,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        .cuda()
        .eval()
    )
    yield native, reference
    del native, reference
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def tokenizer(checkpoint):
    return Qwen2Tokenizer(checkpoint)


@pytest.mark.parametrize("text", ["What is the capital of France?", "Explain why the sky is blue in one sentence."])
@torch.inference_mode()
def test_real_full_chunked_and_forced_decode(models, tokenizer, text):
    native, reference = models
    ids = torch.tensor(tokenizer.encode_chat([{"role": "user", "content": text}]), device="cuda")
    full = native(ids).logits
    oracle = reference(ids[None], use_cache=False).logits[0].float()
    torch.testing.assert_close(full, oracle, atol=2e-4, rtol=2e-5)
    cache = None
    hf_cache = None
    rows = []
    for chunk in ids.split(7):
        out = native(chunk, cache)
        expected = reference(chunk[None], past_key_values=hf_cache, use_cache=True)
        torch.testing.assert_close(out.logits, expected.logits[0].float(), atol=2e-4, rtol=2e-5)
        rows.append(out.logits)
        cache, hf_cache = out.cache, expected.past_key_values
    chunked = torch.cat(rows)
    torch.testing.assert_close(chunked, full, atol=2e-4, rtol=2e-5)
    print(
        json.dumps(
            {
                "prompt": text,
                "dtype": "float32",
                "full_vs_hf_max_error": (full - oracle).abs().max().item(),
                "chunk_vs_full_max_error": (chunked - full).abs().max().item(),
            }
        )
    )
    # Consume identical generated tokens to isolate incremental-forward correctness.
    for _ in range(8):
        token = out.logits[-1].argmax().reshape(1)
        out = native(token, cache)
        expected = reference(token[None], past_key_values=hf_cache, use_cache=True)
        torch.testing.assert_close(out.logits, expected.logits[0].float(), atol=2e-4, rtol=2e-5)
        assert out.logits.argmax().item() == expected.logits.argmax().item()
        cache, hf_cache = out.cache, expected.past_key_values


@torch.inference_mode()
def test_real_engine_greedy_generation_and_eos(models, tokenizer, checkpoint):
    native, reference = models
    runner = Qwen2Runner(native, read_eos_token_ids(checkpoint, native.config.vocab_size))
    assert runner.eos_token_ids == (151645, 151643)
    ids = tokenizer.encode_chat([{"role": "user", "content": "What is the capital of France?"}])
    params = SamplingParams(max_tokens=24, temperature=0)
    actual = []
    for budget in (7, 128):
        engine = Engine(runner, EngineConfig(max_model_len=256, max_num_batched_tokens=budget))
        engine.add_request("answer", ids, params)
        final = list(engine.run())[-1]
        assert final.finished and runner.cache_bytes == 0
        actual.append(final.output_token_ids)
    assert actual[0] == actual[1]
    generated = reference.generate(
        torch.tensor([ids], device="cuda"),
        attention_mask=torch.ones((1, len(ids)), device="cuda", dtype=torch.long),
        do_sample=False,
        max_new_tokens=params.max_tokens,
        repetition_penalty=1.0,
        eos_token_id=list(runner.eos_token_ids),
        pad_token_id=151643,
    )[0, len(ids) :].tolist()
    assert list(actual[0]) == generated
    print(json.dumps({"greedy_text": tokenizer.decode(actual[0]), "token_ids": actual[0]}))


@torch.inference_mode()
def test_real_bfloat16_matches_oracle_for_same_chunk_schedule(checkpoint, tokenizer):
    native = load_qwen2(checkpoint, dtype=torch.bfloat16)
    reference = (
        AutoModelForCausalLM.from_pretrained(
            checkpoint,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .cuda()
        .eval()
    )
    ids = torch.tensor(
        tokenizer.encode_chat([{"role": "user", "content": "What is the capital of France?"}]), device="cuda"
    )
    full = native(ids).logits
    oracle = reference(ids[None], use_cache=False).logits[0].float()
    torch.testing.assert_close(full, oracle, atol=0, rtol=0)
    cache = None
    hf_cache = None
    rows = []
    for chunk in ids.split(7):
        out = native(chunk, cache)
        expected = reference(chunk[None], past_key_values=hf_cache, use_cache=True)
        torch.testing.assert_close(out.logits, expected.logits[0].float(), atol=0, rtol=0)
        cache, hf_cache = out.cache, expected.past_key_values
        rows.append(out.logits)
    print(
        json.dumps(
            {
                "dtype": "bfloat16",
                "native_vs_hf_max_error": (full - oracle).abs().max().item(),
                "chunk_vs_full_max_error": (torch.cat(rows) - full).abs().max().item(),
            }
        )
    )
    # BF16 batch/chunk invariance is not promised; the independent oracle shows the same effect.


@torch.inference_mode()
def test_real_packed_mixed_batch_matches_full_oracle(models, tokenizer):
    from ajvllm.execution.batch import ModelBatch
    from ajvllm.scheduling.batch import Phase, ScheduledRequest, SchedulerOutput

    native, reference = models
    prompts = [
        tokenizer.encode_chat([{"role": "user", "content": text}])
        for text in ("What is the capital of France?", "Name two planets.", "Explain grouped-query attention.")
    ]
    starts = [0, len(prompts[1]) - 1, 7]
    runner = Qwen2Runner(native)
    for row, start in enumerate(starts):
        if start:
            runner._caches[str(row)] = native(torch.tensor(prompts[row][:start], device="cuda")).cache
    plan = SchedulerOutput(
        tuple(
            ScheduledRequest(str(row), tuple(ids[start:]), start, Phase.DECODE if row == 1 else Phase.PREFILL, True)
            for row, (ids, start) in enumerate(zip(prompts, starts, strict=True))
        )
    )
    calls = []
    hook = native.register_forward_pre_hook(lambda _, args: calls.append(args[0]))
    try:
        actual = runner.execute(plan)
    finally:
        hook.remove()
    assert len(calls) == 1 and isinstance(calls[0], ModelBatch)
    assert calls[0].num_requests == 3
    for row, ids in enumerate(prompts):
        expected = reference(torch.tensor([ids], device="cuda"), use_cache=False).logits[0, -1].float()
        torch.testing.assert_close(torch.tensor(actual[str(row)], device="cuda"), expected, atol=2e-4, rtol=2e-5)
        runner.release(str(row))
    assert runner.cache_bytes == 0
