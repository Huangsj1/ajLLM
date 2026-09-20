"""CUDA numerical, loader, and engine tests. No CPU model fallback."""

import json
from dataclasses import asdict, replace
from unittest.mock import patch

import pytest
import torch
from model_inputs import forward_tokens
from safetensors.torch import load_file, save_file
from transformers import Qwen2Config as HFConfig
from transformers import Qwen2ForCausalLM as HFModel

from ajvllm import Engine, EngineConfig, SamplingParams
from ajvllm.execution.qwen2 import Qwen2Runner
from ajvllm.modeling.qwen2 import Qwen2Config, Qwen2ForCausalLM, load_qwen2
from ajvllm.scheduling.batch import Phase, ScheduledRequest, SchedulerOutput

pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module", autouse=True)
def cuda_required():
    assert torch.cuda.is_available(), "These tests require CUDA; do not substitute CPU or silently skip them"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def tiny_config(**kwargs):
    values = dict(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        torch_dtype="float32",
    )
    return Qwen2Config(**(values | kwargs))


def pair(config, dtype=torch.float32):
    torch.manual_seed(42)
    hf_config = HFConfig(**asdict(config))
    hf_config._attn_implementation = "eager"
    reference = HFModel(hf_config).to(device="cuda", dtype=dtype).eval()
    native = Qwen2ForCausalLM(config).to(device="cuda", dtype=dtype).eval()
    native.load_state_dict(reference.state_dict())
    return native, reference


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_dense_gqa_full_chunked_and_decode(kv_heads, dtype):
    native, reference = pair(tiny_config(num_key_value_heads=kv_heads), dtype)
    ids = torch.tensor([1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6], device="cuda")
    full = forward_tokens(native, ids)
    expected = reference(ids[None], use_cache=False).logits[0].float()
    # Same dtype and schedule should match the independent eager oracle tightly.
    tolerance = {torch.float32: 2e-6, torch.float16: 0.002, torch.bfloat16: 0.02}[dtype]
    torch.testing.assert_close(full.logits, expected, atol=tolerance, rtol=1e-4)
    cache = None
    hf_cache = None
    rows = []
    for start, stop in [(0, 3), (3, 8), (8, 9), (9, 10), (10, 11)]:
        result = forward_tokens(native, ids[start:stop], cache)
        oracle = reference(ids[None, start:stop], past_key_values=hf_cache, use_cache=True)
        torch.testing.assert_close(result.logits, oracle.logits[0].float(), atol=tolerance, rtol=1e-4)
        if cache is not None:
            assert cache[0][0].shape[1] == start  # Functional, unmodified input cache.
        cache, hf_cache = result.caches[0], oracle.past_key_values
        for k, v in cache:
            assert k.shape == v.shape == (kv_heads, stop, native.config.head_dim)
            assert k.is_contiguous() and v.is_contiguous() and k.is_cuda
        rows.append(result.logits)
    torch.testing.assert_close(torch.cat(rows), full.logits, atol=tolerance, rtol=1e-3)


@torch.inference_mode()
def test_causal_mask_ignores_future_and_partial_prefill_skips_head():
    model, _ = pair(tiny_config())
    first = torch.tensor([1, 2, 3, 4, 5], device="cuda")
    changed = first.clone()
    changed[3:] = 10
    torch.testing.assert_close(
        forward_tokens(model, first).logits[:3], forward_tokens(model, changed).logits[:3], atol=1e-6, rtol=1e-5
    )
    calls = []
    handle = model.lm_head.register_forward_hook(lambda *_: calls.append(True))
    result = forward_tokens(model, first[:2], logits_to_keep=None)
    assert result.logits is None and not calls
    result = forward_tokens(model, first[2:], result.caches[0], logits_to_keep=1)
    assert result.logits.shape == (1, 97) and len(calls) == 1
    handle.remove()


def save_checkpoint(directory, config, state, shard=False):
    directory.mkdir(exist_ok=True)
    data = asdict(config) | {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"], "eos_token_id": 96}
    (directory / "config.json").write_text(json.dumps(data))
    # Clone only for on-disk serialization of potentially tied parameters.
    state = {name: value.detach().cpu().clone() for name, value in state.items()}
    if config.tie_word_embeddings:
        state.pop("lm_head.weight")
    if not shard:
        save_file(state, directory / "model.safetensors")
        return
    names = sorted(state)
    weight_map = {}
    for number, group in enumerate((names[::2], names[1::2])):
        filename = f"model-{number}.safetensors"
        save_file({name: state[name] for name in group}, directory / filename)
        weight_map.update({name: filename for name in group})
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


@pytest.mark.parametrize("tied,sharded", [(True, False), (True, True), (False, True)])
@torch.inference_mode()
def test_strict_checkpoint_loading_and_tying(tmp_path, tied, sharded):
    config = tiny_config(tie_word_embeddings=tied)
    native, _ = pair(config)
    save_checkpoint(tmp_path, config, native.state_dict(), shard=sharded)
    allocations = []
    original = torch.empty_like

    def allocate(tensor, *args, **kwargs):
        if tensor.is_meta and tensor.shape == (config.vocab_size, config.hidden_size):
            allocations.append(tensor.shape)
        return original(tensor, *args, **kwargs)

    with patch("torch.empty_like", side_effect=allocate):
        loaded = load_qwen2(tmp_path)
    assert len(allocations) == (1 if tied else 2)
    assert loaded.device.type == "cuda"
    assert (loaded.model.embed_tokens.weight is loaded.lm_head.weight) == tied
    ids = torch.tensor([4, 5, 6], device="cuda")
    torch.testing.assert_close(forward_tokens(loaded, ids).logits, forward_tokens(native, ids).logits, atol=0, rtol=0)


@pytest.mark.parametrize("fault", ["missing", "extra", "shape", "conflicting_tie"])
def test_reject_corrupt_checkpoint(tmp_path, fault):
    config = tiny_config()
    native, _ = pair(config)
    save_checkpoint(tmp_path, config, native.state_dict())
    path = tmp_path / "model.safetensors"
    state = load_file(path)
    if fault == "missing":
        state.pop("model.norm.weight")
    elif fault == "extra":
        state["unexpected.weight"] = torch.ones(1)
    elif fault == "shape":
        state["model.norm.weight"] = state["model.norm.weight"][:1].clone()
    else:
        state["lm_head.weight"] = state["model.embed_tokens.weight"] + 1
    save_file(state, path)
    with pytest.raises(ValueError, match="mismatch|conflicting"):
        load_qwen2(tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        {"model_type": "qwen2_moe"},
        {"use_sliding_window": True},
        {"rope_scaling": {"rope_type": "yarn", "factor": 4}},
        {"quantization_config": {}},
        {"num_key_value_heads": 3},
        {"hidden_act": "gelu"},
    ],
)
def test_reject_unsupported_architecture(tmp_path, change):
    data = asdict(tiny_config()) | {"model_type": "qwen2"} | change
    (tmp_path / "config.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        Qwen2Config.from_directory(tmp_path)


def test_runner_continuity_release_and_context_guard():
    model, _ = pair(tiny_config())
    runner = Qwen2Runner(model)
    batch = SchedulerOutput((ScheduledRequest("a", (1, 2), 0, Phase.PREFILL, False),))
    assert runner.execute(batch) == {}
    assert runner.cached_tokens("a") == 2
    assert runner.cache_bytes == 2 * 2 * 2 * 2 * 16 * 4
    with pytest.raises(ValueError, match="noncontiguous"):
        runner.execute(batch)
    runner.release("a")
    runner.release("a")
    assert runner.cache_bytes == 0 and runner.num_active_states == 0
    with pytest.raises(ValueError, match="context"):
        Engine(runner, EngineConfig(max_model_len=129))
    engine = Engine(runner, EngineConfig(max_model_len=128, max_num_batched_tokens=1))
    engine.add_request("a", [1, 2], SamplingParams(max_tokens=3))
    engine.step()
    engine.cancel_request("a")
    assert runner.num_active_states == 0


@torch.inference_mode()
def test_mixed_engine_batches_match_isolated_greedy_requests():
    model, _ = pair(tiny_config())
    params = SamplingParams(temperature=0, max_tokens=5)
    prompts = {"a": [1, 2], "b": [3, 4, 5, 6, 7, 8, 9]}
    config = EngineConfig(max_model_len=128, max_num_batched_tokens=4)
    expected = {}
    for rid, ids in prompts.items():
        engine = Engine(Qwen2Runner(model), config)
        engine.add_request(rid, ids, params)
        expected[rid] = list(engine.run())[-1].output_token_ids
    runner = Qwen2Runner(model)
    engine = Engine(runner, replace(config, max_prefill_chunk_size=2))
    engine.add_request("a", prompts["a"], params)
    outputs = engine.step()
    engine.add_request("b", prompts["b"], params)
    mixed = False
    while engine.has_unfinished_requests:
        outputs.extend(engine.step())
        mixed |= {item.phase for item in engine.last_batch.requests} == {Phase.PREFILL, Phase.DECODE}
    assert mixed
    assert {out.request_id: out.output_token_ids for out in outputs if out.finished} == expected
    assert runner.cache_bytes == 0
