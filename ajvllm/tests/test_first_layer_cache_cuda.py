"""GPU cache identity, invalidation, graph replay and model-path regressions."""

import json
from dataclasses import asdict

import pytest
import torch
from model_inputs import forward_tokens
from safetensors.torch import save_file

from ajvllm.config import ComputeConfig, EngineConfig, GraphConfig, MemoryConfig
from ajvllm.config.first_layer_cache import FirstLayerCacheConfig
from ajvllm.execution.qwen2 import Qwen2Runner
from ajvllm.modeling.qwen2.config import Qwen2Config
from ajvllm.modeling.qwen2.first_layer_cache import (
    FirstLayerFrontEnd,
    artifact_path,
    build_cache,
    configure_first_layer_cache,
    load_table,
)
from ajvllm.modeling.qwen2.model import Qwen2ForCausalLM
from ajvllm.modeling.qwen2.projections import pack_projections
from ajvllm.scheduling.batch import Phase, ScheduledRequest, SchedulerOutput

pytestmark = pytest.mark.cuda


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(12)
    cfg = Qwen2Config(67, 64, 96, 2, 4, 2, 128, tie_word_embeddings=True, torch_dtype="float32")
    model = Qwen2ForCausalLM(cfg).cuda().eval().requires_grad_(False)
    tensors = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(asdict(cfg) | {"model_type": "qwen2"}))
    return tmp_path, model


@pytest.mark.parametrize("optimized", [False, True])
@torch.inference_mode()
def test_chunked_artifact_matches_direct_and_changed_ids_in_graph(checkpoint, optimized):
    directory, _ = checkpoint
    module, metadata = FirstLayerFrontEnd.from_directory(directory, dtype=torch.float32, optimized=optimized)
    path = build_cache(module, metadata, directory / "cache", chunk_tokens=13)
    module.table = load_table(path, metadata, device="cuda", dtype=torch.float32)
    for ids in [torch.tensor([66, 0, 66, 1], device="cuda"), torch.arange(67, device="cuda")]:
        expected, actual = module(ids), module.lookup(ids)
        for a, b in zip(expected, actual, strict=True):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=3e-6)
    ids = torch.tensor([1, 2, 3], device="cuda")
    for _ in range(3):
        module.lookup(ids)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = module.lookup(ids)
    ids.copy_(torch.tensor([66, 0, 4], device="cuda"))
    graph.replay()
    for a, b in zip(output, module(ids), strict=True):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=3e-6)


@torch.inference_mode()
def test_invalidates_changed_weights_dtype_and_backend(checkpoint):
    directory, _ = checkpoint
    module, metadata = FirstLayerFrontEnd.from_directory(directory, dtype=torch.float32)
    path = build_cache(module, metadata, directory / "cache")
    for dtype, optimized in [(torch.bfloat16, True), (torch.float32, False)]:
        _, other = FirstLayerFrontEnd.from_directory(directory, dtype=dtype, optimized=optimized)
        assert artifact_path(directory / "cache", other) != path
        with pytest.raises(ValueError, match="manifest"):
            load_table(path, other, device="cuda", dtype=dtype)
    from safetensors.torch import load_file

    tensors = load_file(directory / "model.safetensors")
    tensors["model.layers.0.self_attn.k_proj.bias"][0] += 1
    save_file(tensors, directory / "model.safetensors")
    _, other = FirstLayerFrontEnd.from_directory(directory, dtype=torch.float32)
    assert other["fingerprint"] != metadata["fingerprint"]


@torch.inference_mode()
def test_model_full_chunked_decode_and_residual(checkpoint):
    directory, model = checkpoint
    module, metadata = FirstLayerFrontEnd.from_directory(directory, dtype=torch.float32, optimized=False)
    build_cache(module, metadata, directory / "cache", chunk_tokens=11)
    tokens = torch.tensor([1, 3, 7, 2, 1, 66, 0, 13], device="cuda")
    expected = forward_tokens(model, tokens).logits
    configure_first_layer_cache(
        model, directory, FirstLayerCacheConfig(True, str(directory / "cache")),
        compute_config=ComputeConfig("eager"),
    )
    actual = forward_tokens(model, tokens).logits
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
    cache, rows = None, []
    for chunk in tokens.split(3):
        output = forward_tokens(model, chunk, cache)
        cache = output.caches[0]
        rows.append(output.logits)
    torch.testing.assert_close(torch.cat(rows), expected, rtol=3e-5, atol=3e-5)
    assert "first_layer_qkv" not in model.state_dict()
    assert model.model.embed_tokens.weight is model.lm_head.weight


@torch.inference_mode()
def test_paged_triton_and_graph_use_cached_qkv(checkpoint):
    directory, model = checkpoint
    module, metadata = FirstLayerFrontEnd.from_directory(directory, dtype=torch.float32)
    build_cache(module, metadata, directory / "cache")
    expected = forward_tokens(model, torch.tensor([1, 7, 8, 3], device="cuda")).logits
    configure_first_layer_cache(
        model, directory, FirstLayerCacheConfig(True, str(directory / "cache")),
        compute_config=ComputeConfig("triton"),
    )
    pack_projections(model)
    # Catch accidental projection/norm execution after the cache is attached.
    def unused(*args):
        raise AssertionError("cached first-layer transform was executed")

    model.model.layers[0].input_layernorm.register_forward_pre_hook(unused)
    model.model.layers[0].self_attn.qkv_proj.register_forward_pre_hook(unused)
    runner = Qwen2Runner(
        model, memory_config=MemoryConfig(num_blocks=16), compute_config=ComputeConfig("triton"),
        engine_config=EngineConfig(max_model_len=128, max_num_batched_tokens=8, max_num_seqs=1),
        graph_config=GraphConfig(enabled=True, batch_sizes=(1,), memory_limit_mb=32),
    )
    try:
        for pos, token in enumerate([1, 7, 8, 3]):
            batch = SchedulerOutput((ScheduledRequest("r", (token,), pos, Phase.DECODE, True),))
            actual = runner.execute(batch)["r"]
            torch.testing.assert_close(actual, expected[pos], atol=4e-4, rtol=4e-4)
        assert runner.graphs.snapshot()["replays"] > 0
    finally:
        runner.release("r")


@pytest.mark.model
@pytest.mark.parametrize("name", ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct"])
@torch.inference_mode()
def test_local_cached_runtime_and_logits(name):
    from ajvllm.runtime.inference import InferenceRuntime

    runtime = InferenceRuntime.from_directory(
        f"model/{name}", EngineConfig(max_model_len=128, max_num_batched_tokens=128, max_num_seqs=3),
        dtype=torch.bfloat16, memory_config=MemoryConfig(num_blocks=32, enable_prefix_cache=False),
        compute_config=ComputeConfig("triton"),
        first_layer_cache_config=FirstLayerCacheConfig(enabled=True),
    )
    runner = runtime.engine.runner
    model = runner.model
    table = model.first_layer_qkv
    assert runtime.budget.stats.first_layer_cache_bytes == table.numel() * 2
    assert runtime.budget.stats.buffer_bytes >= table.numel() * 2
    plan = SchedulerOutput((
        ScheduledRequest("a", (1, 7, 8, 3, 16, 2, 23, 17), 0, Phase.PREFILL, True),
        ScheduledRequest("b", (0, 1, 7), 0, Phase.PREFILL, True),
    ))
    try:
        cached = runner.execute(plan)
        for rid in ("a", "b"):
            runner.release(rid)
        # No graphs in this test: switching only isolates numerical equivalence.
        model.first_layer_qkv = None
        reference = runner.execute(plan)
        for rid in ("a", "b"):
            runner.release(rid)
        padded = runner.execute(SchedulerOutput(plan.requests + (
            ScheduledRequest("pad", (1,) * 53, 0, Phase.PREFILL, False),
        )))
        # BF16 shape-dependent GEMM rounding is also present without caching.
        # Compare to that independent control rather than promise bitwise parity.
        for rid in cached:
            error = (cached[rid] - reference[rid]).square().mean().sqrt()
            control = (padded[rid] - reference[rid]).square().mean().sqrt()
            scale = reference[rid].square().mean().sqrt()
            assert (error / scale).item() <= max(0.001, 1.25 * (control / scale).item())
            assert (cached[rid] - reference[rid]).abs().max().item() < 0.3
        print(name, "max logit error", max((cached[r] - reference[r]).abs().max().item() for r in cached))
    finally:
        runtime.engine.close()
