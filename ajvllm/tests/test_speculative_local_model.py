"""Opt-in real-checkpoint greedy equivalence with one resident target/draft pair."""

import pytest
import torch

from ajvllm import Engine, EngineConfig, SamplingParams
from ajvllm.config import ComputeConfig, GraphConfig, MemoryConfig
from ajvllm.execution.qwen2 import Qwen2Runner
from ajvllm.modeling.qwen2.weights import load_qwen2
from ajvllm.speculative.decoder import SpeculativeDecoder
from ajvllm.tokenization.qwen2 import Qwen2Tokenizer

pytestmark = [pytest.mark.cuda, pytest.mark.model]


@pytest.mark.parametrize("dtype,backend", [(torch.float32, "eager"), (torch.bfloat16, "triton")])
@torch.inference_mode()
def test_local_speculative_greedy_equivalence(dtype, backend):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    directory = "model/Qwen2.5-1.5B-Instruct"
    target = load_qwen2(directory, device="cuda", dtype=dtype)
    draft = load_qwen2("model/Qwen2.5-0.5B-Instruct", device="cuda", dtype=dtype)
    tokenizer = Qwen2Tokenizer(directory)
    config = EngineConfig(max_model_len=512, max_num_seqs=3, max_num_batched_tokens=128)
    memory = MemoryConfig(block_size=16, num_blocks=128, enable_prefix_cache=False)
    compute = ComputeConfig(backend=backend)
    graphs = GraphConfig(enabled=backend == "triton", batch_sizes=(1, 2, 3), memory_limit_mb=64)
    prompts = ["What is the capital of France?", "Count from one to ten.", "Explain grouped-query attention briefly."]
    results, traces = [], []
    for speculative in (False, True):
        runner = Qwen2Runner(
            target, engine_config=config, memory_config=memory, compute_config=compute, graph_config=graphs
        )
        decoder = None
        if speculative:
            c = draft.config
            peer = runner.kv_cache.create_peer(
                layers=c.num_hidden_layers, kv_heads=c.num_key_value_heads, head_dim=c.head_dim
            )
            draft_runner = Qwen2Runner(
                draft,
                engine_config=config,
                memory_config=memory,
                compute_config=compute,
                graph_config=graphs,
                kv_cache=peer,
            )
            decoder = SpeculativeDecoder(runner, draft_runner, 3)
        trace = {}
        original = runner.execute

        def capture(batch, **kwargs):
            prefixes = {i.request_id: runner.kv_cache.states[i.request_id].tokens for i in batch.requests}
            result = original(batch, **kwargs)
            if dtype == torch.bfloat16:
                for item in batch.requests:
                    if not item.do_sample:
                        continue
                    rows = result[item.request_id]
                    positions = [(len(item.token_ids), rows)] if rows.ndim == 1 else enumerate(rows, start=1)
                    for length, row in positions:
                        trace[(item.request_id, prefixes[item.request_id] + item.token_ids[:length])] = row.clone()
            return result

        runner.execute = capture
        traces.append(trace)
        engine = Engine(runner, config, decoder=decoder)
        for i, prompt in enumerate(prompts):
            engine.add_request(
                str(i),
                tokenizer.encode_chat([{"role": "user", "content": prompt}]),
                SamplingParams(temperature=0, max_tokens=32, ignore_eos=True),
            )
        results.append({o.request_id: o.output_token_ids for o in engine.run() if o.finished})
        if speculative:
            assert decoder.counters["rounds"] > 0
            assert decoder.counters["accepted_tokens"] > 0
        engine.close()
        del runner.execute
    if dtype == torch.float32:
        assert results[0] == results[1]
    else:
        # BF16 GEMM/attention shapes can reorder near-tied maxima. Diagnose every
        # first divergence at an identical prefix instead of hiding it as exact parity.
        for rid, baseline in results[0].items():
            speculative = results[1][rid]
            assert len(baseline) == len(speculative) == 32
            first = next((i for i, (a, b) in enumerate(zip(baseline, speculative, strict=True)) if a != b), None)
            if first is None:
                continue
            prefix = tuple(tokenizer.encode_chat([{"role": "user", "content": prompts[int(rid)]}])) + baseline[:first]
            a, b = traces[0][rid, prefix], traces[1][rid, prefix]
            assert (a - b).abs().max().item() < 1.0
            assert a.max().item() - a[speculative[first]].item() <= 0.25
            assert b.max().item() - b[baseline[first]].item() <= 0.25
