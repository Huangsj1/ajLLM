"""Translate one TOML preset into explicit ajvLLM and vLLM comparison commands."""

import argparse
import json
import sys
import tomllib
from dataclasses import asdict
from pathlib import Path

from ajvllm.config import ComputeConfig, EngineConfig, GraphConfig, MemoryConfig, QuantizationConfig


def parse_args(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=Path("configs/engine/benchmark.toml"))
    config_arg, _ = config_parser.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    settings = tomllib.loads(config_arg.config.read_text())
    engine = EngineConfig(**settings["engine"])
    graphs = GraphConfig(**settings.get("graphs", {}))
    sizes = sorted({1 << i for i in range(engine.max_num_seqs.bit_length())} | {engine.max_num_seqs})
    parser.add_argument("--model", type=Path, default=Path("model/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--dataset", type=Path, default=Path("benchmarks/datasets/long.jsonl"))
    parser.add_argument("--concurrency", type=int, nargs="+", default=sizes)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--requests", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--token-budget", type=int, default=engine.max_num_batched_tokens)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--graphs", action=argparse.BooleanOptionalAction, default=graphs.enabled)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/comparison"))
    defaults = settings.get("compare", {})
    allowed = {a.dest for a in parser._actions} - {"help", "config", "token_budget", "graphs"}
    if unknown := defaults.keys() - allowed:
        parser.error(
            f"unknown [compare] settings: {sorted(unknown)}; engine/graph settings belong in their own sections"
        )
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    for name in ("model", "dataset", "output"):
        setattr(args, name, Path(getattr(args, name)))
    args.concurrency = sorted(set(args.concurrency))
    if not args.concurrency or not args.prompt_lengths:
        parser.error("concurrency and prompt_lengths must not be empty")
    if args.requests is None:
        args.requests = max(24, 2 * max(args.concurrency))
    if min(*args.concurrency, *args.prompt_lengths, args.requests, args.token_budget, args.repeats) < 1:
        parser.error("counts must be positive")
    if args.max_tokens < 2 or args.requests < max(args.concurrency):
        parser.error("max_tokens must be >= 2 and requests >= maximum concurrency")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu_memory_utilization must be in (0, 1]")
    if max(args.prompt_lengths) + args.max_tokens > engine.max_model_len:
        parser.error("prompt length plus output length exceeds engine.max_model_len")
    settings["engine"]["max_num_batched_tokens"] = args.token_budget
    settings.setdefault("graphs", {})["enabled"] = args.graphs
    engine = EngineConfig(**settings["engine"])
    if engine.max_prefill_chunk_size is not None or engine.max_prefill_tokens_per_step is not None:
        parser.error("remove optional prefill caps for comparison; vLLM has no equivalent per-request cap")
    memory = MemoryConfig(**settings.get("memory", {}))
    quantization = QuantizationConfig(**settings.get("quantization", {}))
    if memory.backend != "paged" or quantization.mode != "none":
        parser.error("comparison currently supports paged KV and unquantized weights only")
    args.engine_settings = dict(
        engine=asdict(engine),
        memory=asdict(memory),
        graphs=asdict(GraphConfig(**settings.get("graphs", {}))),
        compute=asdict(ComputeConfig(**settings.get("compute", {}))),
        quantization=asdict(quantization),
    )
    # Optional None values must be omitted from TOML, which has no null literal.
    args.engine_settings = {
        section: {k: v for k, v in values.items() if v is not None} for section, values in args.engine_settings.items()
    }
    args.gpu = str(args.gpu)
    args.model = args.model.resolve()
    return args


def commands(args, output):
    settings = args.engine_settings
    engine, memory, graphs = (settings[key] for key in ("engine", "memory", "graphs"))
    kv_bytes = None
    if memory.get("num_blocks") is not None:
        cfg = json.loads((args.model / "config.json").read_text())
        kv_bytes = (
            memory["num_blocks"]
            * memory["block_size"]
            * cfg["num_hidden_layers"]
            * 2
            * cfg["num_key_value_heads"]
            * cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
            * 2
        )
    config = output / "ajvllm.toml"
    config.write_text(
        "\n".join(
            f"[{section}]\n" + "\n".join(f"{k} = {json.dumps(v)}" for k, v in values.items()) + "\n"
            for section, values in settings.items()
        )
    )
    common = [
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    aj = [
        sys.executable,
        "-m",
        "ajvllm.workflows.serve",
        "--model",
        str(args.model),
        "--config",
        str(config),
        "--max-pending-requests",
        str(max(128, 2 * max(args.concurrency))),
        *common,
    ]
    vl = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(args.model),
        "--served-model-name",
        "comparison",
        "--max-model-len",
        str(engine["max_model_len"]),
        "--max-num-seqs",
        str(engine["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(args.token_budget),
        "--block-size",
        str(memory["block_size"]),
        "--enable-prefix-caching" if memory["enable_prefix_cache"] else "--no-enable-prefix-caching",
        "--enable-chunked-prefill" if engine["enable_chunked_prefill"] else "--no-enable-chunked-prefill",
        "--generation-config",
        "vllm",
        "--no-enable-log-requests",
        *common,
    ]
    if kv_bytes is not None:
        vl += ["--kv-cache-memory-bytes", str(kv_bytes)]
    if args.graphs:
        vl += [
            "--compilation-config",
            json.dumps(dict(mode=0, cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=graphs["batch_sizes"])),
        ]
    else:
        vl += ["--enforce-eager"]
    return {"ajvllm": aj, "vllm": vl}, dict(
        context=engine["max_model_len"],
        max_seqs=engine["max_num_seqs"],
        kv_pool_bytes=kv_bytes,
        memory_mode="fixed_kv_bytes" if kv_bytes is not None else "same_utilization",
    )
