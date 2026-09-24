"""Start a persistent single-GPU inference server with conservative memory budgeting."""

import argparse
import json
import tomllib
from dataclasses import asdict
from pathlib import Path

import torch
import uvicorn

from ajvllm import EngineConfig
from ajvllm.config.advanced import GraphConfig, QuantizationConfig
from ajvllm.config.compute import ComputeConfig
from ajvllm.config.memory import MemoryConfig
from ajvllm.runtime.inference import InferenceRuntime
from ajvllm.serving.http import create_app
from ajvllm.serving.presentation import memory_display
from ajvllm.serving.service import EngineService
from ajvllm.tokenization.qwen2 import Qwen2Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("model/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--config", type=Path, default=Path("configs/engine/qwen2.toml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--max-pending-requests", type=int, default=128)
    parser.add_argument(
        "--profile-steps", action="store_true", help="Synchronize CUDA to measure engine step wall time"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    with args.config.open("rb") as file:
        settings = tomllib.load(file)
    config = EngineConfig(**settings["engine"])
    memory_config = MemoryConfig(**settings.get("memory", {}))
    runtime = InferenceRuntime.from_directory(
        args.model,
        config,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        memory_config=memory_config,
        compute_config=ComputeConfig(**settings.get("compute", {})),
        graph_config=GraphConfig(**settings.get("graphs", {})),
        quantization_config=QuantizationConfig(**settings.get("quantization", {})),
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    service = EngineService(runtime, max_pending_requests=args.max_pending_requests, profile_steps=args.profile_steps)
    print(
        json.dumps(
            {"resolved_engine": asdict(runtime.engine.config), "memory": memory_display(runtime.budget.snapshot())}
        ),
        flush=True,
    )
    app = create_app(service, Qwen2Tokenizer(args.model))
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
