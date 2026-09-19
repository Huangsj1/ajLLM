"""Run native Qwen2.5 text generation on a CUDA GPU from local model files."""

import argparse
import json
import tomllib
from dataclasses import asdict
from pathlib import Path

import torch

from ajvllm import Engine, EngineConfig, SamplingParams
from ajvllm.execution.qwen2 import Qwen2Runner
from ajvllm.tokenization.qwen2 import Qwen2Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("model/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--config", type=Path, default=Path("configs/engine/qwen2.toml"))
    parser.add_argument("--prompt", action="append", help="Repeat to submit several requests")
    parser.add_argument("--raw", action="store_true", help="Use raw completion instead of the checkpoint chat template")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    with args.config.open("rb") as file:
        config = EngineConfig(**tomllib.load(file)["engine"])
    params = SamplingParams(
        max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, seed=args.seed
    )
    tokenizer = Qwen2Tokenizer(args.model)
    runner = Qwen2Runner.from_directory(args.model, device=args.device, dtype=getattr(torch, args.dtype))
    engine = Engine(runner, config)
    try:
        for index, prompt in enumerate(args.prompt or ["Explain grouped-query attention in two sentences."]):
            ids = tokenizer.encode(prompt) if args.raw else tokenizer.encode_chat([{"role": "user", "content": prompt}])
            engine.add_request(str(index), ids, params)
        for output in engine.run():
            if output.finished:
                print(
                    json.dumps(
                        {
                            "request_id": output.request_id,
                            "text": tokenizer.decode(output.output_token_ids),
                            "token_ids": output.output_token_ids,
                            "finish_reason": output.finish_reason,
                        },
                        ensure_ascii=False,
                    )
                )
        print(
            json.dumps(
                {
                    "metrics": asdict(engine.metrics),
                    "device": str(runner.model.device),
                    "dtype": str(runner.model.dtype),
                    "remaining_cache_bytes": runner.cache_bytes,
                }
            )
        )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
