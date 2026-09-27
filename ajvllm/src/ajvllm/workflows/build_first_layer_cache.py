"""Precompute all vocabulary rows without loading a full model or an engine."""

import argparse

import torch

from ajvllm.modeling.qwen2.first_layer_cache import FirstLayerFrontEnd, build_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--directory", default="cache/first_layer_qkv")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--backend", choices=["triton", "eager"], default="triton")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-tokens", type=int, default=4096)
    args = parser.parse_args()
    module, metadata = FirstLayerFrontEnd.from_directory(
        args.model, device=args.device, dtype=getattr(torch, args.dtype), optimized=args.backend == "triton"
    )
    path = build_cache(module, metadata, args.directory, chunk_tokens=args.chunk_tokens)
    print(f"Cache: {path} ({path.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
