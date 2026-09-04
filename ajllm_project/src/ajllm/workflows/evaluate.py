"""Evaluate a pre-training checkpoint on JSONL text records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ajllm.datasets import PretrainDataset
from ajllm.modeling import ModelConfig, build_model
from ajllm.tokenization import MiniMindTokenizer
from ajllm.training.evaluation import evaluate_causal_lm


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_path: str | Path,
    tokenizer_path: str | Path,
    batch_size: int,
    device: str = "auto",
    max_batches: int | None = None,
) -> dict[str, float]:
    """Load a portable dense or MoE checkpoint and calculate validation losses."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**checkpoint["metadata"]["model_config"])
    target_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
    model = build_model(model_config).to(target_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    tokenizer = MiniMindTokenizer.from_pretrained(tokenizer_path)
    dataset = PretrainDataset(
        data_path,
        tokenizer,
        model_config.context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    metrics = evaluate_causal_lm(
        model,
        DataLoader(dataset, batch_size=batch_size, shuffle=False),
        target_device,
        mixed_precision=None,
        max_batches=max_batches,
    )
    return {"checkpoint": str(checkpoint_path), **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an ajLLM pre-training checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--tokenizer", default="assets/tokenizers/minimind")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = evaluate_checkpoint(
        args.checkpoint, args.data_path, args.tokenizer, args.batch_size, args.device, args.max_batches
    )
    output_path = Path(args.output) if args.output else Path(args.checkpoint).with_suffix(".evaluation.json")
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
