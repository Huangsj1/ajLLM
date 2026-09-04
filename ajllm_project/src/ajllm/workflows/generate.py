"""Prompt generation from a portable pre-training checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ajllm.modeling import ModelConfig, build_model
from ajllm.tokenization import MiniMindTokenizer


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    shifted = logits - torch.amax(logits)
    exponentials = torch.exp(shifted)
    return exponentials / torch.sum(exponentials)


def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> int:
    """Sample one ID with primitive tensor operations and optional filtering."""
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    filtered = logits.float() / temperature
    if top_k > 0:
        cutoff = torch.topk(filtered, min(top_k, filtered.numel())).values[-1]
        filtered = filtered.masked_fill(filtered < cutoff, float("-inf"))
    if 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        probabilities = _softmax(sorted_logits)
        remove = torch.cumsum(probabilities, dim=-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        filtered = filtered.scatter(0, sorted_indices, sorted_logits.masked_fill(remove, float("-inf")))
    probabilities = _softmax(filtered)
    return int(torch.multinomial(probabilities, 1).item())


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer: MiniMindTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
) -> str:
    """Generate continuation tokens until EOS or the configured length limit."""
    context_length = model.config.context_length
    token_ids = [tokenizer.bos_token_id, *tokenizer.encode(prompt)][-context_length:]
    generated_ids: list[int] = []
    model.eval()
    for _ in range(max_new_tokens):
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        token_id = _sample(model(input_ids)[0, -1], temperature, top_k, top_p)
        if token_id == tokenizer.eos_token_id:
            break
        generated_ids.append(token_id)
        token_ids = [*token_ids, token_id][-context_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from an ajLLM pre-training checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokenizer", default="assets/tokenizers/minimind")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    model_config = ModelConfig(**checkpoint["metadata"]["model_config"])
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    torch.manual_seed(args.seed)
    tokenizer = MiniMindTokenizer.from_pretrained(args.tokenizer)
    print(
        generate(
            model,
            tokenizer,
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            device,
        )
    )


if __name__ == "__main__":
    main()
