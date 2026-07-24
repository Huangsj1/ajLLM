"""Autoregressive token generation."""

from __future__ import annotations

import torch

from ajllm.generation.sampling import sample_next_token


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    model.eval()
    generated = prompt_tokens.reshape(1, -1)
    for _ in range(max_new_tokens):
        model_input = generated[:, -model.context_length :]
        logits = model(model_input)[:, -1, :]
        next_token = sample_next_token(logits, temperature, top_p)
        generated = torch.cat((generated, next_token), dim=1)
        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break
    return generated.squeeze(0)
