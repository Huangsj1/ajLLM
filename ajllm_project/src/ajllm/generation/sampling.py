"""Sampling transformations for next-token distributions."""

from __future__ import annotations

import torch


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float | None) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probabilities = torch.softmax(logits / temperature, dim=-1)
    if top_p is not None and top_p < 1.0:
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative - sorted_probabilities >= top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
        sorted_probabilities /= sorted_probabilities.sum(dim=-1, keepdim=True)
        sampled_sorted = torch.multinomial(sorted_probabilities, 1)
        return torch.gather(sorted_indices, -1, sampled_sorted)
    return torch.multinomial(probabilities, 1)
