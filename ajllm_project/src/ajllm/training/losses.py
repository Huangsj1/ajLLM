"""Numerically stable training losses."""

from __future__ import annotations

import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return mean cross-entropy over arbitrary leading dimensions."""

    vocabulary_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocabulary_size)
    flat_targets = targets.reshape(-1)
    maximum = torch.amax(flat_logits, dim=-1, keepdim=True)
    shifted = flat_logits - maximum
    log_normalizer = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    target_logits = shifted[torch.arange(flat_targets.numel(), device=logits.device), flat_targets]
    return torch.mean(log_normalizer - target_logits)
