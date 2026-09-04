"""Loss helpers for causal language modelling."""

from __future__ import annotations

import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Stable causal cross entropy from log-sum-exp, ignoring ``-100`` labels."""
    vocabulary_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocabulary_size)
    flat_targets = targets.reshape(-1)
    valid = flat_targets != -100
    if not torch.any(valid):
        return flat_logits.sum() * 0.0
    selected_logits = flat_logits[valid]
    selected_targets = flat_targets[valid]
    maximum = torch.amax(selected_logits, dim=-1, keepdim=True)
    shifted = selected_logits - maximum
    log_normalizer = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    target_logits = shifted[torch.arange(selected_targets.numel(), device=logits.device), selected_targets]
    return torch.mean(log_normalizer - target_logits)
