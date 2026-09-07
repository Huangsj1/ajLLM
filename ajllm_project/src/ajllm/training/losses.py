"""Loss helpers for causal language modelling."""

from __future__ import annotations

import torch

from ajllm.modeling.cuda_kernels import fused_cross_entropy


def torch_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Stable causal cross entropy from log-sum-exp, ignoring ``-100`` labels."""
    vocabulary_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocabulary_size)
    flat_targets = targets.reshape(-1)
    valid = flat_targets != -100
    if not torch.any(valid):
        return flat_logits.sum() * 0.0
    # Keep the expensive model projections in BF16/FP16, but reduce this
    # log-sum-exp loss in FP32 for numerical stability.
    selected_logits = flat_logits[valid].float()
    selected_targets = flat_targets[valid]
    maximum = torch.amax(selected_logits, dim=-1, keepdim=True)
    shifted = selected_logits - maximum
    log_normalizer = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    target_logits = shifted[torch.arange(selected_targets.numel(), device=logits.device), selected_targets]
    return torch.mean(log_normalizer - target_logits)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor, use_cuda_kernels: bool = True) -> torch.Tensor:
    """Choose the fused Triton or explicit Torch causal-loss equation."""
    return fused_cross_entropy(logits, targets) if use_cuda_kernels else torch_cross_entropy(logits, targets)
