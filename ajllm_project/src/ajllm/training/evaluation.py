"""Causal-LM evaluation shared by periodic validation and checkpoint analysis."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from ajllm.training.distributed import FullyShardedDataParallel
from ajllm.training.losses import cross_entropy


def _distributed_sum(values: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


@torch.no_grad()
def evaluate_causal_lm(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    mixed_precision: str | None,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Return token-weighted LM, router, total loss, and perplexity."""
    was_training = model.training
    model.eval()
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    enabled = device.type == "cuda" and mixed_precision in {"bf16", "fp16"}
    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            logits = model(input_ids)
            base_model = model.module if isinstance(model, FullyShardedDataParallel) else model
            lm_loss = cross_entropy(logits, labels, base_model.config.use_cuda_kernels)
            auxiliary_loss = base_model.auxiliary_loss()
        token_count = (labels != -100).sum()
        totals[0] += lm_loss.detach().double() * token_count
        totals[1] += auxiliary_loss.detach().double() * token_count
        totals[2] += (lm_loss.detach().double() + auxiliary_loss.detach().double()) * token_count
        totals[3] += token_count
    _distributed_sum(totals)
    if was_training:
        model.train()
    if totals[3] == 0:
        raise ValueError("Evaluation data contains no non-padding target tokens")
    lm_loss, auxiliary_loss, total_loss = (totals[:3] / totals[3]).tolist()
    return {
        "lm_loss": lm_loss,
        "auxiliary_loss": auxiliary_loss,
        "total_loss": total_loss,
        "perplexity": math.exp(min(lm_loss, 50.0)),
        "evaluated_tokens": int(totals[3].item()),
    }
