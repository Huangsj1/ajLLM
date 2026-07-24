"""Loss and perplexity evaluation over random held-out batches."""

from __future__ import annotations

import math

import numpy as np
import torch

from ajllm.training.data import TokenDataset
from ajllm.training.losses import cross_entropy


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataset: TokenDataset,
    batches: int,
    batch_size: int,
    context_length: int,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    model.eval()
    generator = np.random.default_rng(seed)
    losses = []
    for _ in range(batches):
        inputs, targets = dataset.batch(batch_size, context_length, device, generator)
        losses.append(float(cross_entropy(model(inputs), targets).item()))
    loss = sum(losses) / len(losses)
    return {"loss": loss, "perplexity": math.exp(min(loss, 50.0)), "batches": batches}
