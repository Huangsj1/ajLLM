"""Explicit packed inputs for numerical reference tests."""

import torch

from ajvllm.execution.batch import ModelBatch


def forward_tokens(model, tokens, cache=None, *, logits_to_keep=0):
    batch = ModelBatch.build([tokens], [cache], model.device, [])
    if logits_to_keep is not None:
        start = 0 if logits_to_keep == 0 else max(0, len(tokens) - logits_to_keep)
        batch.sample_indices = torch.arange(start, len(tokens), device=model.device)
    return model(batch)
