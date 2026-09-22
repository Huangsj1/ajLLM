"""Contiguous baseline storage, retained as a numerical/performance reference."""

import torch


def update_layer(batch, layer_index, key, value):
    keys = key.new_zeros(batch.num_requests, key.shape[1], batch.max_context_len, key.shape[2])
    values = torch.zeros_like(keys)
    for row, cache in enumerate(batch.caches):
        if cache is not None:
            old_k, old_v = cache[layer_index]
            keys[row, :, : old_k.shape[1]] = old_k
            values[row, :, : old_v.shape[1]] = old_v
    keys[batch.sequence_ids, :, batch.positions] = key
    values[batch.sequence_ids, :, batch.positions] = value
    present = tuple(
        (keys[row, :, :length].clone().contiguous(), values[row, :, :length].clone().contiguous())
        for row, length in enumerate(batch.context_lengths)
    )
    return keys, values, present
