"""Small CPU metadata uploads without synchronizing the CUDA execution stream."""

import torch


def upload(values, *, device, dtype=torch.long):
    # Pinned allocations stay alive until the asynchronous copy completes through
    # PyTorch's caching host allocator. Never mutate a submitted staging tensor.
    return torch.tensor(values, dtype=dtype, pin_memory=True).to(device, non_blocking=True)
