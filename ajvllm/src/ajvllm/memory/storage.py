"""Fixed CUDA KV pages and eager gather/scatter; kernels can replace these operations."""

from dataclasses import dataclass

import torch


@dataclass
class PagedBatch:
    storage: "PagedKVStorage"
    block_tables: torch.Tensor
    slot_mapping: torch.Tensor      # slot index in the storage tensor for each request token, shape: (num_requests, max_context_len)
    read_slots: torch.Tensor
    valid: torch.Tensor

    def update(self, layer: int, key: torch.Tensor, value: torch.Tensor):
        keys, values = self.storage.layer(layer)
        keys.index_copy_(0, self.slot_mapping, key)
        values.index_copy_(0, self.slot_mapping, value)
        shape = (*self.read_slots.shape, key.shape[1], key.shape[2])
        gathered = []
        for tensor in (keys, values):
            dense = tensor.index_select(0, self.read_slots.flatten()).view(shape)
            # Recycled/uninitialized padding must never leak NaNs or stale values.
            dense.masked_fill_(~self.valid[:, :, None, None], 0)
            gathered.append(dense.permute(0, 2, 1, 3))
        return tuple(gathered)


class PagedKVStorage:
    def __init__(self, layers, blocks, block_size, kv_heads, head_dim, *, device, dtype):
        self.block_size = block_size
        # One allocation holds KV blocks across all layers, heads and requests.
        self.tensor = torch.empty((layers, 2, blocks, block_size, kv_heads, head_dim), device=device, dtype=dtype)
        self.last_use: torch.cuda.Event | None = None

    @property
    def nbytes(self):
        return self.tensor.numel() * self.tensor.element_size()

    def layer(self, index):
        pair = self.tensor[index]
        # (k, v) each of shape (num_blocks, block_size, num_kv_heads, head_dim)
        return tuple(tensor.view(-1, *tensor.shape[-2:]) for tensor in pair)

    def wait(self):
        if self.last_use is not None:
            torch.cuda.current_stream(self.tensor.device).wait_event(self.last_use)

    def record(self):
        self.last_use = torch.cuda.Event()
        self.last_use.record(torch.cuda.current_stream(self.tensor.device))

    def copy_block(self, source, target):
        self.wait()
        self.tensor[:, :, target].copy_(self.tensor[:, :, source])
        self.record()
