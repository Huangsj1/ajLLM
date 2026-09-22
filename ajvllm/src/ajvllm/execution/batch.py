"""Packed token inputs and ragged sequence metadata shared by all decoder layers."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate

import torch

from ajvllm.memory.storage import PagedBatch
from ajvllm.memory.types import KVCache  # re-export for existing reference callers


@dataclass
class ModelBatch:
    token_ids: torch.Tensor
    positions: torch.Tensor
    sequence_ids: torch.Tensor
    query_offsets: torch.Tensor
    query_lengths: tuple[int, ...]
    context_lengths: tuple[int, ...]
    caches: tuple[KVCache | None, ...]
    sample_indices: torch.Tensor
    causal_mask: torch.Tensor
    paged: PagedBatch | None = None

    @property
    def num_requests(self) -> int:
        return len(self.query_lengths)

    @property
    def max_query_len(self) -> int:
        return max(self.query_lengths)

    @property
    def max_context_len(self) -> int:
        return max(self.context_lengths)

    @classmethod
    def build(
        cls,
        sequences: Sequence[Sequence[int] | torch.Tensor],
        caches: Sequence[KVCache | None],
        device: torch.device,
        sample_requests: Sequence[int],
        *,
        starts: Sequence[int] | None = None,
    ) -> "ModelBatch":
        # all request sequences length
        lengths = tuple(len(tokens) for tokens in sequences)
        # cache[0][0] means 1st layer's k tensor, shape: (num_heads, seq_len, head_dim)
        if starts is None:
            starts = [0 if cache is None else cache[0][0].shape[1] for cache in caches]
        # contexts = cached tokens + current query tokens
        contexts = tuple(start + length for start, length in zip(starts, lengths, strict=True))
        offsets = list(accumulate(lengths))
        # packed token positions of all requests, used for RoPE and attention masking,
        #  e.g. for 3 requests with lengths [2, 3, 1] and starts [0, 5, 2], positions = [0,1, 5,6,7, 2]
        positions = torch.tensor(
            [start + i for start, length in zip(starts, lengths, strict=True) for i in range(length)], device=device
        )
        # packed token request IDs of all requests, used for packing/unpacking attention caches,
        #  e.g. for 3 requests with lengths [2, 3, 1], sequence_ids = [0,0, 1,1,1, 2]
        sequence_ids = torch.tensor([row for row, length in enumerate(lengths) for _ in range(length)], device=device)
        # packed token local query offsets of all requests
        #  e.g. for 3 requests with lengths [2, 3, 1], query_offsets = [0,1, 0,1,2, 0]
        query_offsets = torch.tensor([i for length in lengths for i in range(length)], device=device)
        # padding token positions for causal attention masking, shape = (num_requests, max_query_len)
        #  e.g. for 3 requests with lengths [2, 3, 1], query_positions = [[0,1,2], [5,6,7], [2,3,4]]
        query_positions = torch.tensor(starts, device=device)[:, None] + torch.arange(max(lengths), device=device)
        # all keys positions for causal attention masking
        #  e.g. for 3 requests with contexts [2, 8, 3], keys = [0,1,2,3,4,5,6,7]
        keys = torch.arange(max(contexts), device=device)
        # shape = (num_requests, max_query_len, max_context_len), True if key position is masked for query position
        mask = (keys[None, None, :] > query_positions[:, :, None]) | (
            keys[None, None, :] >= torch.tensor(contexts, device=device)[:, None, None]
        )
        return cls(
            # packed token IDs of all requests, shape = (sum(lengths),)
            torch.cat([torch.as_tensor(tokens, dtype=torch.long, device=device) for tokens in sequences]),
            positions,
            sequence_ids,
            query_offsets,
            lengths,
            contexts,
            tuple(caches),
            # sample indices of all requests, shape = (num_sample_requests,)
            torch.tensor([offsets[row] - 1 for row in sample_requests], dtype=torch.long, device=device),
            mask[:, None, :, :],  # shape = (num_requests, 1, max_query_len, max_context_len)
        )
