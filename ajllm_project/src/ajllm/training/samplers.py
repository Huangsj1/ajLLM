"""Deterministic samplers that resume at a sample index without loading skipped data."""

from __future__ import annotations

from collections.abc import Sized
from itertools import islice

import torch
from torch.utils.data import RandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler


class ResumableRandomSampler(Sampler[int]):
    """Epoch-seeded random permutation with an efficient resume offset.

    The offset is applied to sampler indices before ``DataLoader`` requests an
    item, avoiding JSONL reads and tokenization for batches already completed.
    """

    def __init__(self, data_source: Sized, seed: int) -> None:
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0
        self.start_index = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_start_index(self, start_index: int) -> None:
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        self.start_index = min(start_index, len(self.data_source))

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        sampler = RandomSampler(self.data_source, generator=generator)
        return islice(iter(sampler), self.start_index, None)

    def __len__(self) -> int:
        return len(self.data_source) - self.start_index


class ResumableDistributedSampler(DistributedSampler):
    """``DistributedSampler`` with a per-rank offset for checkpoint recovery."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.start_index = 0

    def set_start_index(self, start_index: int) -> None:
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        self.start_index = min(start_index, super().__len__())

    def __iter__(self):
        return islice(super().__iter__(), self.start_index, None)

    def __len__(self) -> int:
        return super().__len__() - self.start_index
