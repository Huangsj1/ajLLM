"""Memory-mapped uint32 token datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class TokenDataset:
    """Read random contiguous language-model batches without loading the corpus."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.stat().st_size % 4 != 0:
            raise ValueError(f"Token file size is not divisible by four bytes: {self.path}")
        self.tokens = np.memmap(self.path, dtype=np.uint32, mode="r")

    def __len__(self) -> int:
        return len(self.tokens)

    def batch(
        self,
        batch_size: int,
        context_length: int,
        device: torch.device,
        generator: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        maximum_start = len(self.tokens) - context_length - 1
        if maximum_start < 0:
            raise ValueError(f"Dataset {self.path} has {len(self.tokens)} tokens, fewer than context_length + 1")
        starts = generator.integers(0, maximum_start + 1, size=batch_size)
        inputs = np.stack([np.array(self.tokens[start : start + context_length], dtype=np.int64) for start in starts])
        targets = np.stack(
            [np.array(self.tokens[start + 1 : start + context_length + 1], dtype=np.int64) for start in starts]
        )
        return torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device)
