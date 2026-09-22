"""The only interface the control plane needs from an execution backend."""

from collections.abc import Mapping
from typing import Protocol

import torch

from ajvllm.memory.manager import KVCacheManager
from ajvllm.scheduling.batch import SchedulerOutput


class ModelRunner(Protocol):
    kv_cache: KVCacheManager | None
    vocab_size: int
    eos_token_ids: tuple[int, ...]

    def execute(self, batch: SchedulerOutput) -> Mapping[str, torch.Tensor]:
        """Consume every slice; return CUDA vocabulary rows for sampling-ready requests.

        Positions must continue the previously committed prefix for each request.
        A failed batch may have partially advanced backend state and must be released.
        """
        ...

    def release(self, request_id: str) -> None:
        """Idempotently discard per-request state. Must not raise, including for unknown IDs."""
        ...
