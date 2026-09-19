"""The only interface the control plane needs from an execution backend."""

from collections.abc import Mapping, Sequence
from typing import Protocol

from ajvllm.scheduling.batch import SchedulerOutput


class ModelRunner(Protocol):
    vocab_size: int
    eos_token_ids: tuple[int, ...]

    def execute(self, batch: SchedulerOutput) -> Mapping[str, Sequence[float]]:
        """Consume every slice; return one vocabulary row per do_sample entry only.

        Positions must continue the previously committed prefix for each request.
        A failed batch may have partially advanced backend state and must be released.
        """
        ...

    def release(self, request_id: str) -> None:
        """Idempotently discard per-request state. Must not raise, including for unknown IDs."""
        ...
