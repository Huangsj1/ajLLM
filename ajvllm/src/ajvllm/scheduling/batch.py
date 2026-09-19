"""Execution plans contain logical positions, never physical memory addresses."""

from dataclasses import dataclass
from enum import StrEnum


class Phase(StrEnum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class ScheduledRequest:
    request_id: str
    token_ids: tuple[int, ...]  # tokens to be processed in this batch
    start_pos: int  # the start position of the first token in the request's whole sequence
    phase: Phase
    do_sample: bool  # true if token_ids is the last chunk of the request's input sequence

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def end_pos(self) -> int:
        return self.start_pos + self.num_tokens


@dataclass(frozen=True)
class SchedulerOutput:
    requests: tuple[ScheduledRequest, ...] = ()

    @property
    def num_scheduled_tokens(self) -> int:
        return sum(item.num_tokens for item in self.requests)
