"""Internal request state and detached, immutable streaming events."""

from dataclasses import dataclass, field
from enum import StrEnum

import torch

from ajvllm.sampling.params import SamplingParams


class RequestStatus(StrEnum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class Request:
    # the initial request has first 4 fields
    request_id: str
    prompt_token_ids: tuple[int, ...]
    sampling_params: SamplingParams
    arrival_time: float
    output_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    status: RequestStatus = RequestStatus.WAITING
    first_token_time: float | None = None
    finish_time: float | None = None
    rng: torch.Generator | None = field(default=None, repr=False)

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def token_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids + tuple(self.output_token_ids)

    @property
    def is_prefill(self) -> bool:
        return self.num_computed_tokens < len(self.prompt_token_ids)


@dataclass(frozen=True)
class RequestOutput:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    new_token_ids: tuple[int, ...]
    status: RequestStatus
    finish_reason: FinishReason | None
    stop_token_id: int | None
    logprob: float | None
    arrival_time: float
    first_token_time: float | None
    finish_time: float | None
    error: str | None = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None
