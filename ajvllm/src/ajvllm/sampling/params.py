"""Immutable sampling policy, independent of any tensor library."""

import math
from dataclasses import dataclass

from ajvllm.config import require_int


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 256
    min_tokens: int = 0
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        for name in ("max_tokens", "min_tokens", "top_k"):
            require_int(name, getattr(self, name))
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens must not exceed max_tokens")
        for name in ("temperature", "top_p", "repetition_penalty", "presence_penalty", "frequency_penalty"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.temperature < 0 or not 0 < self.top_p <= 1 or self.repetition_penalty <= 0:
            raise ValueError("require temperature >= 0, 0 < top_p <= 1, repetition_penalty > 0")
        if self.seed is not None:
            require_int("seed", self.seed)
        if type(self.ignore_eos) is not bool:
            raise ValueError("ignore_eos must be a boolean")
        object.__setattr__(self, "stop_token_ids", tuple(self.stop_token_ids))
        for token in self.stop_token_ids:
            require_int("stop token ID", token)

    def effective_stop_ids(self, eos_token_ids: tuple[int, ...]) -> frozenset[int]:
        return frozenset(self.stop_token_ids) | (frozenset() if self.ignore_eos else frozenset(eos_token_ids))
