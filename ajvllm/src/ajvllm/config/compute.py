"""Execution backend selection, independent of KV ownership and scheduling."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeConfig:
    backend: str = "auto"

    def __post_init__(self):
        if self.backend not in ("auto", "eager", "triton"):
            raise ValueError("compute backend must be auto, eager or triton")
