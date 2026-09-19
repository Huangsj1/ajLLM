"""A future Qwen tokenizer adapter lives outside the engine and scheduler."""

from collections.abc import Sequence
from typing import Protocol


class Tokenizer(Protocol):
    def encode(self, text: str) -> Sequence[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...
