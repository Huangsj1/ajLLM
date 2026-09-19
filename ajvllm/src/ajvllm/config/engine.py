"""Configuration of the model-independent control plane."""

from dataclasses import dataclass


def require_int(name: str, value: int, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class EngineConfig:
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 128
    max_model_len: int = 4096
    enable_chunked_prefill: bool = True
    max_prefill_chunk_size: int | None = None
    max_prefill_tokens_per_step: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_num_seqs", "max_num_batched_tokens", "max_model_len"):
            require_int(name, getattr(self, name), 1)
        if type(self.enable_chunked_prefill) is not bool:
            raise ValueError("enable_chunked_prefill must be a boolean")
        if self.max_prefill_chunk_size is not None:
            require_int("max_prefill_chunk_size", self.max_prefill_chunk_size, 1)
            if not self.enable_chunked_prefill:
                raise ValueError("max_prefill_chunk_size requires chunked prefill")
        if self.max_prefill_tokens_per_step is not None:
            require_int("max_prefill_tokens_per_step", self.max_prefill_tokens_per_step, 1)
            if not self.enable_chunked_prefill and self.max_prefill_tokens_per_step < self.max_model_len:
                raise ValueError("unchunked prefill requires prefill budget >= max_model_len")
        if not self.enable_chunked_prefill and self.max_num_batched_tokens < self.max_model_len:
            raise ValueError("unchunked prefill requires token budget >= max_model_len")
