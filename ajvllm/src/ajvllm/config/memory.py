"""KV storage configuration, separate from token scheduling."""

from dataclasses import dataclass

from ajvllm.config.engine import require_int


@dataclass(frozen=True)
class MemoryConfig:
    backend: str = "paged"
    block_size: int = 16
    num_blocks: int | None = None
    enable_prefix_cache: bool = True
    cache_namespace: str = "local"

    def __post_init__(self):
        if self.backend not in ("paged", "contiguous"):
            raise ValueError("memory backend must be paged or contiguous")
        require_int("block_size", self.block_size, 1)
        if self.num_blocks is not None:
            require_int("num_blocks", self.num_blocks, 1)
        if type(self.enable_prefix_cache) is not bool or not isinstance(self.cache_namespace, str):
            raise ValueError("invalid prefix cache configuration")
