"""Independent graph replay and weight-only quantization settings."""

from dataclasses import dataclass

from ajvllm.config.engine import require_int


@dataclass(frozen=True)
class GraphConfig:
    enabled: bool = False
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8)
    max_graphs: int = 8
    memory_limit_mb: int = 128

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("graph enabled must be a boolean")
        sizes = tuple(self.batch_sizes)
        if not sizes:
            raise ValueError("graph batch_sizes must not be empty")
        for size in sizes:
            require_int("graph batch size", size, 1)
        if sizes != tuple(sorted(set(sizes))):
            raise ValueError("graph batch_sizes must be sorted and unique")
        object.__setattr__(self, "batch_sizes", sizes)
        require_int("max_graphs", self.max_graphs, 1)
        require_int("memory_limit_mb", self.memory_limit_mb, 1)

    @property
    def reserve_bytes(self):
        return self.memory_limit_mb * 1024**2 if self.enabled else 0


@dataclass(frozen=True)
class QuantizationConfig:
    mode: str = "none"

    def __post_init__(self):
        if self.mode not in ("none", "w8a16"):
            raise ValueError("quantization mode must be none or w8a16")
