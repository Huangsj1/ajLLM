"""Optional, checkpoint-specific pre-RoPE first-layer QKV lookup."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FirstLayerCacheConfig:
    enabled: bool = False
    directory: str = "cache/first_layer_qkv"

    def __post_init__(self):
        if type(self.enabled) is not bool or not isinstance(self.directory, str) or not self.directory:
            raise ValueError("first_layer_cache requires a boolean enabled and a nonempty directory")
