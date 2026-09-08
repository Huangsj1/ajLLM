"""Compatibility imports; FSDP implementation moved to :mod:`training.parallel.fsdp`."""

from ajllm.training.parallel.fsdp import FullyShardedDataParallel, destroy_distributed, initialize_distributed

__all__ = ["FullyShardedDataParallel", "destroy_distributed", "initialize_distributed"]
