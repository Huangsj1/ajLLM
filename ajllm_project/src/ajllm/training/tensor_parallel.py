"""Compatibility imports; TP implementation moved to :mod:`training.parallel.tensor_parallel`."""

from ajllm.training.parallel.tensor_parallel import TensorParallel, TensorParallelLinear, tensor_parallelize

__all__ = ["TensorParallel", "TensorParallelLinear", "tensor_parallelize"]
