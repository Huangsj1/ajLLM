"""Compatibility imports; EP implementation moved to :mod:`training.parallel.expert_parallel`."""

from ajllm.training.parallel.expert_parallel import ExpertParallel, ExpertParallelMoE, expert_parallelize

__all__ = ["ExpertParallel", "ExpertParallelMoE", "expert_parallelize"]
