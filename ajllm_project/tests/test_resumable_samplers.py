"""Index-level resume coverage for deterministic training samplers."""

from __future__ import annotations

from ajllm.training.samplers import ResumableDistributedSampler, ResumableRandomSampler


def test_resumable_random_sampler_reuses_epoch_order_after_an_offset() -> None:
    dataset = list(range(31))
    sampler = ResumableRandomSampler(dataset, seed=17)
    sampler.set_epoch(3)
    complete_order = list(sampler)
    sampler.set_start_index(11)
    assert list(sampler) == complete_order[11:]
    assert len(sampler) == len(dataset) - 11


def test_resumable_distributed_sampler_reuses_rank_local_order_after_an_offset() -> None:
    dataset = list(range(31))
    sampler = ResumableDistributedSampler(dataset, num_replicas=2, rank=1, shuffle=True, seed=17)
    sampler.set_epoch(3)
    complete_order = list(sampler)
    sampler.set_start_index(7)
    assert list(sampler) == complete_order[7:]
    assert len(sampler) == len(complete_order) - 7
