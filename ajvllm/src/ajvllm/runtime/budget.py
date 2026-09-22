"""Conservative memory admission plus gradual input-token budget growth."""

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

import torch

from ajvllm.config import EngineConfig


@dataclass
class BudgetStats:
    total_bytes: int
    target_bytes: int
    baseline_bytes: int  # memory used by model weights and static buffers
    profile_peak_bytes: int = 0  # peak memory observed during warmup
    observed_peak_bytes: int = 0  # peak memory observed during runtime
    token_budget: int = 1  # current token budget for the next step
    token_ceiling: int = 1  # maximum token budget allowed by the memory target
    max_num_seqs: int = 1  # maximum number of sequences allowed by the memory target


class MemoryBudget:
    """A process memory ceiling, not a target GPU compute utilization.

    Account for the selected KV backend separately from padded attention workspace.
    The scheduler enforces physical page capacity within the fixed paged pool.
    """

    def __init__(
        self,
        device,
        config: EngineConfig,
        *,
        gpu_memory_utilization: float = 0.5,
        estimate: Callable[[int, int], int],
        safety_bytes: int = 512 * 1024**2,  # 512 MB
        growth_interval: int = 8,
    ):
        if not 0 < gpu_memory_utilization < 1 or growth_interval < 1 or safety_bytes < 0:
            raise ValueError("invalid memory utilization, safety reserve, or growth interval")
        if not config.enable_chunked_prefill:
            raise ValueError("adaptive memory budgeting requires chunked prefill")
        self.device = device
        self._estimate = estimate
        self.safety_bytes = safety_bytes
        self.growth_interval = growth_interval
        self.steps = 0
        self.config = config
        free, total = torch.cuda.mem_get_info(self.device)
        baseline = torch.cuda.memory_allocated(self.device)
        target = int(min(total * gpu_memory_utilization, baseline + free)) - safety_bytes
        self.stats = BudgetStats(total, target, baseline)
        # 1. max_num_seqs: the number of sequences that can be admitted without exceeding the memory target
        slots = config.max_num_seqs
        while slots > 1 and self.estimate(slots, 1) > target:
            slots -= 1
        if self.estimate(slots, 1) > target:
            raise MemoryError("memory target cannot fit weights, one full-context request, and workspace")
        # 2. max_num_batched_tokens: tokens admitted without exceeding the memory target.
        ceiling = config.max_num_batched_tokens
        while ceiling > 1 and self.estimate(slots, ceiling) > target:
            ceiling = max(1, ceiling // 2)
        self.config = replace(config, max_num_seqs=slots, max_num_batched_tokens=ceiling)
        self.stats.max_num_seqs, self.stats.token_ceiling = slots, ceiling
        self.stats.token_budget = ceiling

    def estimate(self, slots: int, tokens: int) -> int:
        return self.stats.baseline_bytes + self._estimate(slots, tokens)

    @torch.inference_mode()
    def warmup(self, probe) -> None:
        """Profile admitted capacity, reducing the budget when the probe cannot fit."""
        while True:
            try:
                budget = min(
                    self.stats.token_budget, self.config.max_prefill_tokens_per_step or self.stats.token_budget
                )
                slots = min(self.config.max_num_seqs, budget)
                chunk = min(self.config.max_model_len, self.config.max_prefill_chunk_size or budget)
                sequences = [[0] * min(chunk, budget // slots + (row < budget % slots)) for row in range(slots)]
                torch.cuda.reset_peak_memory_stats(self.device)
                probe(sequences)
                torch.cuda.synchronize(self.device)
                self.stats.profile_peak_bytes = torch.cuda.max_memory_allocated(self.device)
                if self.stats.profile_peak_bytes <= self.stats.target_bytes:
                    return
            except (torch.cuda.OutOfMemoryError, MemoryError):
                pass
            if self.stats.token_budget == 1:
                raise MemoryError("even the minimum warmup exceeds the memory target")
            self.stats.token_budget = max(1, self.stats.token_budget // 2)
            torch.cuda.empty_cache()

    def before_step(self, engine) -> None:
        free, _ = torch.cuda.mem_get_info(self.device)
        # Include free blocks held by PyTorch, which mem_get_info counts as unavailable.
        allocated = torch.cuda.memory_allocated(self.device)
        reusable = torch.cuda.memory_reserved(self.device) - allocated
        allowed = min(self.stats.target_bytes, allocated + free + reusable - self.safety_bytes)
        while (
            self.stats.token_budget > 1 and self.estimate(self.config.max_num_seqs, self.stats.token_budget) > allowed
        ):
            self.stats.token_budget = max(1, self.stats.token_budget // 2)
        if self.estimate(self.config.max_num_seqs, self.stats.token_budget) > allowed:
            raise MemoryError("available GPU memory no longer covers the reserved request capacity")
        engine.set_token_budget(self.stats.token_budget)
        torch.cuda.reset_peak_memory_stats(self.device)

    def after_step(self, engine) -> None:
        peak = torch.cuda.max_memory_allocated(self.device)
        self.stats.observed_peak_bytes = max(self.stats.observed_peak_bytes, peak)
        self.steps += 1
        budget = self.stats.token_budget
        if peak > self.stats.target_bytes * 0.9:
            self.stats.token_budget = max(1, budget // 2)
        elif self.steps % self.growth_interval == 0 and engine.last_batch.num_scheduled_tokens >= budget:
            candidate = min(self.stats.token_ceiling, budget * 2)
            if (
                peak < self.stats.target_bytes * 0.8
                and self.estimate(self.config.max_num_seqs, candidate) <= self.stats.target_bytes
            ):
                self.stats.token_budget = candidate

    def on_oom(self) -> None:
        self.stats.token_budget = max(1, self.stats.token_budget // 2)
        torch.cuda.empty_cache()

    def snapshot(self) -> dict:
        return asdict(self.stats)
