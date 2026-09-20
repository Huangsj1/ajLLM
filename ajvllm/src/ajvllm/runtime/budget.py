"""Conservative memory admission plus gradual input-token budget growth."""

from dataclasses import asdict, dataclass, replace

import torch

from ajvllm.config import EngineConfig
from ajvllm.execution.batch import ModelBatch
from ajvllm.execution.qwen2 import Qwen2Runner


@dataclass
class BudgetStats:
    total_bytes: int
    target_bytes: int
    baseline_bytes: int                 # memory used by model weights and static buffers
    profile_peak_bytes: int = 0         # peak memory observed during warmup
    observed_peak_bytes: int = 0        # peak memory observed during runtime
    token_budget: int = 1  # current token budget for the next step
    token_ceiling: int = 1  # maximum token budget allowed by the memory target
    max_num_seqs: int = 1  # maximum number of sequences allowed by the memory target


class MemoryBudget:
    """A process memory ceiling, not a target GPU compute utilization.

    Contiguous caches and padded attention need a conservative worst-case estimate.
    This is replaced by block-level capacity accounting when paged KV is introduced.
    """

    def __init__(
        self,
        runner: Qwen2Runner,
        config: EngineConfig,
        *,
        gpu_memory_utilization: float = 0.5,
        initial_token_budget: int = 4096,
        safety_bytes: int = 512 * 1024**2,  # 512 MB
        growth_interval: int = 8,
    ):
        if not 0 < gpu_memory_utilization < 1 or initial_token_budget < 1 or growth_interval < 1 or safety_bytes < 0:
            raise ValueError("invalid memory utilization, initial budget, safety reserve, or growth interval")
        if not config.enable_chunked_prefill:
            raise ValueError("adaptive memory budgeting requires chunked prefill")
        self.runner = runner
        self.device = runner.model.device
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
        self.stats.token_budget = min(initial_token_budget, ceiling)
        # use maximum admitted batch to warm up the model and measure peak memory usage
        self._warmup()

    def estimate(self, slots: int, tokens: int) -> int:
        """Estimate the memory usage of a given number of sequences and tokens."""
        model = self.runner.model
        cfg = model.config
        size = model.model.embed_tokens.weight.element_size()
        context = self.config.max_model_len
        prefill_tokens = min(tokens, self.config.max_prefill_tokens_per_step or tokens)
        query = min(prefill_tokens, self.config.max_prefill_chunk_size or prefill_tokens)
        kv_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * size
        # Old + replacement caches plus packing; reserve all admitted requests to their context limit.
        kv = 3 * slots * context * kv_per_token
        # Retain the earlier expanded-KV-sized reserve for backend workspace.
        # Grouped eager GQA no longer allocates repeated K/V, but removing this
        # safety margin should follow capacity profiling on longer workloads.
        workspace_reserve = 2 * slots * context * cfg.hidden_size * size
        # Mixed batches pad every request to the largest query/context dimensions,
        # including decode rows. Reserve that workspace even when prefill is capped.
        attention = min(slots, tokens) * cfg.num_attention_heads * query * context * (3 * size + 4)
        activations = tokens * (8 * cfg.hidden_size + 4 * cfg.intermediate_size) * size
        # General CUDA sampling keeps FP32 scores/probabilities/CDFs, history
        # buffers, int64 sorted IDs and sorting workspace on device.
        logits = slots * cfg.vocab_size * 64
        return self.stats.baseline_bytes + kv + workspace_reserve + attention + activations + logits

    @torch.inference_mode()
    def _warmup(self) -> None:
        '''Warm up the model with a single forward pass of the maximum admitted batch.
        If the warmup fits within the memory target, we can use the current token budget.
        If the warmup exceeds the memory target, halve the token budget and try again, until we reach the minimum of 1 token.'''
        while True:
            try:
                budget = min(
                    self.stats.token_budget, self.config.max_prefill_tokens_per_step or self.stats.token_budget
                )
                slots = min(self.config.max_num_seqs, budget)
                chunk = min(self.config.max_model_len, self.config.max_prefill_chunk_size or budget)
                # Warmup with a single forward pass of the maximum admitted batch.
                sequences = [[0] * min(chunk, budget // slots + (row < budget % slots)) for row in range(slots)]
                torch.cuda.reset_peak_memory_stats(self.device)
                # 1.prefill
                output = self.runner.model(ModelBatch.build(sequences, [None] * slots, self.device, range(slots)))
                # if prefill requests length is less than max_model_len, run a decode step
                if max(map(len, sequences)) < self.config.max_model_len:
                    caches = output.caches
                    del output
                    # 2.decode
                    output = self.runner.model(ModelBatch.build([[0]] * slots, caches, self.device, range(slots)))
                    del caches
                torch.cuda.synchronize(self.device)
                self.stats.profile_peak_bytes = torch.cuda.max_memory_allocated(self.device)
                del output
                # If the warmup fits within the memory target, we can use the current token budget.
                if self.stats.profile_peak_bytes <= self.stats.target_bytes:
                    return
            # if OOM, halve the token budget and try again, until we reach the minimum of 1 token
            except torch.cuda.OutOfMemoryError:
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
