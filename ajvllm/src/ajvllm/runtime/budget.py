"""Startup memory planning; scheduling limits remain fixed during execution."""

from dataclasses import asdict, dataclass

import torch


@dataclass
class BudgetStats:
    total_bytes: int
    target_bytes: int
    baseline_bytes: int
    model_bytes: int
    safety_bytes: int       # 512 MB
    graph_reserve_bytes: int
    weight_bytes: int = 0
    buffer_bytes: int = 0
    non_torch_growth_bytes: int = 0
    measured_workspace_bytes: int = 0
    workspace_reserve_bytes: int = 0
    profile_peak_bytes: int = 0
    temporary_pool_bytes: int = 0
    workspace_bytes: int = 0
    non_kv_peak_bytes: int = 0
    pool_bytes: int = 0
    num_blocks: int = 0
    token_budget: int = 0
    max_num_seqs: int = 0


class MemoryBudget:
    """Use measured non-KV demand to size a fixed page pool, never resize budgets."""

    def __init__(self, model, config, *, gpu_memory_utilization=0.7, safety_bytes=512 * 1024**2, graph_reserve_bytes=0):
        if not 0 < gpu_memory_utilization < 1 or safety_bytes < 0:
            raise ValueError("invalid GPU memory utilization or safety reserve")
        self.device = model.device
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(self.device)
        baseline = torch.cuda.memory_allocated(self.device)
        self.available_bytes = baseline + free
        self.initial_free = free
        self.initial_reserved = torch.cuda.memory_reserved(self.device)
        model_bytes = sum(t.numel() * t.element_size() for t in (*model.parameters(), *model.buffers()))
        self.stats = BudgetStats(
            total,
            int(total * gpu_memory_utilization),
            baseline,
            model_bytes,
            safety_bytes,
            graph_reserve_bytes,
            weight_bytes=sum(t.numel() * t.element_size() for t in model.parameters()),
            buffer_bytes=sum(t.numel() * t.element_size() for t in model.buffers()),
            token_budget=config.max_num_batched_tokens,
            max_num_seqs=config.max_num_seqs,
        )

    def check_probe(self, temporary_pool_bytes, workspace_reserve):
        required = self.stats.baseline_bytes + temporary_pool_bytes + workspace_reserve + self.stats.safety_bytes
        if required > min(self.stats.target_bytes, self.available_bytes):
            raise MemoryError(
                "configured profiling workload cannot fit; lower the fixed token/sequence limits "
                "or increase gpu_memory_utilization"
            )

    def record_profile(self, *, profile_peak, temporary_pool_bytes, workspace_reserve):
        s = self.stats
        s.profile_peak_bytes = profile_peak
        s.temporary_pool_bytes = temporary_pool_bytes
        s.measured_workspace_bytes = max(0, profile_peak - temporary_pool_bytes - s.baseline_bytes)
        s.workspace_reserve_bytes = workspace_reserve
        s.workspace_bytes = max(workspace_reserve, s.measured_workspace_bytes)
        s.non_kv_peak_bytes = s.baseline_bytes + s.workspace_bytes
        # The graph allowance is separate: the temporary runner never captures graphs.
        free, _ = torch.cuda.mem_get_info(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        s.non_torch_growth_bytes = max(0, self.initial_free - free - (reserved - self.initial_reserved))
        s.non_kv_peak_bytes += s.non_torch_growth_bytes
        available = reserved + free
        capacity = min(s.target_bytes, self.available_bytes, available)
        if s.non_kv_peak_bytes + s.graph_reserve_bytes + s.safety_bytes > capacity:
            raise MemoryError("measured non-KV demand exceeds the target; adjust fixed budgets or GPU utilization")
        return capacity

    def resolve(
        self,
        *,
        profile_peak,
        temporary_pool_bytes,
        workspace_reserve,
        block_bytes,
        minimum_blocks,
        explicit_blocks=None,
    ):
        capacity = self.record_profile(
            profile_peak=profile_peak, temporary_pool_bytes=temporary_pool_bytes, workspace_reserve=workspace_reserve
        )
        s = self.stats
        remaining = capacity - s.non_kv_peak_bytes - s.graph_reserve_bytes - s.safety_bytes
        blocks = remaining // block_bytes
        if explicit_blocks is not None:
            if explicit_blocks > blocks:
                raise MemoryError("explicit KV pool exceeds profiled memory capacity")
            blocks = explicit_blocks
        if blocks < minimum_blocks:
            raise MemoryError(
                "remaining GPU memory cannot fit one maximum-context request; lower fixed budgets "
                "or max_model_len, or increase gpu_memory_utilization"
            )
        s.num_blocks = blocks
        s.pool_bytes = blocks * block_bytes
        return blocks

    def snapshot(self):
        return asdict(self.stats)
