"""Shared-budget scheduling: reserve decode tokens, then fairly chunk prefills."""

from collections import deque
from itertools import islice

from ajvllm.config import EngineConfig
from ajvllm.requests import Request, RequestStatus
from ajvllm.scheduling.batch import Phase, ScheduledRequest, SchedulerOutput


def fair_chunks(demands: list[int], budget: int) -> list[int]:
    """Capped equal shares, redistributing unused shares from short prompts.

    Remainder tokens follow candidate order. The scheduler rotates running
    prefills between steps, so a budget smaller than their count cannot starve a tail.
    """
    counts = [0] * len(demands)
    active = list(range(len(demands)))
    while active and budget:
        # average budget
        share = max(1, budget // len(active))
        remaining = []
        for index in active:
            count = min(demands[index] - counts[index], share, budget)
            counts[index] += count
            budget -= count
            if counts[index] < demands[index]:
                remaining.append(index)
        active = remaining
    return counts


def rotate_after(ids: list[str], last: str | None) -> list[str]:
    """Move the back to the front of the list, preventing starvation of tail candidates."""
    if last in ids:
        offset = ids.index(last) + 1
        return ids[offset:] + ids[:offset]
    return ids


class Scheduler:
    def __init__(self, config: EngineConfig, kv_cache=None):
        self.kv_cache = kv_cache
        self.config = config
        self.requests: dict[str, Request] = {}
        self.waiting: deque[str] = deque()
        self.running: list[str] = []
        self._last_decode_id: str | None = None
        self._last_prefill_id: str | None = None
        self.token_budget = config.max_num_batched_tokens

    def add(self, request: Request) -> None:
        self.requests[request.request_id] = request
        self.waiting.append(request.request_id)

    def remove(self, request_id: str) -> Request:
        request = self.requests.pop(request_id)
        if request_id in self.running:
            self.running.remove(request_id)
        else:
            self.waiting.remove(request_id)
        return request

    def _attach(self, rid):
        '''attach a request to the KV cache, and retain any prefix blocks'''
        if self.kv_cache is not None and rid not in self.kv_cache.states:
            request = self.requests[rid]
            request.num_computed_tokens = self.kv_cache.attach(rid, request.token_ids, request.cache_salt)

    def _preempt(self, rid):
        '''preempt a running request, releasing its kv blocks and moving it back to the waiting queue'''
        request = self.requests[rid]
        self.kv_cache.release(rid)
        self.kv_cache.preemptions += 1
        request.num_computed_tokens = 0
        request.status = RequestStatus.WAITING
        self.running.remove(rid)
        self.waiting.append(rid)

    def schedule(self) -> SchedulerOutput:
        budget = self.token_budget
        items = []
        # track requests that have been preempted in this scheduling step, to avoid re-adding them to the schedule
        preempted = set()
        decodes, prefills = [], []
        for rid in self.running:
            (prefills if self.requests[rid].is_prefill else decodes).append(rid)

        def append(rid: str, count: int, phase: Phase):
            request = self.requests[rid]
            if rid in preempted:
                return False
            self._attach(rid)
            start = request.num_computed_tokens
            if self.kv_cache is not None:
                # if cannot reserve the required blocks, preempt a running request and retry
                while not self.kv_cache.reserve(rid, start + count):
                    protected = {item.request_id for item in items} | {rid}
                    victim = next(
                        (candidate for candidate in reversed(self.running) if candidate not in protected), None
                    )
                    if victim is None:
                        if rid in self.running:
                            self._preempt(rid)
                            preempted.add(rid)
                        else:
                            self.kv_cache.release(rid)
                            request.num_computed_tokens = 0
                        return False
                    self._preempt(victim)
                    preempted.add(victim)
            items.append(
                ScheduledRequest(
                    rid, request.token_ids[start : start + count], start, phase, start + count == request.num_tokens
                )
            )
            if request.status == RequestStatus.WAITING:
                # Newly admitted candidates always follow the FIFO waiting prefix.
                self.waiting.remove(rid)
                self.running.append(rid)
                request.status = RequestStatus.RUNNING
            return True

        # Decode priority reserves both token budget and block capacity first.
        # 1. Decode
        selected_decodes = rotate_after(decodes, self._last_decode_id)[:budget]
        for rid in selected_decodes:
            if append(rid, 1, Phase.DECODE):
                self._last_decode_id = rid
        budget -= len(items)
        prefill_budget = min(budget, self.config.max_prefill_tokens_per_step or budget)
        if not prefill_budget:
            return SchedulerOutput(tuple(items))

        # 2. Prefill
        prefills = [rid for rid in prefills if rid not in preempted]
        slots = self.config.max_num_seqs - len(self.running)
        waiting = [rid for rid in self.waiting if rid not in preempted]
        if self.config.enable_chunked_prefill:
            prefills = rotate_after(prefills, self._last_prefill_id)
            # Reserve at least one token for each candidate before admitting more.
            admission_count = min(slots, max(0, prefill_budget - len(prefills)))
            candidates = prefills + list(islice(waiting, admission_count))
            # attach firstly to ensure that any prefix blocks are retained before allocating new blocks for the prefill
            for rid in candidates:
                self._attach(rid)
            demands = [
                min(
                    self.requests[rid].num_tokens - self.requests[rid].num_computed_tokens,
                    self.config.max_prefill_chunk_size or prefill_budget,
                )
                for rid in candidates
            ]
            counts = fair_chunks(demands, prefill_budget)
        else:
            candidates = prefills + list(islice(waiting, slots))
            counts = []
            for rid in candidates:
                self._attach(rid)
                count = self.requests[rid].num_tokens - self.requests[rid].num_computed_tokens
                if count > prefill_budget:
                    break  # Whole prompts only, with FIFO head-of-line blocking.
                counts.append(count)
                prefill_budget -= count
        for rid, count in zip(candidates, counts):
            if count and append(rid, count, Phase.PREFILL):
                self._last_prefill_id = rid
        if self.kv_cache is not None:
            for rid in self.waiting:
                self.kv_cache.release(rid)
                self.requests[rid].num_computed_tokens = 0
        return SchedulerOutput(tuple(items))
