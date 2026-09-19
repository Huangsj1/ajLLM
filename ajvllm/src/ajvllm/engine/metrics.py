"""Small CPU counters; tensor/device profiling is deliberately separate."""

from dataclasses import dataclass


@dataclass
class EngineMetrics:
    num_steps: int = 0
    scheduled_prefill_tokens: int = 0
    scheduled_decode_tokens: int = 0
    generated_tokens: int = 0
    finished_requests: int = 0
    cancelled_requests: int = 0
    failed_requests: int = 0
