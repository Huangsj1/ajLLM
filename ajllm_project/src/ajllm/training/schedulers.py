"""Learning-rate schedules."""

from __future__ import annotations

import math


def warmup_cosine_learning_rate(
    step: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max_learning_rate * step / warmup_steps
    if step >= total_steps:
        return min_learning_rate
    denominator = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / denominator
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)
