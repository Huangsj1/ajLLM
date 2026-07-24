"""Approximate memory and matrix-operation estimates for Transformer training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResourceEstimate:
    parameter_count: int
    parameter_bytes: int
    activation_elements: int
    activation_bytes: int
    gradient_bytes: int
    optimizer_bytes: int
    total_memory_bytes: int
    forward_flops_per_step: int
    training_flops_per_step: int
    total_training_flops: int
    total_training_steps: int
    total_training_tokens: int
    parameter_dtype_bytes: int
    activation_dtype_bytes: int
    optimizer_state_dtype_bytes: int


def estimate_resources(
    model_config: dict[str, Any],
    parameter_count: int,
    batch_size: int,
    training_config: dict[str, Any] | None = None,
) -> ResourceEstimate:
    """Estimate training resources using a transparent matrix-only model.

    The activation estimate assumes tensors retained for backward, while FLOPs
    count multiply-adds as two operations and approximate training as three
    forward-equivalent passes. CUDA workspace, allocator fragmentation, data
    loading, and non-matrix operations are intentionally excluded.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive for resource estimation")
    training_config = training_config or {}
    context_length = int(model_config["context_length"])
    d_model = int(model_config["d_model"])
    num_layers = int(model_config["num_layers"])
    num_heads = int(model_config["num_heads"])
    d_ff = int(model_config.get("d_ff", ((int(8 * d_model / 3) + 63) // 64) * 64))
    vocab_size = int(model_config.get("vocab_size", training_config.get("vocab_size", 0)))
    if vocab_size < 1:
        vocab_size = int(model_config.get("report_vocab_size", 1))
    feed_forward_type = model_config.get("feed_forward", {}).get("type", "swiglu")

    parameter_dtype_bytes = int(model_config.get("report_parameter_dtype_bytes", 4))
    activation_dtype_bytes = int(model_config.get("report_activation_dtype_bytes", parameter_dtype_bytes))
    optimizer_state_dtype_bytes = int(model_config.get("report_optimizer_state_dtype_bytes", 4))
    if min(parameter_dtype_bytes, activation_dtype_bytes, optimizer_state_dtype_bytes) < 1:
        raise ValueError("Resource dtype byte sizes must be positive")

    tokens_per_step = batch_size * context_length
    ffn_intermediate_count = 3 if feed_forward_type == "swiglu" else 2
    hidden_elements = batch_size * context_length * d_model
    attention_matrix_elements = batch_size * num_heads * context_length * context_length
    feed_forward_elements = batch_size * context_length * d_ff
    activation_elements = (
        hidden_elements
        + num_layers
        * (5 * hidden_elements + 2 * attention_matrix_elements + ffn_intermediate_count * feed_forward_elements)
        + batch_size * context_length * vocab_size
    )

    attention_projection_flops = 10 * tokens_per_step * d_model * d_model
    attention_product_flops = 4 * batch_size * context_length * context_length * d_model
    ffn_flops = ffn_intermediate_count * 2 * tokens_per_step * d_model * d_ff
    lm_head_flops = 2 * tokens_per_step * d_model * vocab_size
    forward_flops = num_layers * (attention_projection_flops + attention_product_flops + ffn_flops) + lm_head_flops
    training_flops = 3 * forward_flops

    total_steps = int(training_config.get("max_steps", 0))
    if total_steps <= 0 and training_config.get("total_tokens") is not None:
        total_steps = max(1, math.ceil(int(training_config["total_tokens"]) / tokens_per_step))
    total_tokens = total_steps * tokens_per_step

    parameter_bytes = parameter_count * parameter_dtype_bytes
    activation_bytes = activation_elements * activation_dtype_bytes
    gradient_bytes = parameter_count * parameter_dtype_bytes
    optimizer_bytes = parameter_count * 2 * optimizer_state_dtype_bytes
    return ResourceEstimate(
        parameter_count=parameter_count,
        parameter_bytes=parameter_bytes,
        activation_elements=activation_elements,
        activation_bytes=activation_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_bytes=optimizer_bytes,
        total_memory_bytes=parameter_bytes + activation_bytes + gradient_bytes + optimizer_bytes,
        forward_flops_per_step=forward_flops,
        training_flops_per_step=training_flops,
        total_training_flops=training_flops * total_steps,
        total_training_steps=total_steps,
        total_training_tokens=total_tokens,
        parameter_dtype_bytes=parameter_dtype_bytes,
        activation_dtype_bytes=activation_dtype_bytes,
        optimizer_state_dtype_bytes=optimizer_state_dtype_bytes,
    )
