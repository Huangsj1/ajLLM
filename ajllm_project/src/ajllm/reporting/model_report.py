"""Generate Markdown architecture, parameter, memory, and FLOPs reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ajllm.reporting.resource_estimates import ResourceEstimate, estimate_resources


def _parameter_formula(model_config: dict[str, Any]) -> str:
    feed_forward = model_config.get("feed_forward", {}).get("type", "swiglu")
    ffn_factor = 3 if feed_forward == "swiglu" else 2
    embeddings = "VD" if model_config.get("tie_embeddings", False) else "2VD"
    learned_positions = " + TD" if model_config.get("position_encoding", {}).get("type") == "learned" else ""
    norm_type = model_config.get("normalization", {}).get("type", "rmsnorm")
    norm_terms = " + 2D" if norm_type == "rmsnorm" else ""
    final_norm = (
        " + D"
        if norm_type == "rmsnorm" and model_config.get("normalization", {}).get("placement", "pre") == "pre"
        else ""
    )
    return f"N = {embeddings}{learned_positions} + L(4D^2 + {ffn_factor}DF{norm_terms}){final_norm}"


def _format_bytes(value: int) -> str:
    gibibytes = value / (1024**3)
    mebibytes = value / (1024**2)
    if gibibytes >= 1:
        return f"{value:,} bytes ({gibibytes:.3f} GiB)"
    return f"{value:,} bytes ({mebibytes:.3f} MiB)"


def _format_flops(value: int) -> str:
    if value >= 1e15:
        return f"{value / 1e15:.3f} PFLOPs"
    if value >= 1e12:
        return f"{value / 1e12:.3f} TFLOPs"
    if value >= 1e9:
        return f"{value / 1e9:.3f} GFLOPs"
    return f"{value:,} FLOPs"


def _resource_rows(resources: ResourceEstimate) -> str:
    rows = [
        ("Parameters", resources.parameter_bytes),
        ("Saved activations (estimated)", resources.activation_bytes),
        ("Parameter gradients", resources.gradient_bytes),
        ("AdamW first and second moments", resources.optimizer_bytes),
        ("Total estimated training memory", resources.total_memory_bytes),
    ]
    return "\n".join(f"| {name} | {_format_bytes(value)} |" for name, value in rows)


def write_model_report(
    path: str | Path,
    model: torch.nn.Module,
    model_config: dict[str, Any],
    vocab_size: int,
    batch_size: int | str = "B",
    training_config: dict[str, Any] | None = None,
) -> Path:
    """Write a report with exact parameter counts and approximate resources."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    context_length = int(model_config["context_length"])
    d_model = int(model_config["d_model"])
    num_heads = int(model_config["num_heads"])
    num_layers = int(model_config["num_layers"])
    d_ff = int(getattr(model, "d_ff"))
    head_dimension = d_model // num_heads
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    numeric_batch_size = (
        int(batch_size) if isinstance(batch_size, int) else int(model_config.get("report_batch_size", 1))
    )
    report_model_config = {**model_config, "vocab_size": vocab_size}
    resources = estimate_resources(report_model_config, total_parameters, numeric_batch_size, training_config)
    parameter_rows = [
        f"| `{name}` | `{list(parameter.shape)}` | {parameter.numel():,} |"
        for name, parameter in model.named_parameters()
    ]
    configured_batch = (
        batch_size if isinstance(batch_size, int) else f"{batch_size} (estimate uses {numeric_batch_size})"
    )
    training_steps = resources.total_training_steps or "not specified"
    training_tokens = resources.total_training_tokens or "not specified"
    total_training_flops = (
        _format_flops(resources.total_training_flops) if resources.total_training_flops else "not specified"
    )

    document = f"""# Model Architecture: {model_config.get("name", "transformer")}

This report is generated from the resolved model configuration and instantiated model.

```mermaid
flowchart LR
    A["Token IDs [B, T]"] --> B["Token Embedding [B, T, D]"]
    B --> C["Transformer Blocks x {num_layers}"]
    C --> D["Final Normalization [B, T, D]"]
    D --> E["LM Head [B, T, V]"]
```

Each Transformer block contains causal multi-head self-attention and a feed-forward network
connected by residual paths. Normalization placement and position encoding are controlled by
the model configuration.

## Configured Dimensions

| Symbol | Meaning | Value |
|---|---|---:|
| B | Batch size | {configured_batch} |
| T | Context length | {context_length} |
| V | Vocabulary size | {vocab_size} |
| D | Model width | {d_model} |
| L | Transformer layers | {num_layers} |
| H | Attention heads | {num_heads} |
| D/H | Head dimension | {head_dimension} |
| F | Feed-forward width | {d_ff} |

## Tensor Shapes

| Stage | Shape |
|---|---|
| Input token IDs | `[{configured_batch}, {context_length}]` |
| Token embeddings | `[{configured_batch}, {context_length}, {d_model}]` |
| Q, K, V per head | `[{configured_batch}, {num_heads}, {context_length}, {head_dimension}]` |
| Attention scores | `[{configured_batch}, {num_heads}, {context_length}, {context_length}]` |
| Block output | `[{configured_batch}, {context_length}, {d_model}]` |
| LM logits | `[{configured_batch}, {context_length}, {vocab_size}]` |

## Parameters

General formula for the selected architecture:

```text
{_parameter_formula(model_config)}
```

Instantiated parameter count: **{total_parameters:,}**.

| Parameter | Shape | Count |
|---|---|---:|
{chr(10).join(parameter_rows)}

## Estimated Training Memory

| Component | Estimate |
|---|---:|
{_resource_rows(resources)}

The estimate uses parameter/gradient dtype size `{resources.parameter_dtype_bytes}` bytes,
activation dtype size `{resources.activation_dtype_bytes}` bytes, and
`{resources.optimizer_state_dtype_bytes}` bytes per AdamW moment. It excludes CUDA workspaces,
allocator fragmentation, data-loader memory, and framework overhead.

## Estimated Matrix FLOPs

| Quantity | Estimate |
|---|---:|
| Forward FLOPs per iteration | {_format_flops(resources.forward_flops_per_step)} |
| Training FLOPs per iteration (3x forward approximation) | {_format_flops(resources.training_flops_per_step)} |
| Planned training steps | {training_steps} |
| Planned training tokens | {training_tokens} |
| Total training FLOPs | {total_training_flops} |

The FLOPs estimate counts matrix multiplications only: Q/K/V/output projections, attention QK^T
and AV products, feed-forward projections, and the LM head. Elementwise activation,
normalization, masking, sampling, and optimizer arithmetic are omitted.
"""
    destination.write_text(document, encoding="utf-8")
    return destination
