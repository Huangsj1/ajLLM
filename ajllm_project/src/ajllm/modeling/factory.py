"""Construct model instances from resolved model configurations."""

from __future__ import annotations

from typing import Any

from ajllm.config.validation import validate_model
from ajllm.modeling.transformer import TransformerLM


def build_model(config: dict[str, Any], vocab_size: int) -> TransformerLM:
    """Build a model instance from a resolved model configuration."""

    validate_model(config)
    position = config.get("position_encoding", {})
    normalization = config.get("normalization", {})
    feed_forward = config.get("feed_forward", {})
    d_model = int(config["d_model"])
    feed_forward_type = str(feed_forward.get("type", "swiglu"))
    default_d_ff = ((int(8 * d_model / 3) + 63) // 64) * 64 if feed_forward_type == "swiglu" else 4 * d_model
    return TransformerLM(
        vocab_size=vocab_size,
        context_length=int(config["context_length"]),
        d_model=d_model,
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        d_ff=int(config.get("d_ff", default_d_ff)),
        position_encoding=str(position.get("type", "rope")),
        rope_theta=float(position.get("theta", 10_000.0)),
        normalization=str(normalization.get("type", "rmsnorm")),
        norm_placement=str(normalization.get("placement", "pre")),
        feed_forward=feed_forward_type,
        tie_embeddings=bool(config.get("tie_embeddings", False)),
    )
