"""Model construction and YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ajllm.modeling.transformer import ModelConfig, TransformerLM


def load_model_config(path: str | Path, vocab_size: int) -> ModelConfig:
    """Load a model YAML file, with the tokenizer vocabulary as the source of truth."""
    with Path(path).open(encoding="utf-8") as source:
        values: dict[str, Any] = yaml.safe_load(source) or {}
    values.pop("name", None)
    values["vocab_size"] = vocab_size
    return ModelConfig(**values)


def build_model(config: ModelConfig) -> TransformerLM:
    return TransformerLM(config)


def create_model_from_config(path: str | Path, vocab_size: int) -> TransformerLM:
    """Compatibility convenience wrapper for scripts and notebooks."""
    return build_model(load_model_config(path, vocab_size))
