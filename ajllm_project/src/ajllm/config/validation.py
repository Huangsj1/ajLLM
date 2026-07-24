"""Validation helpers for resolved configurations."""

from __future__ import annotations

from typing import Any

from ajllm.config.loader import ConfigError


def require_keys(mapping: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigError(f"Missing {context} configuration values: {', '.join(missing)}")


def validate_dataset(dataset: dict[str, Any]) -> None:
    require_keys(dataset, ("name", "splits"), "dataset")
    if not isinstance(dataset["splits"], dict) or not dataset["splits"]:
        raise ConfigError("dataset.splits must be a non-empty mapping")


def validate_tokenizer(tokenizer: dict[str, Any]) -> None:
    require_keys(tokenizer, ("name", "vocab_size"), "tokenizer")
    if int(tokenizer["vocab_size"]) < 256 + len(tokenizer.get("special_tokens", [])):
        raise ConfigError("tokenizer.vocab_size is too small for all bytes and special tokens")


def validate_model(model: dict[str, Any]) -> None:
    require_keys(model, ("name", "context_length", "d_model", "num_layers", "num_heads"), "model")
    if int(model["d_model"]) % int(model["num_heads"]) != 0:
        raise ConfigError("model.d_model must be divisible by model.num_heads")
    if int(model["context_length"]) < 2:
        raise ConfigError("model.context_length must be at least 2")
