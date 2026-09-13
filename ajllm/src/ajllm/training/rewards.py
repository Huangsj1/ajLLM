"""Adapters for frozen external reward models used by online RL stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


class SkyworkRewardModel:
    """Batch scorer for local Skywork-Reward-V2 Qwen sequence classifiers.

    Skywork Reward V2 is a Bradley--Terry reward model. 
    Its official recipe applies its own chat template and reads the one-dimensional sequence-classification logit.
    System messages are deliberately omitted, as instructed by the model card.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: torch.device,
        *,
        max_length: int = 4096,
        dtype: torch.dtype | None = None,
    ) -> None:
        source = Path(model_path)
        if not source.is_dir():
            raise FileNotFoundError(f"reward_model.path does not exist: {source}")
        if max_length < 1:
            raise ValueError("reward_model.max_length must be positive")
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "GRPO reward scoring requires transformers; run `uv sync` after updating the project"
            ) from error
        load_dtype = dtype if device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(source, torch_dtype=load_dtype, num_labels=1)
            .to(device)
            .eval()
        )
        self.device = device
        self.max_length = max_length
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _without_system_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        filtered = [dict(message) for message in messages if message.get("role") != "system"]
        if not filtered:
            raise ValueError("Skywork reward scoring requires at least one non-system conversation message")
        return filtered

    def _format_conversation(self, messages: Sequence[Mapping[str, Any]]) -> str:
        formatted = self.tokenizer.apply_chat_template(
            self._without_system_messages(messages), tokenize=False, add_generation_prompt=False
        )
        bos_token = self.tokenizer.bos_token
        return formatted[len(bos_token) :] if bos_token and formatted.startswith(bos_token) else formatted

    @torch.no_grad()
    def score(self, conversations: Sequence[Sequence[Mapping[str, Any]]]) -> torch.Tensor:
        """Return one raw Bradley--Terry reward logit per complete conversation."""
        if not conversations:
            return torch.empty(0, device=self.device, dtype=torch.float32)
        formatted = [self._format_conversation(conversation) for conversation in conversations]
        encoded = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        ).to(self.device)
        return self.model(**encoded).logits.reshape(-1).float()
