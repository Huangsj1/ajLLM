"""Prompt-only RLAIF dataset used by online GRPO rollouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from ajllm.datasets.sft import encode_sft_conversation


class RLAIFDataset(Dataset[dict[str, Any]]):
    """Lazily expose unfinished conversations as MiniMind generation prompts."""

    def __init__(
        self, data_path: str | Path, tokenizer: object, max_prompt_length: int, *, open_thinking: bool
    ) -> None:
        self.data_path = Path(data_path)
        if not self.data_path.is_file():
            raise FileNotFoundError(self.data_path)
        if max_prompt_length < 2:
            raise ValueError("max_prompt_length must be at least 2")
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.open_thinking = open_thinking
        self.offsets: list[int] = []
        with self.data_path.open("rb") as source:
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        if not self.offsets:
            raise ValueError(f"No JSONL records found in {self.data_path}")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self.data_path.open("rb") as source:
            source.seek(self.offsets[index])
            record = json.loads(source.readline())
        conversations = record.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            raise ValueError(f"Record {index} requires a non-empty conversation plus an unfinished assistant turn")
        final_message = conversations[-1]
        if not isinstance(final_message, dict) or final_message.get("role") != "assistant":
            raise ValueError(f"Record {index} must end with an assistant message reserved for the rollout")
        messages = [dict(message) for message in conversations[:-1]]
        token_ids, _ = encode_sft_conversation(self.tokenizer, messages)
        completion_prefix = "<think>\n" if self.open_thinking else ""
        assistant_prefix = (
            f"{self.tokenizer.bos_token}assistant\n{completion_prefix}"
            if self.open_thinking
            else f"{self.tokenizer.bos_token}assistant\n<think>\n\n</think>\n\n"
        )
        token_ids.extend(self.tokenizer.encode(assistant_prefix))
        token_ids = token_ids[-self.max_prompt_length :]
        if len(token_ids) < 1:
            raise ValueError(f"Record {index} produced an empty rollout prompt")
        return {"prompt_ids": token_ids, "messages": messages, "completion_prefix": completion_prefix}


def collate_rlaif(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Keep variable-length prompts unpadded for the mask-free ajLLM decoder."""
    return {key: [sample[key] for sample in batch] for key in ("prompt_ids", "messages", "completion_prefix")}
