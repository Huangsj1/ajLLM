"""Lazy JSONL dataset for causal-language-model pre-training."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset[dict[str, torch.Tensor]]):
    """Index JSONL byte offsets rather than retaining a large corpus in RAM."""

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: object,
        sequence_length: int,
        *,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int = 0,
    ) -> None:
        self.data_path = Path(data_path)
        if not self.data_path.is_file():
            raise FileNotFoundError(self.data_path)
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.bos_token_id, self.eos_token_id, self.pad_token_id = bos_token_id, eos_token_id, pad_token_id
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

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with self.data_path.open("rb") as source:
            source.seek(self.offsets[index])
            record = json.loads(source.readline())
        text = record.get("text")
        if not isinstance(text, str):
            raise ValueError(f"Record {index} has no string 'text' field")
        token_ids = [self.bos_token_id, *self.tokenizer.encode(text), self.eos_token_id]
        needed = self.sequence_length + 1
        if len(token_ids) > needed:
            token_ids = token_ids[:needed]
            token_ids[-1] = self.eos_token_id
        token_ids.extend([self.pad_token_id] * (needed - len(token_ids)))
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)
        labels.masked_fill_(labels == self.pad_token_id, -100)
        return {"input_ids": input_ids, "labels": labels}
