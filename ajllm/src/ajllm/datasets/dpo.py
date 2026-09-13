"""Preference-pair dataset for Direct Preference Optimization (DPO)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ajllm.datasets.sft import encode_sft_conversation


class DPODataset(Dataset[dict[str, torch.Tensor]]):
    """Lazy JSONL DPO pairs using the same MiniMind chat serialization as SFT.

    A record contains ``chosen`` and ``rejected`` message arrays.  They must
    have the same conversation prefix and end in alternate assistant turns;
    only assistant completion tokens are scored by DPO.  Keeping this encoder
    aligned with :class:`SFTDataset` prevents template drift between SFT, DPO,
    and chat generation.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: object,
        sequence_length: int,
        *,
        pad_token_id: int = 0,
    ) -> None:
        self.data_path = Path(data_path)
        if not self.data_path.is_file():
            raise FileNotFoundError(self.data_path)
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id
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
        chosen, rejected = record.get("chosen"), record.get("rejected")
        self._validate_pair(chosen, rejected, index)
        chosen_ids, chosen_labels = self._encode(chosen, index, "chosen")
        rejected_ids, rejected_labels = self._encode(rejected, index, "rejected")
        return {
            "chosen_input_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_ids,
            "rejected_labels": rejected_labels,
        }

    def _encode(self, conversations: list[Any], index: int, preference: str) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids, target_ids = encode_sft_conversation(self.tokenizer, conversations)
        needed = self.sequence_length + 1
        # Match SFT's recent-context truncation policy.  It retains the end of
        # the answer, which is the part DPO needs to compare.
        token_ids, target_ids = token_ids[-needed:], target_ids[-needed:]
        if not any(target != -100 for target in target_ids):
            raise ValueError(f"Record {index} {preference!r} has no assistant target tokens after truncation")
        padding = needed - len(token_ids)
        token_ids.extend([self.pad_token_id] * padding)
        target_ids.extend([-100] * padding)
        return (
            torch.tensor(token_ids[:-1], dtype=torch.long),
            torch.tensor(target_ids[1:], dtype=torch.long),
        )

    @staticmethod
    def _validate_pair(chosen: Any, rejected: Any, index: int) -> None:
        if not isinstance(chosen, list) or not isinstance(rejected, list) or not chosen or not rejected:
            raise ValueError(f"Record {index} must contain non-empty 'chosen' and 'rejected' message lists")
        if chosen[:-1] != rejected[:-1]:
            raise ValueError(f"Record {index} chosen and rejected must share an identical prompt prefix")
        for preference, messages in (("chosen", chosen), ("rejected", rejected)):
            last_message = messages[-1]
            if not isinstance(last_message, dict) or last_message.get("role") != "assistant":
                raise ValueError(f"Record {index} {preference!r} must end with an assistant message")
