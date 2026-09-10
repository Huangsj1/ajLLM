"""Lazy MiniMind-format conversational dataset for supervised fine-tuning."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def encode_sft_conversation(tokenizer: object, conversations: list[Any]) -> tuple[list[int], list[int]]:
    """Encode a chat without opening a dataset file.

    Generation uses this same serialization path as SFT labels, preventing chat
    prompt/template drift between training and inference.
    """
    encoder = object.__new__(SFTDataset)
    encoder.tokenizer = tokenizer
    return encoder._encode_conversation(conversations, record_index=-1)


class SFTDataset(Dataset[dict[str, torch.Tensor]]):
    """Serialize MiniMind conversations and supervise assistant completions only.

    Records are indexed by byte offset, matching :class:`PretrainDataset`'s
    memory profile.  The serialization is the local tokenizer's chat template,
    including ``<think>`` blocks and tool calls, implemented here without a
    Transformers/Jinja dependency.  The returned labels are already shifted for
    ajLLM's causal-LM loss: all system, user, tool, and assistant-header tokens
    are ``-100``; assistant completion tokens through ``<|im_end|>`` are targets.
    """

    _TOOL_PREAMBLE = (
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
    )
    _TOOL_POSTAMBLE = (
        "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, "
        "\"arguments\": <args-json-object>}\n</tool_call>"
    )
    _VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})

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
        for attribute in ("encode", "bos_token", "eos_token"):
            if not hasattr(tokenizer, attribute):
                raise TypeError(f"SFT tokenizer must provide {attribute}")
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
        conversations = record.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            raise ValueError(f"Record {index} has no non-empty 'conversations' list")
        token_ids, target_ids = self._encode_conversation(conversations, index)
        needed = self.sequence_length + 1
        # Keep the most recent turns and, critically, their answers.  This is
        # preferable to right truncation, which often discards the final target.
        token_ids, target_ids = token_ids[-needed:], target_ids[-needed:]
        if not any(target != -100 for target in target_ids):
            raise ValueError(f"Record {index} has no assistant target tokens after truncation")
        padding = needed - len(token_ids)
        token_ids.extend([self.pad_token_id] * padding)
        target_ids.extend([-100] * padding)
        return {
            "input_ids": torch.tensor(token_ids[:-1], dtype=torch.long),
            "labels": torch.tensor(target_ids[1:], dtype=torch.long),
        }

    def _encode_conversation(self, conversations: list[Any], record_index: int) -> tuple[list[int], list[int]]:
        """Encode one conversation with the bundled MiniMind chat template."""
        messages = [
            self._validate_message(message, record_index, message_index)
            for message_index, message in enumerate(conversations)
        ]
        token_ids: list[int] = []
        target_ids: list[int] = []

        def add(text: str, supervise: bool = False) -> None:
            ids = self.tokenizer.encode(text)
            token_ids.extend(ids)
            target_ids.extend(ids if supervise else [-100] * len(ids))

        tools = self._tools_from_first_system(messages, record_index)
        if tools is not None:
            add(f"{self.tokenizer.bos_token}system\n")
            if messages[0]["role"] == "system":
                add(f"{messages[0]['content']}\n\n")
            add(self._TOOL_PREAMBLE)
            for tool in tools:
                add(f"\n{json.dumps(tool, ensure_ascii=False, separators=(',', ':'))}")
            add(f"{self._TOOL_POSTAMBLE}{self.tokenizer.eos_token}\n")
        elif messages[0]["role"] == "system":
            add(
                f"{self.tokenizer.bos_token}system\n{messages[0]['content']}"
                f"{self.tokenizer.eos_token}\n"
            )

        for message_index, message in enumerate(messages):
            role, content = message["role"], message["content"]
            if role == "system":
                if message_index != 0:
                    add(f"{self.tokenizer.bos_token}system\n{content}{self.tokenizer.eos_token}\n")
            elif role == "user":
                add(f"{self.tokenizer.bos_token}user\n{content}{self.tokenizer.eos_token}\n")
            elif role == "assistant":
                add(f"{self.tokenizer.bos_token}assistant\n")
                reasoning_value = message.get("reasoning_content")
                if reasoning_value is None:
                    reasoning = ""
                elif isinstance(reasoning_value, str):
                    reasoning = reasoning_value
                else:
                    raise ValueError(f"Record {record_index}, message {message_index} has non-string reasoning_content")
                # Match the tokenizer template: an inline <think> block is
                # split only when reasoning_content was not supplied.
                if not isinstance(reasoning_value, str) and "</think>" in content:
                    reasoning, content = self._split_inline_reasoning(content)
                completion = f"<think>\n{reasoning.strip(chr(10))}\n</think>\n\n{content.lstrip(chr(10))}"
                add(completion, supervise=True)
                tool_calls = self._parse_json_field(
                    message.get("tool_calls"), record_index, message_index, "tool_calls"
                )
                if tool_calls is not None:
                    if not isinstance(tool_calls, list):
                        raise ValueError(f"Record {record_index}, message {message_index} tool_calls must be a list")
                    for call_index, tool_call in enumerate(tool_calls):
                        if not isinstance(tool_call, Mapping):
                            raise ValueError(
                                f"Record {record_index}, message {message_index}, "
                                f"tool call {call_index} must be an object"
                            )
                        call = tool_call.get("function", tool_call)
                        if not isinstance(call, Mapping) or not isinstance(call.get("name"), str):
                            raise ValueError(
                                f"Record {record_index}, message {message_index}, "
                                f"tool call {call_index} has no function name"
                            )
                        arguments = call.get("arguments", {})
                        argument_text = arguments if isinstance(arguments, str) else json.dumps(
                            arguments, ensure_ascii=False, separators=(",", ":")
                        )
                        separator = "\n" if call_index or content else ""
                        tool_call_text = (
                            f'{separator}<tool_call>\n{{"name": "{call["name"]}", '
                            f'"arguments": {argument_text}}}\n</tool_call>'
                        )
                        add(tool_call_text, supervise=True)
                add(f"{self.tokenizer.eos_token}\n", supervise=True)
            else:  # role == "tool", validated above
                previous_is_tool = message_index > 0 and messages[message_index - 1]["role"] == "tool"
                next_is_tool = message_index + 1 < len(messages) and messages[message_index + 1]["role"] == "tool"
                if not previous_is_tool:
                    add(f"{self.tokenizer.bos_token}user")
                add(f"\n<tool_response>\n{content}\n</tool_response>")
                if not next_is_tool:
                    add(f"{self.tokenizer.eos_token}\n")
        return token_ids, target_ids

    def _validate_message(self, message: Any, record_index: int, message_index: int) -> dict[str, Any]:
        if not isinstance(message, Mapping):
            raise ValueError(f"Record {record_index}, message {message_index} must be an object")
        role, content = message.get("role"), message.get("content")
        if role not in self._VALID_ROLES:
            raise ValueError(f"Record {record_index}, message {message_index} has unsupported role {role!r}")
        if not isinstance(content, str):
            raise ValueError(f"Record {record_index}, message {message_index} has non-string content")
        return dict(message)

    def _tools_from_first_system(self, messages: Sequence[dict[str, Any]], record_index: int) -> list[Any] | None:
        if messages[0]["role"] != "system" or not messages[0].get("tools"):
            return None
        tools = self._parse_json_field(messages[0]["tools"], record_index, 0, "tools")
        if not isinstance(tools, list):
            raise ValueError(f"Record {record_index}, message 0 tools must be a list")
        return tools

    @staticmethod
    def _parse_json_field(value: Any, record_index: int, message_index: int, field: str) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Record {record_index}, message {message_index} has invalid JSON {field}"
                ) from error
        return value

    @staticmethod
    def _split_inline_reasoning(content: str) -> tuple[str, str]:
        reasoning, answer = content.split("</think>", maxsplit=1)
        return reasoning.rstrip("\n").split("<think>")[-1].lstrip("\n"), answer.lstrip("\n")
