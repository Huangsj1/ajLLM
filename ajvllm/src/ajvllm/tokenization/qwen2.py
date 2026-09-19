"""Local checkpoint tokenizer and chat template; no model execution delegation."""

from collections.abc import Sequence
from pathlib import Path

from transformers import AutoTokenizer


class Qwen2Tokenizer:
    def __init__(self, directory: str | Path):
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(self.tokenizer.encode(text, add_special_tokens=False))

    def encode_chat(self, messages: Sequence[dict[str, str]]) -> tuple[int, ...]:
        return tuple(self.tokenizer.apply_chat_template(list(messages), tokenize=True, add_generation_prompt=True))

    def decode(self, token_ids: Sequence[int]) -> str:
        # Decode accumulated IDs, never single byte-level tokens independently.
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True, clean_up_tokenization_spaces=False)
