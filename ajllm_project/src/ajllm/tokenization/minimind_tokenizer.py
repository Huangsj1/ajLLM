"""Small training interface around MiniMind's published BPE tokenizer."""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer as BackendTokenizer


class MiniMindTokenizer:
    """Load the exact MiniMind tokenizer JSON without depending on Transformers.

    Only the stable encode/decode and special-token surface needed by the
    project's training datasets is exposed.  The SFT dataset owns the explicit
    MiniMind chat serialization so this adapter does not need Transformers or
    a Jinja runtime.
    """

    def __init__(self, tokenizer_path: str | Path, config_path: str | Path | None = None) -> None:
        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.is_file():
            raise FileNotFoundError(tokenizer_path)
        config_path = Path(config_path) if config_path else tokenizer_path.with_name("tokenizer_config.json")
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        self._backend = BackendTokenizer.from_file(str(tokenizer_path))
        self._config = json.loads(config_path.read_text(encoding="utf-8"))
        self.pad_token = self._config["pad_token"]
        self.bos_token = self._config["bos_token"]
        self.eos_token = self._config["eos_token"]
        self.pad_token_id = self._token_id(self.pad_token)
        self.bos_token_id = self._token_id(self.bos_token)
        self.eos_token_id = self._token_id(self.eos_token)
        self.vocab_size = self._backend.get_vocab_size(with_added_tokens=True)

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> MiniMindTokenizer:
        """Load ``tokenizer.json`` and its accompanying tokenizer config."""
        directory = Path(directory)
        return cls(directory / "tokenizer.json", directory / "tokenizer_config.json")

    def encode(self, text: str) -> list[int]:
        """Encode text without adding sequence-level BOS/EOS tokens."""
        return self._backend.encode(text, add_special_tokens=False).ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._backend.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def _token_id(self, token: str) -> int:
        token_id = self._backend.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Configured special token is missing from tokenizer.json: {token}")
        return token_id
