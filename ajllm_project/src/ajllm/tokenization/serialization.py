"""GPT-2-compatible vocabulary and merge-table serialization."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """Map every byte to a printable Unicode code point without collisions."""

    byte_values = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    code_points = byte_values[:]
    extra_index = 0
    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            code_points.append(256 + extra_index)
            extra_index += 1
    return dict(zip(byte_values, map(chr, code_points), strict=True))


@lru_cache(maxsize=1)
def unicode_to_bytes() -> dict[str, int]:
    return {character: byte_value for byte_value, character in bytes_to_unicode().items()}


def _encode_bytes(value: bytes) -> str:
    mapping = bytes_to_unicode()
    return "".join(mapping[byte_value] for byte_value in value)


def _decode_bytes(value: str) -> bytes:
    mapping = unicode_to_bytes()
    return bytes(mapping[character] for character in value)


def save_vocab_and_merges(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_path: str | Path,
    merges_path: str | Path,
) -> None:
    serialized_vocab = {_encode_bytes(token): token_id for token_id, token in sorted(vocab.items())}
    with Path(vocab_path).open("w", encoding="utf-8") as output_file:
        json.dump(serialized_vocab, output_file, indent=2, ensure_ascii=False)
    with Path(merges_path).open("w", encoding="utf-8", newline="\n") as output_file:
        for left, right in merges:
            output_file.write(f"{_encode_bytes(left)} {_encode_bytes(right)}\n")


def load_vocab_and_merges(
    vocab_path: str | Path,
    merges_path: str | Path,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with Path(vocab_path).open("r", encoding="utf-8") as input_file:
        serialized_vocab = json.load(input_file)
    vocab = {int(token_id): _decode_bytes(token) for token, token_id in serialized_vocab.items()}

    merges: list[tuple[bytes, bytes]] = []
    with Path(merges_path).open("r", encoding="utf-8") as input_file:
        for line in input_file:
            cleaned = line.rstrip("\r\n")
            if not cleaned:
                continue
            parts = cleaned.split(" ")
            if len(parts) != 2:
                raise ValueError(f"Invalid merge line: {cleaned!r}")
            merges.append((_decode_bytes(parts[0]), _decode_bytes(parts[1])))
    return vocab, merges
