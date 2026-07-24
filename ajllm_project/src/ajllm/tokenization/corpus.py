"""Shared corpus pre-tokenization utilities."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import BinaryIO

import regex

GPT2_PRETOKEN_PATTERN = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def compile_special_pattern(special_tokens: list[str]) -> regex.Pattern[str] | None:
    if not special_tokens:
        return None
    alternatives = "|".join(regex.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
    return regex.compile(f"({alternatives})")


def iter_pretokens(text: str, special_tokens: list[str], include_special: bool) -> Iterator[str]:
    """Yield GPT-2-style pre-tokens while treating special tokens as hard boundaries."""

    special_pattern = compile_special_pattern(special_tokens)
    if special_pattern is None:
        yield from (match.group(0) for match in GPT2_PRETOKEN_PATTERN.finditer(text))
        return

    special_set = set(special_tokens)
    for segment in special_pattern.split(text):
        if not segment:
            continue
        if segment in special_set:
            if include_special:
                yield segment
            continue
        yield from (match.group(0) for match in GPT2_PRETOKEN_PATTERN.finditer(segment))


def iter_corpus_lines(path: str) -> Iterator[str]:
    """Read a UTF-8 corpus one line at a time with bounded memory."""

    with open(path, encoding="utf-8") as input_file:
        yield from input_file


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Find byte boundaries immediately before document separators.

    Splitting only at a special token prevents a BPE pair from crossing a
    document boundary while allowing each worker to read its own byte range.
    """

    if desired_num_chunks < 1:
        raise ValueError("desired_num_chunks must be positive")
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    if file_size == 0:
        return [0]
    chunk_size = max(1, file_size // desired_num_chunks)
    boundaries = [index * chunk_size for index in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size
    mini_chunk_size = 4096

    for boundary_index in range(1, len(boundaries) - 1):
        position = boundaries[boundary_index]
        while position < file_size:
            file.seek(position)
            chunk = file.read(mini_chunk_size)
            if not chunk:
                boundaries[boundary_index] = file_size
                break
            found_at = chunk.find(split_special_token)
            if found_at >= 0:
                boundaries[boundary_index] = position + found_at
                break
            position += mini_chunk_size
    return sorted(set(boundaries))
