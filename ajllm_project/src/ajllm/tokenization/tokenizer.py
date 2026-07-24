"""Byte-level BPE encoding and decoding."""

from __future__ import annotations

import codecs
import os
import shutil
import tempfile
from array import array
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

from ajllm.tokenization.corpus import iter_pretokens
from ajllm.tokenization.serialization import load_vocab_and_merges

_WORKER_TOKENIZER: Tokenizer | None = None
_TOKEN_BUFFER_SIZE = 65_536


def _init_file_encoder(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
) -> None:
    """Initialize one tokenizer per worker process."""

    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = Tokenizer(vocab, merges, special_tokens)


def _read_text_chunk(file_name: str, start: int, end: int) -> str:
    with open(file_name, "rb") as input_file:
        input_file.seek(start)
        text = input_file.read(end - start).decode("utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _encode_file_chunk_to_bin(task: tuple[int, str, int, int, str]) -> tuple[int, str, int]:
    """Encode one byte range into a temporary binary chunk."""

    chunk_id, file_name, start, end, chunk_path = task
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("File encoder worker was not initialized")
    token_count = 0
    buffer = array("I")
    with open(chunk_path, "wb") as output_file:
        for token_id in _WORKER_TOKENIZER.encode_iter(_read_text_chunk(file_name, start, end)):
            if not 0 <= token_id <= 0xFFFFFFFF:
                raise ValueError(f"Token id does not fit in uint32: {token_id}")
            buffer.append(token_id)
            token_count += 1
            if len(buffer) >= _TOKEN_BUFFER_SIZE:
                buffer.tofile(output_file)
                buffer = array("I")
        if buffer:
            buffer.tofile(output_file)
    return chunk_id, chunk_path, token_count


class Tokenizer:
    """Encode arbitrary Unicode text with a trained byte-level BPE model."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])
        self.id_to_token = self.vocab
        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self.merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}
        missing = [token for token in self.special_tokens if token.encode("utf-8") not in self.token_to_id]
        if missing:
            raise ValueError(f"Special tokens are missing from the vocabulary: {missing}")

    @classmethod
    def from_files(
        cls,
        vocab_path: str | Path,
        merges_path: str | Path,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        vocab, merges = load_vocab_and_merges(vocab_path, merges_path)
        return cls(vocab, merges, special_tokens)

    def _merge_pretoken(self, pretoken: str) -> list[bytes]:
        symbols = [bytes([byte_value]) for byte_value in pretoken.encode("utf-8")]
        while len(symbols) > 1:
            candidates = (
                (self.merge_rank[pair], index)
                for index, pair in enumerate(zip(symbols, symbols[1:]))
                if pair in self.merge_rank
            )
            best = min(candidates, default=None)
            if best is None:
                break
            index = best[1]
            symbols[index : index + 2] = [symbols[index] + symbols[index + 1]]
        return symbols

    def encode_iter(self, text: str) -> Iterator[int]:
        special_set = set(self.special_tokens)
        for pretoken in iter_pretokens(text, self.special_tokens, include_special=True):
            if pretoken in special_set:
                yield self.token_to_id[pretoken.encode("utf-8")]
                continue
            for token in self._merge_pretoken(pretoken):
                yield self.token_to_id[token]

    def encode(self, text: str) -> list[int]:
        return list(self.encode_iter(text))

    def _iter_encoded_ids(self, text: str) -> Iterator[int]:
        """Compatibility alias used by file workers and downstream callers."""

        yield from self.encode_iter(text)

    def encode_lines(self, lines: Iterable[str]) -> Iterator[int]:
        for line in lines:
            yield from self.encode_iter(line)

    @staticmethod
    def _file_chunk_ranges(file_name: str | Path, chunk_size: int) -> list[tuple[int, int]]:
        """Return UTF-8-safe byte ranges, preferring newline boundaries."""

        if chunk_size < 4:
            raise ValueError("chunk_size must be at least 4 bytes")
        ranges: list[tuple[int, int]] = []
        with Path(file_name).open("rb") as input_file:
            start = 0
            while data := input_file.read(chunk_size):
                newline = data.rfind(b"\n")
                if newline >= 0:
                    length = newline + 1
                elif len(data) < chunk_size:
                    length = len(data)
                else:
                    length = len(data)
                    if data[-1] & 0xC0 == 0x80:
                        while length > 0 and data[length - 1] & 0xC0 == 0x80:
                            length -= 1
                        length -= 1
                    if length > 0 and data[length - 1] == ord("\r"):
                        length -= 1
                if length <= 0:
                    raise ValueError("chunk_size is too small for a UTF-8 character")
                end = start + length
                ranges.append((start, end))
                input_file.seek(end)
                start = end
        return ranges

    def encode_file_to_disk(
        self,
        input_path: str | Path,
        output_path: str | Path,
        chunk_size: int = 16 * 1024 * 1024,
        parallel: bool = True,
        num_processes: int | None = None,
        buffer_tokens: int = _TOKEN_BUFFER_SIZE,
    ) -> int:
        """Encode a UTF-8 corpus with optional process-based chunk parallelism."""

        source = Path(input_path)
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if chunk_size < 4:
            raise ValueError("chunk_size must be at least 4 bytes")
        ranges = self._file_chunk_ranges(source, chunk_size)
        if not ranges:
            destination.write_bytes(b"")
            return 0

        process_count = min(num_processes or (os.cpu_count() or 1), len(ranges))
        if not parallel or process_count <= 1:
            return self._encode_file_sequential(source, destination, buffer_tokens)

        temporary_fd, temporary_output = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(temporary_fd)
        token_count = 0
        try:
            with tempfile.TemporaryDirectory(prefix="tokenizer-chunks-", dir=destination.parent) as chunk_dir:
                tasks = [
                    (chunk_id, str(source), start, end, str(Path(chunk_dir) / f"{chunk_id:08d}.bin"))
                    for chunk_id, (start, end) in enumerate(ranges)
                ]
                with (
                    ProcessPoolExecutor(
                        max_workers=process_count,
                        initializer=_init_file_encoder,
                        initargs=(self.vocab, self.merges, self.special_tokens),
                    ) as executor,
                    open(temporary_output, "wb") as output_file,
                    tqdm(
                        total=source.stat().st_size,
                        desc=f"Encoding {source.name}",
                        unit="B",
                        unit_scale=True,
                    ) as progress,
                ):
                    for chunk_id, chunk_path, chunk_token_count in executor.map(_encode_file_chunk_to_bin, tasks):
                        with open(chunk_path, "rb") as chunk_file:
                            shutil.copyfileobj(chunk_file, output_file, length=1024 * 1024)
                        os.unlink(chunk_path)
                        token_count += chunk_token_count
                        start, end = ranges[chunk_id]
                        progress.update(end - start)
            os.replace(temporary_output, destination)
        except BaseException:
            if os.path.exists(temporary_output):
                os.unlink(temporary_output)
            raise
        return token_count

    def _encode_file_sequential(self, source: Path, destination: Path, buffer_tokens: int) -> int:
        token_count = 0
        buffer = array("I")
        with source.open(encoding="utf-8") as input_file, destination.open("wb") as output_file:
            for line in tqdm(input_file, desc=f"Encoding {source.name}", unit="line"):
                for token_id in self.encode_iter(line):
                    if not 0 <= token_id <= 0xFFFFFFFF:
                        raise ValueError(f"Token id does not fit in uint32: {token_id}")
                    buffer.append(token_id)
                    token_count += 1
                    if len(buffer) >= buffer_tokens:
                        buffer.tofile(output_file)
                        buffer = array("I")
            if buffer:
                buffer.tofile(output_file)
        return token_count

    def decode(self, token_ids: Iterable[int]) -> str:
        combined = b"".join(self.id_to_token[int(token_id)] for token_id in token_ids)
        return combined.decode("utf-8", errors="replace")

    def decode_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        chunk_tokens: int = 65_536,
    ) -> int:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        token_count = 0
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Path(input_path).open("rb") as input_file, destination.open("w", encoding="utf-8") as output_file:
            while True:
                token_ids = array("I")
                try:
                    token_ids.fromfile(input_file, chunk_tokens)
                except EOFError:
                    pass
                if not token_ids:
                    break
                token_count += len(token_ids)
                token_bytes = b"".join(self.id_to_token[int(token_id)] for token_id in token_ids)
                output_file.write(decoder.decode(token_bytes, final=False))
            output_file.write(decoder.decode(b"", final=True))
        return token_count
