"""Memory-bounded byte-level BPE training."""

from __future__ import annotations

import heapq
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from ajllm.tokenization.corpus import find_chunk_boundaries, iter_corpus_lines, iter_pretokens

ByteSequence = tuple[bytes, ...]
BytePair = tuple[bytes, bytes]


@dataclass(frozen=True)
class _DescendingPair:
    """Reverse pair ordering so heap ties prefer lexicographically larger pairs."""

    pair: BytePair

    def __lt__(self, other: _DescendingPair) -> bool:
        return self.pair > other.pair


def _merge_sequence(sequence: ByteSequence, pair: BytePair) -> ByteSequence:
    merged = pair[0] + pair[1]
    output: list[bytes] = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
            output.append(merged)
            index += 2
        else:
            output.append(sequence[index])
            index += 1
    return tuple(output)


def _count_pretokens_chunk(
    input_path: str | Path,
    start: int,
    end: int,
    special_tokens: list[str],
) -> Counter[ByteSequence]:
    """Count pre-tokens in one independently processable byte range."""

    with Path(input_path).open("rb") as input_file:
        input_file.seek(start)
        text = input_file.read(end - start).decode("utf-8", errors="ignore")
    counts: Counter[ByteSequence] = Counter()
    for pretoken in iter_pretokens(text, special_tokens, include_special=False):
        counts[tuple(bytes([byte_value]) for byte_value in pretoken.encode("utf-8"))] += 1
    return counts


def _count_pretokens(
    input_path: str | Path,
    special_tokens: list[str],
    parallel: bool = True,
    num_processes: int | None = None,
    chunk_size_bytes: int = 8 * 1024 * 1024,
) -> Counter[ByteSequence]:
    """Count pre-tokens serially or across document-aligned processes."""

    counts: Counter[ByteSequence] = Counter()
    input_path = Path(input_path)
    if not parallel or not special_tokens:
        for line in tqdm(iter_corpus_lines(str(input_path)), desc="Pre-tokenizing", unit="line"):
            for pretoken in iter_pretokens(line, special_tokens, include_special=False):
                counts[tuple(bytes([byte_value]) for byte_value in pretoken.encode("utf-8"))] += 1
        return counts

    if chunk_size_bytes < 1:
        raise ValueError("chunk_size_bytes must be positive")
    file_size = input_path.stat().st_size
    desired_chunks = max(1, (file_size + chunk_size_bytes - 1) // chunk_size_bytes)
    process_count = min(num_processes or (os.cpu_count() or 1), desired_chunks)
    with input_path.open("rb") as input_file:
        boundaries = find_chunk_boundaries(
            input_file,
            desired_chunks,
            max(special_tokens, key=len).encode("utf-8"),
        )
    ranges = list(zip(boundaries[:-1], boundaries[1:]))
    if len(ranges) <= 1 or process_count <= 1:
        for start, end in tqdm(ranges, desc="Pre-tokenizing", unit="chunk"):
            counts.update(_count_pretokens_chunk(input_path, start, end, special_tokens))
        return counts

    tasks = [(input_path, start, end, special_tokens) for start, end in ranges]
    with ProcessPoolExecutor(max_workers=process_count) as executor:
        futures = [executor.submit(_count_pretokens_chunk, *task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Pre-tokenizing", unit="chunk"):
            counts.update(future.result())
    return counts


def _unique_pairs(sequence: ByteSequence) -> set[BytePair]:
    return set(zip(sequence, sequence[1:]))


def _compute_merges(counts: Counter[ByteSequence], merge_count: int) -> list[BytePair]:
    pair_counts: Counter[BytePair] = Counter()
    pair_sequences: defaultdict[BytePair, set[ByteSequence]] = defaultdict(set)
    heap: list[tuple[int, _DescendingPair, BytePair]] = []

    def add_sequence(sequence: ByteSequence, frequency: int) -> None:
        for pair in zip(sequence, sequence[1:]):
            pair_counts[pair] += frequency
        for pair in _unique_pairs(sequence):
            pair_sequences[pair].add(sequence)

    def remove_sequence(sequence: ByteSequence, frequency: int) -> None:
        for pair in zip(sequence, sequence[1:]):
            pair_counts[pair] -= frequency
            if pair_counts[pair] <= 0:
                pair_counts.pop(pair, None)
        for pair in _unique_pairs(sequence):
            sequences = pair_sequences.get(pair)
            if sequences is not None:
                sequences.discard(sequence)
                if not sequences:
                    pair_sequences.pop(pair, None)

    # 1.first iteration: count pairs and sequences, populate heap
    for sequence, frequency in counts.items():
        add_sequence(sequence, frequency)
    for pair, frequency in pair_counts.items():
        heapq.heappush(heap, (-frequency, _DescendingPair(pair), pair))

    # 2. merge pairs until we reach the desired number of merges or run out of pairs
    merges: list[BytePair] = []
    progress = tqdm(total=merge_count, desc="Learning merges", unit="merge")
    while len(merges) < merge_count and pair_counts:
        while heap:
            negative_frequency, _, pair = heapq.heappop(heap)
            if pair_counts.get(pair) == -negative_frequency:
                break
        else:
            break

        # get all sequences affected by this merge
        affected = list(pair_sequences.get(pair, ()))
        if not affected:
            pair_counts.pop(pair, None)
            continue

        # for each affected sequence, remove it from counts and pair_counts, merge the pair
        updates: Counter[ByteSequence] = Counter()
        touched_pairs: set[BytePair] = set()
        for sequence in affected:
            frequency = counts.pop(sequence, 0)
            if frequency == 0:
                continue
            touched_pairs.update(_unique_pairs(sequence))
            remove_sequence(sequence, frequency)
            merged_sequence = _merge_sequence(sequence, pair)
            updates[merged_sequence] += frequency
            touched_pairs.update(_unique_pairs(merged_sequence))

        # update counts and pair_counts with the merged sequences
        for sequence, frequency in updates.items():
            counts[sequence] += frequency
            add_sequence(sequence, frequency)

        merges.append(pair)
        progress.update(1)
        for touched_pair in touched_pairs:
            frequency = pair_counts.get(touched_pair)
            if frequency:
                heapq.heappush(heap, (-frequency, _DescendingPair(touched_pair), touched_pair))
    progress.close()
    return merges


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    parallel: bool = True,
    num_processes: int | None = None,
    chunk_size_bytes: int = 8 * 1024 * 1024,
) -> tuple[dict[int, bytes], list[BytePair]]:
    """Train a byte-level BPE vocabulary with deterministic tie breaking."""

    special_tokens = list(special_tokens or [])
    minimum_size = 256 + len(special_tokens)
    if vocab_size < minimum_size:
        raise ValueError(f"vocab_size must be at least {minimum_size}")

    # 1. Count pre-tokens(tuple(bytes)) in the corpus.
    counts = _count_pretokens(
        input_path,
        special_tokens,
        parallel=parallel,
        num_processes=num_processes,
        chunk_size_bytes=chunk_size_bytes,
    )
    # 2. Compute merges and build the vocabulary.
    merges = _compute_merges(counts, vocab_size - minimum_size)
    vocab: dict[int, bytes] = {token_id: bytes([token_id]) for token_id in range(256)}
    next_token_id = 256
    for left, right in merges:
        vocab[next_token_id] = left + right
        next_token_id += 1
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1
    return vocab, merges
