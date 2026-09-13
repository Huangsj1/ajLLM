"""Inspect a JSONL pre-training dataset without loading it into memory."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _count_physical_lines(path: Path, chunk_size: int = 8 * 1024 * 1024) -> int:
    """Count newline-delimited records using binary chunks instead of JSON parsing."""
    count = 0
    last_byte = b""
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    return count + int(bool(last_byte) and last_byte != b"\n")


def _read_examples(path: Path, num_examples: int) -> list[tuple[int, dict[str, Any]]]:
    """Read only enough JSONL lines to collect the requested valid examples."""
    if num_examples == 0:
        return []
    examples: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                examples.append((line_number, record))
            if len(examples) == num_examples:
                break
    return examples


def inspect_jsonl(
    file_path: str | Path, num_examples: int = 3, pretty: bool = True, detailed: bool = False
) -> None:
    """Print JSONL line count and examples; validate every record only in detailed mode."""
    if num_examples < 0:
        raise ValueError("num_examples must be non-negative")

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file was not found: {path.resolve()}")

    file_size_mib = path.stat().st_size / (1024 * 1024)
    print("Dataset summary")
    print(f"  Path: {path.resolve()}")
    print(f"  File size: {file_size_mib:.2f} MiB")
    print(f"  Physical JSONL records: {_count_physical_lines(path):,}")
    print("  Count mode: fast binary newline count (one line must be one record)")

    examples = _read_examples(path, num_examples)
    if examples:
        print(f"  Fields in first valid record: {', '.join(sorted(examples[0][1]))}")

    if not detailed:
        _print_examples(examples, pretty)
        return

    field_counts: Counter[str] = Counter()
    record_count = empty_line_count = invalid_json_count = 0
    text_count = text_total_length = 0
    text_min_length: int | None = None
    text_max_length = 0

    # Detailed mode parses every line to validate the JSONL contract and text fields.
    with path.open(encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                empty_line_count += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_json_count += 1
                continue
            if not isinstance(record, dict):
                invalid_json_count += 1
                continue
            record_count += 1
            field_counts.update(record)
            text = record.get("text")
            if isinstance(text, str):
                text_length = len(text)
                text_count += 1
                text_total_length += text_length
                text_min_length = text_length if text_min_length is None else min(text_min_length, text_length)
                text_max_length = max(text_max_length, text_length)

    print(f"  Valid JSON objects: {record_count:,}")
    print(f"  Empty lines: {empty_line_count:,}")
    print(f"  Invalid or non-object JSON lines: {invalid_json_count:,}")
    print(f"  All fields: {', '.join(sorted(field_counts)) or '(none)'}")
    if text_count:
        print(
            "  Text field lengths (characters): "
            f"count={text_count:,}, min={text_min_length}, "
            f"mean={text_total_length / text_count:.1f}, max={text_max_length}"
        )
    else:
        print("  Text field lengths (characters): no string 'text' fields found")

    _print_examples(examples, pretty)


def _print_examples(examples: list[tuple[int, dict[str, Any]]], pretty: bool) -> None:
    """Print already-read JSON examples without touching the dataset again."""

    print(f"\nFirst {len(examples)} valid record(s)")
    print("-" * 72)
    for example_index, (line_number, record) in enumerate(examples, start=1):
        print(f"[{example_index}] JSONL line {line_number}")
        print(json.dumps(record, ensure_ascii=False, indent=2) if pretty else record)
        print("-" * 72)


def preview_jsonl(file_path: str | Path, n: int = 3, pretty: bool = True) -> None:
    """Backward-compatible name for :func:`inspect_jsonl`."""
    inspect_jsonl(file_path, num_examples=n, pretty=pretty)


def main() -> None:
    """Run the JSONL inspector from the command line."""
    parser = argparse.ArgumentParser(description="Print JSONL dataset statistics and example records")
    parser.add_argument("file_path", help="Path to a JSONL dataset")
    parser.add_argument("--num-examples", type=int, default=3, help="Number of valid records to print")
    parser.add_argument("--compact", action="store_true", help="Print each example on one line")
    parser.add_argument(
        "--detailed", action="store_true", help="Parse all JSON records for validation and text statistics"
    )
    args = parser.parse_args()
    inspect_jsonl(args.file_path, num_examples=args.num_examples, pretty=not args.compact, detailed=args.detailed)


if __name__ == "__main__":
    main()
