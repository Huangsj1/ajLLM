"""Train and register a tokenizer artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ajllm.artifacts.manifests import write_manifest
from ajllm.artifacts.paths import ArtifactPaths
from ajllm.config.loader import LoadedConfig
from ajllm.config.validation import validate_dataset, validate_tokenizer
from ajllm.tokenization.bpe_trainer import train_bpe
from ajllm.tokenization.serialization import save_vocab_and_merges


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def run(config: LoadedConfig) -> dict[str, Any]:
    data = config.data
    dataset = data["dataset"]
    tokenizer_config = data["tokenizer"]
    validate_dataset(dataset)
    validate_tokenizer(tokenizer_config)

    split = str(data.get("split", "train"))
    if split not in dataset["splits"]:
        raise ValueError(f"Dataset has no '{split}' split")
    input_path = _project_path(config.project_root, dataset["splits"][split])
    output_dir = ArtifactPaths(config.project_root).tokenizer(tokenizer_config["name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"

    special_tokens = list(tokenizer_config.get("special_tokens", []))
    pretokenization = tokenizer_config.get("pretokenization", {})
    vocab, merges = train_bpe(
        input_path,
        int(tokenizer_config["vocab_size"]),
        special_tokens,
        parallel=bool(pretokenization.get("parallel", True)),
        num_processes=pretokenization.get("num_processes"),
        chunk_size_bytes=int(pretokenization.get("chunk_size_bytes", 8 * 1024 * 1024)),
    )
    save_vocab_and_merges(vocab, merges, vocab_path, merges_path)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(data, output_file, sort_keys=False, allow_unicode=True)

    manifest = {
        "name": tokenizer_config["name"],
        "dataset": dataset["name"],
        "source_split": split,
        "source_path": str(input_path),
        "source_size_bytes": input_path.stat().st_size,
        "vocab_size": len(vocab),
        "special_tokens": special_tokens,
        "merge_count": len(merges),
        "files": {"vocab": "vocab.json", "merges": "merges.txt"},
    }
    write_manifest(output_dir / "manifest.json", "tokenizer", manifest)
    return {"output_dir": str(output_dir), **manifest}
