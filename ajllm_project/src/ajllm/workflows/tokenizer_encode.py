"""Encode dataset splits and register the resulting token files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ajllm.artifacts.manifests import write_manifest
from ajllm.artifacts.paths import ArtifactPaths
from ajllm.artifacts.registry import load_tokenizer_artifact
from ajllm.config.loader import LoadedConfig
from ajllm.config.validation import validate_dataset, validate_tokenizer
from ajllm.tokenization.tokenizer import Tokenizer
from ajllm.utils.hashing import sha256_file


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def run(config: LoadedConfig) -> dict[str, Any]:
    data = config.data
    dataset = data["dataset"]
    tokenizer_config = data["tokenizer"]
    validate_dataset(dataset)
    validate_tokenizer(tokenizer_config)

    tokenizer_dir, tokenizer_manifest = load_tokenizer_artifact(config.project_root, tokenizer_config["name"])
    tokenizer = Tokenizer.from_files(
        tokenizer_dir / tokenizer_manifest["files"]["vocab"],
        tokenizer_dir / tokenizer_manifest["files"]["merges"],
        tokenizer_manifest.get("special_tokens", []),
    )
    requested_splits = data.get("splits", list(dataset["splits"]))
    output_dir = ArtifactPaths(config.project_root).encoded(dataset["name"], tokenizer_config["name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded_files: dict[str, Any] = {}
    encoding = data.get("encoding", {})

    for split in requested_splits:
        if split not in dataset["splits"]:
            raise ValueError(f"Dataset has no '{split}' split")
        input_path = _project_path(config.project_root, dataset["splits"][split])
        output_path = output_dir / f"{split}.uint32"
        token_count = tokenizer.encode_file_to_disk(
            input_path,
            output_path,
            chunk_size=int(encoding.get("chunk_size", 16 * 1024 * 1024)),
            parallel=bool(encoding.get("parallel", True)),
            num_processes=encoding.get("num_processes"),
            buffer_tokens=int(encoding.get("buffer_tokens", 65_536)),
        )
        encoded_files[split] = {
            "path": output_path.name,
            "token_count": token_count,
            "size_bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "source_path": str(input_path),
        }

    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(data, output_file, sort_keys=False, allow_unicode=True)
    manifest = {
        "dataset": dataset["name"],
        "tokenizer": tokenizer_config["name"],
        "tokenizer_fingerprint": tokenizer_manifest["fingerprint"],
        "vocab_size": tokenizer_manifest["vocab_size"],
        "dtype": "uint32",
        "byte_order": "native",
        "splits": encoded_files,
    }
    write_manifest(output_dir / "manifest.json", "encoded_dataset", manifest)
    return {"output_dir": str(output_dir), **manifest}
