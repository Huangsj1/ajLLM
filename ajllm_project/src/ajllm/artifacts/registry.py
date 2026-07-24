"""Resolve named artifacts and verify their manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ajllm.artifacts.manifests import read_manifest
from ajllm.artifacts.paths import ArtifactPaths


def load_tokenizer_artifact(project_root: Path, name: str) -> tuple[Path, dict[str, Any]]:
    directory = ArtifactPaths(project_root).tokenizer(name)
    return directory, read_manifest(directory / "manifest.json")


def load_encoded_artifact(project_root: Path, dataset: str, tokenizer: str) -> tuple[Path, dict[str, Any]]:
    directory = ArtifactPaths(project_root).encoded(dataset, tokenizer)
    return directory, read_manifest(directory / "manifest.json")
