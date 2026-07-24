"""Read and write machine-readable artifact manifests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def write_manifest(path: str | Path, artifact_type: str, payload: dict[str, Any]) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "artifact_type": artifact_type,
        "created_at": datetime.now(UTC).isoformat(),
        **payload,
    }
    with manifest_path.open("w", encoding="utf-8") as output_file:
        json.dump(document, output_file, indent=2, ensure_ascii=False)
    return manifest_path


def read_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Artifact manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)
