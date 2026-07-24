"""Centralized construction of artifact and run paths."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def _run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid4().hex[:8]}"


@dataclass(frozen=True)
class ArtifactPaths:
    project_root: Path

    def tokenizer(self, name: str) -> Path:
        return self.project_root / "artifacts" / "tokenizers" / name

    def encoded(self, dataset: str, tokenizer: str) -> Path:
        return self.project_root / "artifacts" / "encoded" / dataset / tokenizer

    def training_run(
        self,
        dataset: str,
        tokenizer: str,
        model: str,
        experiment: str,
        run_id: str | None = None,
    ) -> Path:
        return (
            self.project_root / "runs" / "training" / dataset / tokenizer / model / experiment / (run_id or _run_id())
        )

    def task_run(self, task: str, name: str, run_id: str | None = None) -> Path:
        return self.project_root / "runs" / task / name / (run_id or _run_id())
