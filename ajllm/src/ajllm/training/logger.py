"""Append-only JSONL metric logging."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class RunLogger:
    """Write metrics for one fresh run, or append only when resuming it."""

    def __init__(self, run_directory: str | Path, append_existing: bool = False) -> None:
        self.run_directory = Path(run_directory)
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_directory / "metrics.jsonl"
        if not append_existing:
            # Start a new run without inheriting curves from an older run that
            # happened to use the same output directory.
            self.metrics_path.write_text("", encoding="utf-8")
        self.started_at = time.perf_counter()

    def log(self, step: int, event: str, values: dict[str, Any]) -> None:
        record = {
            "step": step,
            "event": event,
            "wall_time_seconds": time.perf_counter() - self.started_at,
            "timestamp": datetime.now(UTC).isoformat(),
            **values,
        }
        with self.metrics_path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
