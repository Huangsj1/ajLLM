"""Load and rank completed training runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_run_summary(run_directory: str | Path) -> dict[str, Any]:
    run_path = Path(run_directory)
    with (run_path / "summary.json").open("r", encoding="utf-8") as input_file:
        summary = json.load(input_file)
    return {"run_directory": str(run_path.resolve()), **summary}


def compare_runs(run_directories: list[str | Path], metric: str, mode: str = "min") -> list[dict[str, Any]]:
    if mode not in {"min", "max"}:
        raise ValueError("comparison mode must be min or max")
    results = [load_run_summary(run_directory) for run_directory in run_directories]
    return sorted(results, key=lambda result: float(result.get(metric, float("inf"))), reverse=mode == "max")


def write_comparison_csv(path: str | Path, results: list[dict[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for result in results for key in result})
    with destination.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return destination
