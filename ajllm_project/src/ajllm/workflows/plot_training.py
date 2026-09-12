"""Plot the training metrics recorded by any ajLLM optimization workflow."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_LOSS_NAMES = ("total_loss", "loss", "dpo_loss", "policy_loss")
_GRPO_NAMES = ("reward", "kl")


def _read_train_metrics(run_directory: str | Path) -> list[dict[str, Any]]:
    metrics_path = Path(run_directory) / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"No metrics.jsonl found in {Path(run_directory)}")
    records: list[dict[str, Any]] = []
    with metrics_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {metrics_path}:{line_number}") from error
            if record.get("event") == "train":
                records.append(record)
    if not records:
        raise ValueError(f"No training metrics found in {metrics_path}")
    return records


def _first_available(records: Iterable[dict[str, Any]], names: tuple[str, ...]) -> str | None:
    return next((name for name in names if any(name in record for record in records)), None)


def _series(records: Iterable[dict[str, Any]], metric: str) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for record in records:
        step, value = record.get("step"), record.get(metric)
        if isinstance(step, (int, float)) and isinstance(value, (int, float)) and math.isfinite(value):
            points.append((step, value))
    if not points:
        raise ValueError(f"No finite '{metric}' values found in training metrics")
    steps, values = zip(*points, strict=True)
    return np.asarray(steps), np.asarray(values, dtype=float)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window == 1:
        return values
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    averaged = (cumulative[window:] - cumulative[:-window]) / window
    # Retain the same x-axis and use progressively shorter leading windows.
    leading = np.array([values[: index + 1].mean() for index in range(min(window - 1, len(values)))])
    return np.concatenate((leading, averaged))


def plot_training_curves(
    run_directory: str | Path,
    output_path: str | Path | None = None,
    *,
    smooth: int = 1,
) -> Path:
    """Write loss and, when present, GRPO reward/KL curves as one PNG.

    ``metrics.jsonl`` is intentionally the only input.  This works unchanged
    for pretrain, SFT, DPO, and GRPO and also keeps resumed runs continuous.
    """
    if smooth < 1:
        raise ValueError("smooth must be at least 1")
    run_directory = Path(run_directory)
    records = _read_train_metrics(run_directory)
    loss_name = _first_available(records, _LOSS_NAMES)
    if loss_name is None:
        raise ValueError("Training metrics contain no supported loss field")
    metric_names = [loss_name]
    metric_names.extend(name for name in _GRPO_NAMES if _first_available(records, (name,)) is not None)

    figure, axes = plt.subplots(len(metric_names), 1, figsize=(9, 3.5 * len(metric_names)), sharex=True)
    axes_array = np.atleast_1d(axes)
    for axis, metric_name in zip(axes_array, metric_names, strict=True):
        steps, values = _series(records, metric_name)
        axis.plot(steps, values, color="#8aa0b7", alpha=0.45, linewidth=1, label="logged")
        if smooth > 1 and len(values) > 1:
            axis.plot(steps, _moving_average(values, min(smooth, len(values))), color="#1769aa", linewidth=1.8,
                      label=f"moving average ({min(smooth, len(values))})")
            axis.legend()
        axis.set_ylabel(metric_name)
        axis.grid(alpha=0.25)
    axes_array[-1].set_xlabel("optimizer step")
    figure.suptitle(f"Training curves: {run_directory.name}")
    figure.tight_layout()
    destination = Path(output_path) if output_path is not None else run_directory / "training_curves.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ajLLM loss, and GRPO reward/KL when present")
    parser.add_argument("--run", required=True, help="Training output directory containing metrics.jsonl")
    parser.add_argument("--output", help="PNG path; defaults to <run>/training_curves.png")
    parser.add_argument("--smooth", type=int, default=1, help="Trailing moving-average window; 1 keeps raw values")
    args = parser.parse_args()
    print(plot_training_curves(args.run, args.output, smooth=args.smooth))


if __name__ == "__main__":
    main()
