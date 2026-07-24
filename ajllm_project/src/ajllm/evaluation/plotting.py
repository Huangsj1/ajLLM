"""Learning-curve plotting for one or more training runs."""

from __future__ import annotations

import json
from pathlib import Path


def _load_metrics(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def plot_learning_curves(run_directories: list[str | Path], output_path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for directory in run_directories:
        run_path = Path(directory)
        metrics = _load_metrics(run_path / "metrics.jsonl")
        label = run_path.parent.name + "/" + run_path.name
        training = [record for record in metrics if record["event"] == "train"]
        evaluation = [record for record in metrics if record["event"] == "evaluation"]
        if training:
            axes[0].plot([record["step"] for record in training], [record["loss"] for record in training], label=label)
        if evaluation and "validation_loss" in evaluation[0]:
            axes[1].plot(
                [record["step"] for record in evaluation],
                [record["validation_loss"] for record in evaluation],
                marker="o",
                label=label,
            )
    axes[0].set(title="Training Loss", xlabel="Step", ylabel="Cross-Entropy")
    axes[1].set(title="Validation Loss", xlabel="Step", ylabel="Cross-Entropy")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination
