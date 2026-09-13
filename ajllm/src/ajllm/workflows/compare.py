"""Plot comparable train or validation metrics from pre-training run folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_metric(
    run_directories: list[str | Path], output_path: str | Path, metric: str = "total_loss", event: str = "train"
) -> Path:
    """Draw one metric from each rank-zero ``metrics.jsonl`` file."""
    figure, axis = plt.subplots(figsize=(8, 5))
    plotted = 0
    for run_directory in run_directories:
        run_directory = Path(run_directory)
        steps, values = [], []
        with (run_directory / "metrics.jsonl").open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                if record["event"] == event and metric in record:
                    steps.append(record["step"])
                    values.append(record[metric])
        if steps:
            axis.plot(steps, values, label=run_directory.name)
            plotted += 1
    if not plotted:
        raise ValueError(f"No '{metric}' values with event '{event}' were found")
    axis.set(xlabel="optimizer step", ylabel=metric, title=f"{event}: {metric}")
    axis.grid(alpha=0.25)
    axis.legend()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare metrics from dense/MoE pre-training runs")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories, e.g. output/pretrain/dense")
    parser.add_argument("--output", required=True, help="PNG output path")
    parser.add_argument(
        "--metric", default="total_loss", choices=["loss", "lm_loss", "auxiliary_loss", "total_loss", "perplexity"]
    )
    parser.add_argument("--event", default="train", choices=["train", "evaluation"])
    args = parser.parse_args()
    print(plot_metric(args.runs, args.output, args.metric, args.event))


if __name__ == "__main__":
    main()
