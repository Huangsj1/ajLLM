"""Compare metrics from an explicit list of completed runs."""

from __future__ import annotations

import json
from typing import Any

from ajllm.artifacts.manifests import write_manifest
from ajllm.artifacts.paths import ArtifactPaths
from ajllm.config.loader import LoadedConfig
from ajllm.evaluation.comparison import compare_runs, write_comparison_csv
from ajllm.evaluation.plotting import plot_learning_curves
from ajllm.workflows.common import project_path


def run(config: LoadedConfig) -> dict[str, Any]:
    data = config.data
    name = str(data.get("name", "comparison"))
    run_directories = [project_path(config.project_root, path) for path in data.get("runs", [])]
    if not run_directories:
        raise ValueError("Comparison config requires at least one run directory")
    metric = str(data.get("metric", "validation_loss"))
    mode = str(data.get("mode", "min"))
    results = compare_runs(run_directories, metric, mode)
    output_dir = ArtifactPaths(config.project_root).task_run("comparisons", name)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_comparison_csv(output_dir / "comparison.csv", results)
    if data.get("generate_plot", True):
        plot_learning_curves(run_directories, output_dir / "learning_curves.png")
    with (output_dir / "comparison.json").open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2, ensure_ascii=False)
    write_manifest(
        output_dir / "manifest.json",
        "run_comparison",
        {"name": name, "metric": metric, "mode": mode, "run_count": len(results), "best_run": results[0]},
    )
    return {"output_dir": str(output_dir), "results": results}
