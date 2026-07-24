"""Expand and execute a grid of training configurations."""

from __future__ import annotations

import copy
import itertools
import json
from typing import Any

from ajllm.artifacts.manifests import write_manifest
from ajllm.artifacts.paths import ArtifactPaths
from ajllm.config.loader import LoadedConfig, set_dotted_value
from ajllm.evaluation.comparison import compare_runs, write_comparison_csv
from ajllm.evaluation.plotting import plot_learning_curves
from ajllm.workflows import lm_train


def _trial_name(keys: list[str], values: tuple[Any, ...]) -> str:
    parts = []
    for key, value in zip(keys, values, strict=True):
        safe_value = str(value).replace(".", "p").replace("+", "").replace("-", "m")
        parts.append(f"{key.split('.')[-1]}_{safe_value}")
    return "__".join(parts)


def run(config: LoadedConfig) -> dict[str, Any]:
    data = config.data
    name = str(data.get("name", "sweep"))
    grid = data.get("grid")
    if not isinstance(grid, dict) or not grid:
        raise ValueError("Sweep config requires a non-empty grid mapping")
    keys = list(grid)
    value_lists = [values if isinstance(values, list) else [values] for values in grid.values()]
    output_dir = ArtifactPaths(config.project_root).task_run("sweeps", name)
    trials_dir = output_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    trial_records = []
    successful_runs: list[str] = []

    for values in itertools.product(*value_lists):
        trial_name = _trial_name(keys, values)
        trial_data = copy.deepcopy(data)
        trial_data.pop("grid", None)
        trial_data.pop("name", None)
        trial_data["experiment"] = f"{name}_{trial_name}"
        for key, value in zip(keys, values, strict=True):
            set_dotted_value(trial_data, key, value)
        trial_config = LoadedConfig(trial_data, config.source, config.project_root, config.config_root)
        trial_directory = trials_dir / trial_name
        try:
            result = lm_train.run(trial_config, trial_directory)
            successful_runs.append(result["run_directory"])
            trial_records.append({"name": trial_name, "overrides": dict(zip(keys, values, strict=True)), **result})
        except Exception as error:
            trial_records.append(
                {
                    "name": trial_name,
                    "overrides": dict(zip(keys, values, strict=True)),
                    "status": "failed",
                    "error": str(error),
                }
            )
            if not data.get("continue_on_error", False):
                raise

    comparison_config = data.get("comparison", {})
    metric = str(comparison_config.get("metric", "validation_loss"))
    mode = str(comparison_config.get("mode", "min"))
    comparison = compare_runs(successful_runs, metric, mode) if successful_runs else []
    write_comparison_csv(output_dir / "comparison.csv", comparison)
    if successful_runs and comparison_config.get("generate_plot", True):
        plot_learning_curves(successful_runs, output_dir / "learning_curves.png")
    with (output_dir / "trials.json").open("w", encoding="utf-8") as output_file:
        json.dump(trial_records, output_file, indent=2, ensure_ascii=False)
    best_run = comparison[0] if comparison else None
    write_manifest(
        output_dir / "manifest.json",
        "sweep",
        {
            "name": name,
            "metric": metric,
            "mode": mode,
            "trial_count": len(trial_records),
            "successful_trial_count": len(successful_runs),
            "best_run": best_run,
        },
    )
    return {"output_dir": str(output_dir), "trials": trial_records, "best_run": best_run}
