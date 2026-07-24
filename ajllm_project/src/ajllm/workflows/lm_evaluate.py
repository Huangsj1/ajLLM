"""Evaluate a saved training run on an encoded dataset split."""

from __future__ import annotations

from typing import Any

from ajllm.artifacts.manifests import write_manifest
from ajllm.artifacts.paths import ArtifactPaths
from ajllm.artifacts.registry import load_encoded_artifact
from ajllm.config.loader import LoadedConfig
from ajllm.evaluation.evaluator import evaluate_model
from ajllm.training.data import TokenDataset
from ajllm.utils.device import resolve_device
from ajllm.workflows.common import load_model_from_run, project_path


def run(config: LoadedConfig) -> dict[str, Any]:
    data = config.data
    name = str(data.get("name", "evaluation"))
    run_directory = project_path(config.project_root, data["run_directory"])
    device = resolve_device(str(data.get("device", "auto")))
    model, training_config, _ = load_model_from_run(run_directory, str(data.get("checkpoint", "best")), device)
    dataset_name = training_config["dataset"]["name"]
    tokenizer_name = training_config["tokenizer"]["name"]
    encoded_dir, encoded_manifest = load_encoded_artifact(config.project_root, dataset_name, tokenizer_name)
    split = str(data.get("split", "validation"))
    dataset = TokenDataset(encoded_dir / encoded_manifest["splits"][split]["path"])
    results = evaluate_model(
        model,
        dataset,
        int(data.get("batches", 100)),
        int(data.get("batch_size", training_config.get("training", {}).get("batch_size", 32))),
        int(training_config["model"]["context_length"]),
        device,
        int(data.get("seed", 42)),
    )
    output_dir = ArtifactPaths(config.project_root).task_run("evaluations", name)
    payload = {"training_run": str(run_directory), "split": split, **results}
    write_manifest(output_dir / "result.json", "evaluation", payload)
    return {"output_dir": str(output_dir), **payload}
