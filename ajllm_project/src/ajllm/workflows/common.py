"""Shared helpers for workflows that consume completed training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from ajllm.modeling.factory import build_model
from ajllm.training.checkpoint import load_checkpoint


def project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def load_run_configuration(run_directory: Path) -> dict[str, Any]:
    with (run_directory / "resolved_config.yaml").open("r", encoding="utf-8") as input_file:
        return yaml.safe_load(input_file)


def select_checkpoint(run_directory: Path, selector: str) -> Path:
    candidate = Path(selector)
    if candidate.suffix == ".pt" and candidate.is_file():
        return candidate
    aliases = {
        "best": run_directory / "checkpoints" / "best.pt",
        "latest": run_directory / "checkpoints" / "latest.pt",
        "final": run_directory / "checkpoints" / "final.pt",
    }
    if selector not in aliases:
        candidate = run_directory / selector
    else:
        candidate = aliases[selector]
    if not candidate.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {candidate}")
    return candidate


def load_model_from_run(
    run_directory: Path,
    checkpoint_selector: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    run_config = load_run_configuration(run_directory)
    vocab_size = int(run_config["_runtime"]["vocab_size"])
    model = build_model(run_config["model"], vocab_size).to(device)
    checkpoint = load_checkpoint(select_checkpoint(run_directory, checkpoint_selector), model, map_location=device)
    model.eval()
    return model, run_config, checkpoint
