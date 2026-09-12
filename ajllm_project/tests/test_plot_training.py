"""Tests for the stage-independent training-curve workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ajllm.workflows.plot_training import plot_training_curves


def _write_metrics(directory: Path, records: list[dict]) -> None:
    directory.mkdir()
    (directory / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_plot_training_curves_uses_only_loss_for_supervised_stages(tmp_path: Path) -> None:
    run = tmp_path / "sft"
    _write_metrics(
        run,
        [
            {"step": 1, "event": "train", "total_loss": 2.0},
            {"step": 2, "event": "train", "total_loss": 1.5},
            {"step": 2, "event": "evaluation", "total_loss": 1.4},
        ],
    )
    output = plot_training_curves(run, smooth=2)
    assert output == run / "training_curves.png"
    assert output.is_file() and output.stat().st_size > 0


def test_plot_training_curves_adds_grpo_reward_and_kl(tmp_path: Path) -> None:
    run = tmp_path / "grpo"
    _write_metrics(
        run,
        [
            {"step": 1, "event": "train", "total_loss": -0.1, "reward": -1.0, "kl": 0.0},
            {"step": 2, "event": "train", "total_loss": -0.2, "reward": 0.5, "kl": 0.03},
        ],
    )
    output = plot_training_curves(run, tmp_path / "grpo.png")
    assert output.is_file() and output.stat().st_size > 0


def test_plot_training_curves_rejects_missing_loss(tmp_path: Path) -> None:
    run = tmp_path / "broken"
    _write_metrics(run, [{"step": 1, "event": "train", "reward": 1.0}])
    with pytest.raises(ValueError, match="loss"):
        plot_training_curves(run)
