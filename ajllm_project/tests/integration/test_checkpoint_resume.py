from array import array
from pathlib import Path

import torch

from ajllm.modeling.factory import build_model
from ajllm.training.data import TokenDataset
from ajllm.training.trainer import Trainer


def _model_config() -> dict:
    return {
        "name": "resume_test",
        "context_length": 8,
        "d_model": 16,
        "num_layers": 1,
        "num_heads": 4,
        "d_ff": 32,
    }


def _training_config(max_steps: int, resume_from: str | None = None) -> dict:
    training = {"batch_size": 2, "max_steps": max_steps}
    if resume_from is not None:
        training["resume_from"] = resume_from
    return {
        "training": training,
        "optimizer": {"max_lr": 1e-3},
        "scheduler": {"warmup_steps": 0},
        "logging": {"log_interval": 1, "eval_interval": 1, "eval_batches": 1, "checkpoint_interval": 1},
    }


def test_training_resumes_from_saved_step(tmp_path: Path) -> None:
    token_path = tmp_path / "tokens.uint32"
    tokens = array("I", [index % 31 for index in range(256)])
    with token_path.open("wb") as output_file:
        tokens.tofile(output_file)
    dataset = TokenDataset(token_path)
    first_model = build_model(_model_config(), vocab_size=31)
    first_trainer = Trainer(
        first_model,
        dataset,
        dataset,
        tmp_path / "first",
        _training_config(1),
        torch.device("cpu"),
        5,
    )
    first_result = first_trainer.train({"test": True})

    second_model = build_model(_model_config(), vocab_size=31)
    second_config = _training_config(2, first_result["checkpoint"])
    second_trainer = Trainer(second_model, dataset, dataset, tmp_path / "second", second_config, torch.device("cpu"), 5)
    second_result = second_trainer.train({"test": True})

    assert first_result["steps"] == 1
    assert second_result["steps"] == 2
    assert (tmp_path / "second" / "checkpoints" / "final.pt").is_file()
