from pathlib import Path

import torch

from ajllm.modeling.factory import build_model
from ajllm.training.checkpoint import load_checkpoint, save_checkpoint
from ajllm.training.losses import cross_entropy
from ajllm.training.optimizers import AdamW
from ajllm.training.schedulers import warmup_cosine_learning_rate


def test_cross_entropy_matches_torch() -> None:
    logits = torch.tensor([[[1.0, 2.0, 3.0], [3.0, -1.0, 0.5]]])
    targets = torch.tensor([[2, 0]])
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1))

    assert torch.allclose(cross_entropy(logits, targets), expected)


def test_warmup_cosine_schedule_boundaries() -> None:
    assert warmup_cosine_learning_rate(0, 1.0, 0.1, 2, 10) == 0.0
    assert warmup_cosine_learning_rate(2, 1.0, 0.1, 2, 10) == 1.0
    assert warmup_cosine_learning_rate(10, 1.0, 0.1, 2, 10) == 0.1


def test_checkpoint_restores_model_and_optimizer(tmp_path: Path) -> None:
    config = {
        "name": "test",
        "context_length": 4,
        "d_model": 8,
        "num_layers": 1,
        "num_heads": 2,
        "d_ff": 16,
    }
    model = build_model(config, vocab_size=20)
    optimizer = AdamW(model.parameters())
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = save_checkpoint(tmp_path / "model.pt", model, optimizer, 7, {"name": "test"})
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    checkpoint = load_checkpoint(path, model, optimizer)

    assert checkpoint["step"] == 7
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])
