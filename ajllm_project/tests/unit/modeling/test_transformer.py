import pytest
import torch

from ajllm.modeling.factory import build_model


@pytest.mark.parametrize(
    ("position", "normalization", "placement", "feed_forward", "tie_embeddings"),
    [
        ("rope", "rmsnorm", "pre", "swiglu", False),
        ("none", "none", "pre", "silu", False),
        ("learned", "rmsnorm", "post", "swiglu", True),
    ],
)
def test_model_variants_produce_logits(
    position: str,
    normalization: str,
    placement: str,
    feed_forward: str,
    tie_embeddings: bool,
) -> None:
    config = {
        "name": "test",
        "context_length": 8,
        "d_model": 16,
        "num_layers": 2,
        "num_heads": 4,
        "d_ff": 32,
        "position_encoding": {"type": position, "theta": 10_000},
        "normalization": {"type": normalization, "placement": placement},
        "feed_forward": {"type": feed_forward},
        "tie_embeddings": tie_embeddings,
    }
    model = build_model(config, vocab_size=37)
    token_ids = torch.randint(0, 37, (3, 8))

    logits = model(token_ids)

    assert logits.shape == (3, 8, 37)
    assert torch.isfinite(logits).all()


def test_model_rejects_inputs_longer_than_context() -> None:
    config = {
        "name": "test",
        "context_length": 4,
        "d_model": 8,
        "num_layers": 1,
        "num_heads": 2,
        "d_ff": 16,
    }
    model = build_model(config, vocab_size=20)

    with pytest.raises(ValueError, match="exceeds context length"):
        model(torch.zeros((1, 5), dtype=torch.long))
