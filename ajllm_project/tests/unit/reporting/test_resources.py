from pathlib import Path

from ajllm.modeling.factory import build_model
from ajllm.reporting.model_report import write_model_report
from ajllm.reporting.resource_estimates import estimate_resources


def _config() -> dict:
    return {
        "name": "resource_test",
        "context_length": 8,
        "d_model": 16,
        "num_layers": 2,
        "num_heads": 4,
        "d_ff": 32,
        "position_encoding": {"type": "rope"},
        "normalization": {"type": "rmsnorm", "placement": "pre"},
        "feed_forward": {"type": "swiglu"},
    }


def test_resource_estimate_contains_memory_and_flops() -> None:
    config = _config()
    model = build_model(config, vocab_size=40)
    estimate = estimate_resources(
        config | {"vocab_size": 40},
        sum(p.numel() for p in model.parameters()),
        4,
        {"max_steps": 10},
    )

    assert estimate.parameter_bytes > 0
    assert estimate.activation_bytes > 0
    assert estimate.gradient_bytes > 0
    assert estimate.optimizer_bytes == estimate.parameter_count * 8
    assert estimate.total_memory_bytes == (
        estimate.parameter_bytes + estimate.activation_bytes + estimate.gradient_bytes + estimate.optimizer_bytes
    )
    assert estimate.forward_flops_per_step > 0
    assert estimate.total_training_flops == estimate.training_flops_per_step * 10


def test_model_report_writes_resource_sections(tmp_path: Path) -> None:
    config = _config()
    model = build_model(config, vocab_size=40)
    report_path = write_model_report(
        tmp_path / "model.md",
        model,
        config,
        40,
        4,
        {"max_steps": 10},
    )

    report = report_path.read_text(encoding="utf-8")
    assert "Estimated Training Memory" in report
    assert "Estimated Matrix FLOPs" in report
    assert "AdamW first and second moments" in report
    assert "Total training FLOPs" in report
