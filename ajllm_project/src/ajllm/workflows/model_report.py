"""Generate a standalone report for a model component configuration."""

from __future__ import annotations

from typing import Any

from ajllm.config.loader import LoadedConfig
from ajllm.modeling.factory import build_model
from ajllm.reporting.model_report import write_model_report


def run(config: LoadedConfig) -> dict[str, Any]:
    data = config.data
    model_config = data.get("model", data)
    vocab_size = int(data.get("report_vocab_size", model_config.get("report_vocab_size", 10_000)))
    model = build_model(model_config, vocab_size)
    report_batch_size = int(data.get("report_batch_size", model_config.get("report_batch_size", 1)))
    report_training = dict(data.get("training", {}))
    if "report_training_tokens" in data and "total_tokens" not in report_training:
        report_training["total_tokens"] = int(data["report_training_tokens"])
    if "report_training_tokens" in model_config and "total_tokens" not in report_training:
        report_training["total_tokens"] = int(model_config["report_training_tokens"])
    report_training.setdefault("batch_size", report_batch_size)
    output_path = config.project_root / "reports" / "models" / f"{model_config['name']}.md"
    write_model_report(output_path, model, model_config, vocab_size, report_batch_size, report_training)
    return {
        "output_path": str(output_path),
        "model": model_config["name"],
        "vocab_size": vocab_size,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
