"""Command-line interface for all ajLLM workflows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

from ajllm.config.loader import LoadedConfig, load_run_config
from ajllm.workflows import (
    lm_compare,
    lm_evaluate,
    lm_generate,
    lm_sweep,
    lm_train,
    model_report,
    tokenizer_encode,
    tokenizer_train,
)

Workflow = Callable[[LoadedConfig], dict[str, Any]]


def _add_config_command(subparsers, name: str, help_text: str, workflow: Workflow) -> None:
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument("--config", "-c", required=True, help="Path to a YAML configuration file")
    parser.set_defaults(workflow=workflow)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ajllm", description="Train tokenizers and Transformer language models")
    domains = parser.add_subparsers(dest="domain", required=True)

    tokenizer_parser = domains.add_parser("tokenizer", help="Tokenizer workflows")
    tokenizer_commands = tokenizer_parser.add_subparsers(dest="command", required=True)
    _add_config_command(tokenizer_commands, "train", "Train a byte-level BPE tokenizer", tokenizer_train.run)
    _add_config_command(tokenizer_commands, "encode", "Encode dataset splits to uint32", tokenizer_encode.run)

    lm_parser = domains.add_parser("lm", help="Language-model workflows")
    lm_commands = lm_parser.add_subparsers(dest="command", required=True)
    _add_config_command(lm_commands, "train", "Train a language model", lm_train.run)
    _add_config_command(lm_commands, "sweep", "Run a hyperparameter grid", lm_sweep.run)
    _add_config_command(lm_commands, "evaluate", "Evaluate loss and perplexity", lm_evaluate.run)
    _add_config_command(lm_commands, "compare", "Compare completed runs", lm_compare.run)
    _add_config_command(lm_commands, "generate", "Generate text", lm_generate.run)

    model_parser = domains.add_parser("model", help="Model inspection")
    model_commands = model_parser.add_subparsers(dest="command", required=True)
    _add_config_command(model_commands, "report", "Write a Markdown architecture report", model_report.run)
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        result = arguments.workflow(load_run_config(arguments.config))
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
