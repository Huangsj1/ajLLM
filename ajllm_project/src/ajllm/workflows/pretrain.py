"""Command-line entry point for JSONL causal-LM pre-training."""

from __future__ import annotations

import argparse

from ajllm.workflows.causal_lm import run_pretrain

run = run_pretrain


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train ajLLM on JSONL records containing a text field")
    parser.add_argument("--config", required=True, help="Training YAML path")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
