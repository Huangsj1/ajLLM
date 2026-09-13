"""Command-line entry point for Group Relative Policy Optimization after SFT."""

from __future__ import annotations

import argparse

from ajllm.workflows.causal_lm import run_grpo

run = run_grpo


def main() -> None:
    parser = argparse.ArgumentParser(description="Group Relative Policy Optimization (GRPO) for ajLLM")
    parser.add_argument("--config", required=True, help="Training YAML path")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
