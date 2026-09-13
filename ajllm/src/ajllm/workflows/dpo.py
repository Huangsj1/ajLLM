"""Command-line entry point for Direct Preference Optimization after SFT."""

from __future__ import annotations

import argparse

from ajllm.workflows.causal_lm import run_dpo

run = run_dpo


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct Preference Optimization (DPO) for ajLLM")
    parser.add_argument("--config", required=True, help="Training YAML path")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
