"""Command-line entry point for MiniMind-format supervised fine-tuning."""

from __future__ import annotations

import argparse

from ajllm.workflows.causal_lm import _upgrade_legacy_top1_moe_state_dict, run_sft

run = run_sft

__all__ = ["_upgrade_legacy_top1_moe_state_dict", "run"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised fine-tune ajLLM on MiniMind conversation JSONL")
    parser.add_argument("--config", required=True, help="Training YAML path")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
