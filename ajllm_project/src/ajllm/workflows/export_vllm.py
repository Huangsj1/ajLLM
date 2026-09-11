"""Export a dense ajLLM SFT checkpoint to a Qwen3-compatible vLLM directory."""

from __future__ import annotations

import argparse

from ajllm.utils.vllm_export import export_dense_checkpoint_to_vllm, validate_qwen3_export_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dense ajLLM checkpoint for vLLM GRPO rollouts")
    parser.add_argument("--checkpoint", required=True, help="Dense portable SFT/GRPO .pt checkpoint")
    parser.add_argument("--tokenizer", required=True, help="MiniMind tokenizer directory paired with the checkpoint")
    parser.add_argument("--output", required=True, help="New or empty output HF/vLLM model directory")
    args = parser.parse_args()
    output = export_dense_checkpoint_to_vllm(args.checkpoint, args.tokenizer, args.output)
    validate_qwen3_export_directory(output)
    print(output)


if __name__ == "__main__":
    main()
