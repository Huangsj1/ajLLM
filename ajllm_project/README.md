# ajLLM

An educational, compact LLM pre-training project. The current codebase focuses on one clean decoder-only baseline rather than a broad collection of partially connected workflows.

- Dense decoder: GQA, Q/K RMSNorm, RoPE, SwiGLU, tied embeddings (~62M parameters)
- Optional MoE decoder: four Top-1 SwiGLU experts (~197M total parameters)
- Custom Torch/Triton FlashAttention by default, with SDPA fallback
- Single-device training plus a readable FULL_SHARD FSDP implementation
- JSONL pre-training data and copied MiniMind 6,400-token BPE tokenizer

## Start here

```bash
uv sync --extra dev
uv run pytest
uv run python -m ajllm.workflows.pretrain --config configs/pretrain_dense.yaml
```

For FSDP on two GPUs, set `use_fsdp: true` in the training YAML and run:

```bash
uv run torchrun --nproc_per_node=2 \
  -m ajllm.workflows.pretrain --config configs/pretrain_config.yaml
```

The bundled configuration uses the copied official MiniMind tokenizer at `assets/tokenizers/minimind`. Its vocabulary and special-token IDs must remain paired with every model checkpoint.

## Layout

```text
src/ajllm/modeling/   canonical model implementation
src/ajllm/datasets/   pre-training data now; future SFT/DPO/etc. datasets
src/ajllm/training/   optimizer loop, checkpointing, educational FSDP
src/ajllm/workflows/  thin CLI entry point
src/ajllm/tokenization/ minimal MiniMind tokenizer adapter
configs/              dense, MoE, and pre-training YAML
docs/architecture.md  model and parameter design
docs/training.md      data, launch, FSDP, and resume guide
```

Train the MoE comparison with `configs/pretrain_moe.yaml`. Both runs emit isolated
JSONL metrics; `workflows.compare`, `workflows.evaluate`, and `workflows.generate`
provide loss plots, checkpoint evaluation, and prompt generation.

See [architecture](docs/architecture.md) for the model design, [training](docs/training.md) for the data and baseline training path, and [parallel training](docs/parallel_training.md) for FSDP, TP and EP launches, limits and validation.
