# ajLLM

An educational, compact LLM project with pre-training, supervised fine-tuning
(SFT), and Direct Preference Optimization (DPO) workflows built around one
clean decoder-only baseline, plus online Group Relative Policy Optimization
(GRPO) with a frozen external reward model.

- Dense decoder: GQA, Q/K RMSNorm, RoPE, SwiGLU, tied embeddings (~62M parameters)
- Optional MoE decoder: four Top-1 SwiGLU experts (~197M total parameters)
- Custom Torch/Triton FlashAttention by default, with SDPA fallback
- Single-device training plus a readable FULL_SHARD FSDP implementation
- JSONL pre-training data and copied MiniMind 6,400-token BPE tokenizer

## Start here

```bash
uv sync --extra dev
uv run pytest
uv run python -m ajllm.workflows.pretrain --config configs/pretrain/dense.yaml
# after pre-training, fine-tune on conversations
uv run python -m ajllm.workflows.sft --config configs/sft/dense.yaml
# then align preferred versus rejected answers
uv run python -m ajllm.workflows.dpo --config configs/dpo/dense.yaml
# or run online group-relative RLAIF after SFT
uv run python -m ajllm.workflows.grpo --config configs/grpo/dense.yaml
```

For FSDP on two GPUs, set `use_fsdp: true` in the training YAML and run:

```bash
uv run torchrun --nproc_per_node=2 \
  -m ajllm.workflows.pretrain --config configs/pretrain/dense.yaml
```

The bundled configuration uses the copied official MiniMind tokenizer at `assets/tokenizers/minimind`. Its vocabulary and special-token IDs must remain paired with every model checkpoint.

## Layout

```text
src/ajllm/modeling/   canonical model implementation
src/ajllm/datasets/   pre-training, SFT, and DPO datasets
src/ajllm/training/   optimizer loops, GRPO reward adapter, checkpointing, educational FSDP
src/ajllm/workflows/  thin CLI entry point
src/ajllm/tokenization/ minimal MiniMind tokenizer adapter
configs/model/        dense and MoE architecture YAML
configs/pretrain/     single-card and parallel pre-training YAML
configs/sft/          single-card and parallel SFT YAML
configs/dpo/          dense and MoE DPO YAML
configs/grpo/         dense and MoE GRPO YAML
docs/architecture.md  model and parameter design
docs/training.md      data, launch, FSDP, and resume guide
```

Train the MoE comparison with `configs/pretrain/moe.yaml`. Both runs emit isolated
JSONL metrics; `workflows.compare`, `workflows.evaluate`, and `workflows.generate`
provide loss plots, checkpoint evaluation, and prompt generation.

See [architecture](docs/architecture.md) for the model design, [training](docs/training.md) for the pre-training path, [SFT](docs/sft.md) for conversational fine-tuning, [DPO](docs/dpo.md) for preference alignment, [GRPO](docs/grpo.md) for online RLAIF, and [parallel training](docs/parallel_training.md) for FSDP, TP and EP launches, limits and validation.
