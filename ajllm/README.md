# ajLLM

ajLLM is an educational compact decoder-only LLM project. One implementation
supports a 62M dense decoder and a four-expert Top-1 MoE decoder, then carries
the same model through four stages: pre-training, supervised fine-tuning (SFT),
Direct Preference Optimization (DPO), and online Group Relative Policy
Optimization (GRPO).

The project keeps the important pieces readable: MiniMind BPE tokenization,
GQA, Q/K RMSNorm, RoPE, SwiGLU, custom Torch/Triton kernels, portable
checkpoints, JSONL metrics, dense/MoE configurations, and optional educational
FSDP, TP, EP, and TP×EP wrappers.

## Project layout

```text
assets/
  prompt/                       reusable text and chat-generation templates
  tokenizers/minimind/          paired MiniMind tokenizer JSON and special-token config
configs/
  model/{dense,moe}.yaml        architecture: dimensions, heads, experts, kernels
  pretrain/                     dense/MoE single-GPU and TP/EP/TP×EP examples
  sft/                          dense/MoE single-GPU and FSDP/TP/EP/TP×EP examples
  dpo/{dense,moe}.yaml          preference-alignment runs from SFT checkpoints
  grpo/{dense,moe}.yaml         online RLAIF runs from SFT checkpoints
data/
  pretrain_t2t*.jsonl           causal-LM text records (the *_mini file is included)
  sft_t2t*.jsonl                MiniMind conversations (the *_mini file is included)
  dpo.jsonl                     chosen/rejected preference pairs
  rlaif.jsonl                   conversations ending in an empty assistant response
docs/
  pretrain.md, sft.md, dpo.md, grpo.md  stage guides: objective to generation
  architecture.md                model, attention, dense, and MoE design
  parallel_training.md           FSDP/TP/EP/TP×EP mechanics and resume rules
  cuda_kernels.md                Triton/Torch kernel details and validation
output/
  <stage>/<dense|moe>/           checkpoints, metrics.jsonl, resolved config, plots
reward_model/                    local Skywork reward-model download location (not tracked)
src/ajllm/
  datasets/                      JSONL readers, chat serialization, collators
  modeling/                      TransformerLM, attention, layers, SwiGLU, MoE
  tokenization/                  minimal MiniMind tokenizer adapter
  training/                      trainers, losses, reward adapter, checkpointing
  training/parallel/             educational FSDP, tensor/expert parallel wrappers
  utils/                         vLLM export/sync and diagnostic utilities
  workflows/                     CLI entry points: train, generate, evaluate, plot
tests/                           CPU, CUDA, workflow, and distributed correctness coverage
```

## Quick start

Use Python 3.12 or 3.13 on Linux x86_64. Install the locked environment and run
the fast local checks:

```bash
uv sync --extra dev
uv run pytest tests/test_model_architecture.py tests/test_plot_training.py
```

Every checkpoint must stay paired with `assets/tokenizers/minimind`; do not
replace its vocabulary or special-token IDs. Review the YAML in `configs/`
before changing a run. Reusable input templates are
[continuation.txt](assets/prompt/continuation.txt) and
[chat_messages.json](assets/prompt/chat_messages.json).

## Training guides

Each guide follows the same order: objective, data, prerequisites, dense/MoE
training, plotting, generation, and supported multi-GPU launch.

| Stage | Purpose | Guide |
| --- | --- | --- |
| Pre-training | Learn next-token prediction from text | [Pre-training](docs/pretrain.md) |
| SFT | Learn assistant responses from conversations | [SFT](docs/sft.md) |
| DPO | Prefer chosen responses over rejected ones | [DPO](docs/dpo.md) |
| GRPO | Online RLAIF with grouped rollouts and Skywork reward | [GRPO](docs/grpo.md) |

For model internals see [architecture](docs/architecture.md); for parallel
topology, checkpoint, and resume constraints see
[parallel training](docs/parallel_training.md).
