# Supervised fine-tuning (SFT)

## 1. Objective

SFT starts from pre-training and optimizes only assistant completion tokens.
For the assistant-token index set \(\mathcal{A}\), its masked causal-LM loss is:

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) =
-\frac{1}{|\mathcal{A}|}\sum_{t\in\mathcal{A}}
\log \pi_\theta(x_t\mid x_{<t}) + \mathcal{L}_{\mathrm{MoE\ aux}}.
$$

User prompts, role headers, and tool-response tokens have label `-100`; they
are context but not targets. MoE adds the same router auxiliary loss as in
pre-training.

## 2. Dataset

Each JSONL line contains a MiniMind conversation. The included
`data/sft_t2t_mini.jsonl` includes multi-turn conversations and optional
`reasoning_content`:

```json
{"conversations":[{"role":"user","content":"你背后的模型是哪个版本？"},{"role":"assistant","content":"我是由jingyaogong开发的高效小参数AI模型。"}]}
```

The serializer supports system messages, thinking blocks, tool schemas, tool
calls, and tool responses. Long records retain the latest `max_seq_len + 1`
serialized tokens so the final answer remains a target.

## 3. Prerequisites

```bash
uv sync --extra dev
```

SFT requires matching completed pre-training checkpoints:
`output/pretrain/dense/step_00088217.pt` and
`output/pretrain/moe/step_00088217.pt`. The default YAML expects
`data/sft_t2t.jsonl`; use `data/sft_t2t_mini.jsonl` in a copied YAML for a
short smoke run.

## 4. Train

```bash
# Dense: output/sft/dense/step_00100000.pt
uv run python -m ajllm.workflows.sft --config configs/sft/dense.yaml

# Top-1 MoE: output/sft/moe/step_00159670.pt
uv run python -m ajllm.workflows.sft --config configs/sft/moe.yaml
```

`pretrained_checkpoint` initializes weights only. To resume SFT, set it to
`null` and set `resume_from` to an SFT checkpoint.

## 5. Plot

```bash
uv run python -m ajllm.workflows.plot_training --run output/sft/dense --smooth 20
uv run python -m ajllm.workflows.plot_training --run output/sft/moe --smooth 20
```

## 6. Generate

Chat generation uses the same serialization as SFT. The JSON template below is
ready to use; plain continuation is also available for both models.

```bash
uv run python -m ajllm.workflows.generate \
  --checkpoint output/sft/dense/step_00100000.pt \
  --messages-json assets/prompt/chat_messages.json --open-thinking

uv run python -m ajllm.workflows.generate \
  --checkpoint output/sft/moe/step_00159670.pt \
  --messages-json assets/prompt/chat_messages.json --open-thinking

uv run python -m ajllm.workflows.generate \
  --checkpoint output/sft/dense/step_00100000.pt \
  --prompt-file assets/prompt/continuation.txt
```

## 7. Multi-GPU

SFT ships FSDP, dense TP, MoE EP, and MoE TP×EP configurations:

```bash
# 2 GPUs, dense FULL_SHARD FSDP
NCCL_IB_DISABLE=1 uv run torchrun --standalone --nproc_per_node=2 \
  -m ajllm.workflows.sft --config configs/sft/dense_fsdp2.yaml

# 4 GPUs, dense TP
NCCL_IB_DISABLE=1 uv run torchrun --standalone --nproc_per_node=4 \
  -m ajllm.workflows.sft --config configs/sft/dense_tp4.yaml

# 4 GPUs, MoE EP; 8 GPUs, MoE TP×EP
NCCL_IB_DISABLE=1 uv run torchrun --standalone --nproc_per_node=4 \
  -m ajllm.workflows.sft --config configs/sft/moe_ep4.yaml
NCCL_IB_DISABLE=1 uv run torchrun --standalone --nproc_per_node=8 \
  -m ajllm.workflows.sft --config configs/sft/moe_tp2_ep4.yaml
```

See [parallel training](parallel_training.md) for topology and resume rules.
