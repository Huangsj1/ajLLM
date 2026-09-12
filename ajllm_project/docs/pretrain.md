# Pre-training

## 1. Objective

Pre-training teaches the decoder to predict the next token in a document. For a
token sequence \(x_1,\ldots,x_T\), the causal-LM objective is:

$$
\mathcal{L}_{\mathrm{PT}}(\theta) =
-\frac{1}{T-1}\sum_{t=1}^{T-1}\log
\pi_\theta(x_{t+1}\mid x_{\leq t}) + \mathcal{L}_{\mathrm{MoE\ aux}}.
$$

The dense model has \(\mathcal{L}_{\mathrm{MoE\ aux}}=0\). The Top-1 MoE
model adds its router load-balancing loss. Documents are packed as shifted
`input_ids` / `labels`; padding labels are ignored.

## 2. Dataset

The training schema is one JSON object per line with a `text` field. The
included `data/pretrain_t2t_mini.jsonl` has the same schema as the larger path
configured in `configs/pretrain/*.yaml`:

```json
{"text": "给我生成一首有关秋天的诗歌。秋日早晨，清风拂面。金色的叶子，似火在燃烧。"}
```

`PretrainDataset` adds BOS/EOS, crops or pads to `max_seq_len`, and uses every
valid next-token target.

## 3. Prerequisites

```bash
uv sync --extra dev
```

The tokenizer is included at `assets/tokenizers/minimind`. The default YAML
expects `data/pretrain_t2t.jsonl`; for a smoke run, point a copied YAML to
`data/pretrain_t2t_mini.jsonl` and set `max_steps: 100`.

## 4. Train

```bash
# Dense: output/pretrain/dense/step_00088217.pt
uv run python -m ajllm.workflows.pretrain --config configs/pretrain/dense.yaml

# Top-1 MoE: output/pretrain/moe/step_00088217.pt
uv run python -m ajllm.workflows.pretrain --config configs/pretrain/moe.yaml
```

To resume, set `resume_from` to a `step_<N>.pt` and keep the model, tokenizer,
data, batch, and parallel topology unchanged.

## 5. Plot

```bash
uv run python -m ajllm.workflows.plot_training --run output/pretrain/dense --smooth 20
uv run python -m ajllm.workflows.plot_training --run output/pretrain/moe --smooth 20
```

Both commands produce `<run>/training_curves.png` with `total_loss` versus
optimizer step.

## 6. Generate

Plain generation accepts either `--prompt` or a reusable UTF-8 prompt file:

```bash
uv run python -m ajllm.workflows.generate \
  --checkpoint output/pretrain/dense/step_00088217.pt \
  --prompt-file assets/prompt/continuation.txt

uv run python -m ajllm.workflows.generate \
  --checkpoint output/pretrain/moe/step_00088217.pt \
  --prompt "请解释什么是注意力机制。"
```

## 7. Multi-GPU

Pre-training ships TP, EP, and TP×EP examples. TP is for dense; EP and TP×EP
are for Top-1 MoE. FSDP is supported by setting `use_fsdp: true` in a copied
single-GPU YAML.

```bash
# 4 GPUs, dense TP
NCCL_IB_DISABLE=1 uv run torchrun --standalone --nproc_per_node=4 \
  -m ajllm.workflows.pretrain --config configs/pretrain/dense_tp4.yaml

# 4 GPUs, MoE EP
NCCL_IB_DISABLE=1 uv run torchrun --standalone --nproc_per_node=4 \
  -m ajllm.workflows.pretrain --config configs/pretrain/moe_ep4.yaml

# 8 GPUs, MoE TP×EP
NCCL_IB_DISABLE=1 uv run torchrun --standalone --nproc_per_node=8 \
  -m ajllm.workflows.pretrain --config configs/pretrain/moe_tp2_ep4.yaml
```

Read [parallel training](parallel_training.md) before changing world size or
resuming a parallel checkpoint.
