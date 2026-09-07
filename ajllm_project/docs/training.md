# Pre-training and FSDP

## Scope and quick start

`python -m ajllm.workflows.pretrain` is the only training entry point. It accepts JSONL records shaped as `{"text": "a document"}`. The workflow only connects config, data, model and trainer; datasets live in `src/ajllm/datasets/` and optimization mechanics live in `src/ajllm/training/`.

```bash
uv sync --extra dev
uv run pytest
```

```bash
# pretrain a dense model on a single GPU
uv run python -m ajllm.workflows.pretrain --config configs/pretrain_dense.yaml

# pretrain a moe model on a single GPU
uv run python -m ajllm.workflows.pretrain --config configs/pretrain_moe.yaml
```

Use `device: cuda`; this pre-training path targets the project's CUDA + Triton environment. `batch_size` is per process. Effective batch size is `batch_size × gradient_accumulation_steps × world_size`.
Training is epoch-based: every epoch traverses the current rank's complete
dataloader once. The cosine schedule derives its total update count as
`epochs × ceil(batches_per_epoch / gradient_accumulation_steps)`. An optional
`max_steps` caps that total for a short smoke run while retaining the epoch loop.

AdamW uses defaults `betas=(0.9, 0.95)` and `weight_decay=0.1`. The project uses
its own AdamW update equations, stable log-sum-exp cross entropy, SiLU, and
global-gradient clipping rather than PyTorch's ready-made optimizer/loss/activation
helpers. The learning rate warms up linearly then cosine-decays to `min_lr`.
`bf16` uses CUDA autocast, and `fp16` additionally uses GradScaler.

Model YAML controls the implementation explicitly: `use_flash_attention: true`
selects the project's Triton FlashAttention, and `use_cuda_kernels: true`
selects Triton RMSNorm, RoPE, SwiGLU, Cross Entropy, MoE, AdamW, and gradient
norm kernels. Set either switch to `false` to use the corresponding explicit
Torch equations. Backend details and CUDA-vs-Torch correctness tests are in
[CUDA kernel guide](cuda_kernels.md).

Rank 0 writes `metrics.jsonl`, `resolved_config.yaml`, and `summary.json` into `output_dir`.
A new run truncates that directory's old `metrics.jsonl` before its first metric;
a run with `resume_from` appends so its existing loss curve remains continuous.

## Dense/MoE experiment configuration

Use [pretrain_dense.yaml](../configs/pretrain_dense.yaml) first, then
[pretrain_moe.yaml](../configs/pretrain_moe.yaml). Their default output directories
are isolated as `output/pretrain/dense/` and `output/pretrain/moe/`, so runs,
checkpoints and metrics cannot overwrite each other. For a fair comparison, keep
tokenizer, data, `epochs`, optimizer values, seed, sequence length and effective
batch size the same.

| Setting | Meaning |
| --- | --- |
| `max_seq_len`, `batch_size`, `num_workers` | data-loader and micro-batch shape |
| `epochs`, `max_steps`, `gradient_accumulation_steps` | full passes, optional update cap, and effective batch |
| `learning_rate`, `min_lr`, `warmup_steps` | warmup + cosine schedule |
| `weight_decay`, `beta1`, `beta2`, `grad_clip` | custom AdamW and clipping |
| `mixed_precision` | `bf16`, `fp16`, or `null` |
| `log_interval`, `save_interval` | JSONL metrics and checkpoints |
| `validation_data_path`, `eval_interval`, `eval_batches` | optional fixed validation evaluation |
| `output_dir`, `resume_from` | run partitioning and continuation |
| `use_fsdp`, `fsdp_config` | multi-process custom FULL_SHARD mode |

The supplied Dense and MoE configurations both use `batch_size: 16` and
`gradient_accumulation_steps: 1`. This is the measured one-GPU limit for the
current hardware: Dense uses more than 8 GiB and MoE approaches 12 GiB. Keep
these equal when comparing the two architectures. Their default `epochs: 1`
visits all 1,270,238 records once; increase `epochs` only for another complete
corpus pass.

For a brief test without changing the data or batch size, set for example
`max_steps: 100`. The trainer stops after update 100, writes the final train
metric even if it is between logging intervals, runs the final configured
evaluation, saves `step_00000100.pt`, and writes `summary.json` with
`stopped_early: true`. Set `max_steps: null` to finish all configured epochs.

Every training metric has `lm_loss`, `auxiliary_loss`, and `total_loss`. Dense has
zero auxiliary loss; MoE's total includes its router balance loss. Compare
`lm_loss` when assessing next-token modelling quality, and retain `total_loss` to
inspect the actual optimized objective.

## Evaluate, compare, and generate

Evaluate a portable checkpoint on any JSONL pre-training split:

```bash
uv run python -m ajllm.workflows.evaluate \
  --checkpoint output/pretrain/dense/step_00010000.pt \
  --data-path data/pretrain_validation.jsonl --batch-size 16
```

It writes LM loss, MoE auxiliary loss, total loss, perplexity, and evaluated token
count alongside the checkpoint. Plot rank-zero training curves after both runs:

```bash
uv run python -m ajllm.workflows.compare \
  --runs output/pretrain/dense output/pretrain/moe \
  --output output/comparisons/dense_vs_moe.png

uv run python -m ajllm.workflows.compare \
  --runs output/pretrain/dense output/pretrain/moe \
  --event evaluation \
  --metric perplexity \
  --output output/comparisons/dense_vs_moe_ppl.png
```

Generate from a checkpoint using the paired MiniMind tokenizer:

```bash
uv run python -m ajllm.workflows.generate \
  --checkpoint output/pretrain/dense/step_00010000.pt \
  --prompt "人工智能的发展" \
  --max-new-tokens 128 \
  --temperature 0.8 --top-k 50 --top-p 0.9
```

Generation is deliberately a simple, explicit autoregressive loop without KV
cache. It supports greedy decoding (`temperature <= 0`) and temperature/top-k/top-p
sampling; KV cache belongs to a later inference extension.

## Tokenizer contract

Pre-training uses one tokenizer: the official MiniMind BPE copied from the local
reference project into `assets/tokenizers/minimind/`. The ajLLM wrapper uses the
small `tokenizers` runtime to read `tokenizer.json`; it does not require
Transformers and intentionally exposes only pre-training encode/decode and
special-token IDs.

```yaml
tokenizer:
  path: assets/tokenizers/minimind
```

The vocabulary size (6,400) and pad/BOS/EOS IDs (0/1/2) are inferred and
validated from those JSON files, preventing a silent embedding/tokenizer mismatch.
Do not change a checkpoint's tokenizer when resuming it.

## Educational FULL_SHARD FSDP

Launch one process per GPU:

```bash
uv run torchrun --nproc_per_node=2 \
  -m ajllm.workflows.pretrain --config configs/pretrain_config.yaml
```

```yaml
use_fsdp: true
fsdp_config:
  sharding_strategy: FULL_SHARD
  cpu_offload: false
  use_activation_checkpointing: true
```

`training/distributed.py` is a deliberately readable implementation rather than the framework's production FSDP wrapper. It works as follows:

```text
broadcast rank-0 initialization
  -> replace every Linear weight with a padded local flat shard
  -> all-gather a full weight immediately before that Linear
  -> run Linear and release the full reference
  -> reduce-scatter its full gradient to the matching local shard
  -> all-reduce gradients for replicated parameters
```

Activation checkpointing puts all-gather and Linear execution inside the recomputed function. It avoids keeping full-weight activations. The tied token embedding stays replicated because it is small at this scale and the tied LM head reads it after embedding lookup; this avoids an invalid released-weight path.

`full_state_dict()` and `load_full_state_dict()` are intentionally methods of the
custom FSDP wrapper because only it owns the shard sizes, padding, ranks, and
collective operations. The first all-gathers shards into ordinary unwrapped
parameter names for a portable checkpoint; the second slices that state back to
each rank while resuming. `checkpoint.py` uses an explicit
`isinstance(model, FullyShardedDataParallel)` branch, not attribute detection.

The implementation supports only `FULL_SHARD`, has no CPU offload, and requires a fixed world size for resume. The configuration validation rejects unsupported modes rather than pretending DDP-like options work.

## Checkpoints and resume

Each saved FSDP checkpoint has a portable unwrapped model and one optimizer state per rank:

```text
step_00005000.pt             full model + metadata
step_00005000.pt.rank0.optim rank-0 local optimizer shard
step_00005000.pt.rank1.optim rank-1 local optimizer shard
```

Normal training writes `step_00005000.pt.optim`. All ranks cooperatively all-gather model weights before rank 0 writes the portable model checkpoint. On resume, the full state is loaded into local parameter shards and each rank loads its optimizer shard. Add `resume_from: output/pretrain/step_00005000.pt` to the training YAML, retaining the same model YAML, tokenizer, and FSDP world size.

## Recommended progression

1. Run the CUDA tests, which include a one-step workflow smoke test.
2. Run a brief dense job on a subset of the new corpus and inspect metrics/checkpoints.
3. Scale sequence and batch size; use accumulation to preserve effective batch size.
4. Enable FSDP after the one-GPU loss curve is healthy, starting with two GPUs.
5. Try MoE only after dense training is established: it has different weight-memory and dispatch costs.
