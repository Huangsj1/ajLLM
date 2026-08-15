# FSDP with Activation Checkpointing

## Overview

Fully Sharded Data Parallel (FSDP) distributes parameter, gradient, and optimizer-state storage across ranks. Activation checkpointing (AC) keeps fewer forward activations and recomputes them during backward. Together, they reduce per-GPU memory use and make larger multi-GPU workloads feasible.

In ajLLM, enabling FSDP also enables activation checkpointing for the wrapped training modules. This is a memory-oriented path: it can improve end-to-end throughput when the saved memory allows a larger or more efficient workload, but it also adds recomputation and distributed communication.

## Implementation in ajLLM

The distributed training path is implemented behind the same YAML acceleration block used by the normal trainer.

Key properties:

- `use_fsdp: true` wraps the model for distributed training.
- Activation checkpointing is enabled as part of the FSDP path.
- Parameters, gradients, and optimizer state are sharded across participating ranks.
- FlashAttention-2 can be combined with FSDP by also setting `use_flash_attention: true`.
- Mixed precision is controlled through `acceleration.mixed_precision` and applies to the FSDP path.

The distributed launcher creates the required `torchrun` environment before invoking the normal LM training workflow.

## When to Use It

Use FSDP + activation checkpointing when:

- the target training configuration does not fit comfortably on one GPU;
- two or more GPUs are available;
- larger feasible batches improve GPU utilization or reduce total training time;
- you want to combine state sharding with FlashAttention-2 for additional attention-memory savings.

For small models or already memory-comfortable runs, FSDP + AC may not be faster than the simpler single-process path. Measure the target workload after changing batch size, precision, hardware, or context length.

## Configuration

FSDP + activation checkpointing only:

```yaml
acceleration:
  use_flash_attention: false
  use_fsdp: true
  mixed_precision: null
```

FlashAttention-2 + FSDP + activation checkpointing:

```yaml
acceleration:
  use_flash_attention: true
  use_fsdp: true
  mixed_precision: null
```

Mixed precision options:

- `null`: fp32 path.
- `fp16`: fp16 mixed precision for the FSDP path.
- `bf16`: bfloat16 mixed precision for supported hardware and runtimes.

Use bf16 only after confirming hardware support and validating loss/perplexity for the target workload.

## Launch on Two GPUs

```bash
uv run python -m ajllm.workflows.lm_train_distributed \
  --config configs/runs/lm_train/tinystories_baseline.yaml \
  --nproc-per-node 2
```


## Batch-Size Semantics

For FSDP training, `training.batch_size` is currently passed to each rank. With two ranks, the effective global batch is approximately `2 × training.batch_size` examples per optimizer step.

The distributed benchmark uses a different convention: its input `batch_size` is split across ranks to hold the benchmark's global batch fixed. Keep this difference in mind when comparing benchmark output with full training runs.

## Results

Benchmark and full-training measurements are collected in [Acceleration Results](ACCELERATION_SUMMARY.md). This page intentionally focuses on implementation, configuration, and launch behavior rather than duplicating result tables.
