# FlashAttention-2

## Overview

FlashAttention-2 computes attention in tiles with an online softmax, avoiding materialization of the full attention matrix. Instead of storing the full `batch × heads × sequence × sequence` attention matrix, it streams through key/value blocks and keeps only the state needed to produce the same attention output.

In ajLLM this is exposed as an acceleration option for Transformer language-model training. It is most useful on CUDA workloads where attention activation memory or attention-kernel bandwidth becomes a bottleneck.

## Implementation in ajLLM

The project provides FlashAttention-2 support in the modeling layer and selects the accelerated attention path from the training configuration.

Key properties:

- The model can be switched between standard attention and FlashAttention-2 through config only.
- There are two implementation paths: a Triton kernel and a PyTorch fallback. The Triton kernel is used when available, otherwise the PyTorch implementation is used.
- FlashAttention-2 can be used alone or together with FSDP + activation checkpointing for multi-GPU training.

## Configuration

Enable FlashAttention-2 without FSDP:

```yaml
acceleration:
  use_flash_attention: true
  use_fsdp: false
  mixed_precision: null
```

`mixed_precision` only affects the FSDP path. For a non-FSDP FlashAttention-2 run, the model follows the normal training dtype path.

## Launch Training

```bash
uv run ajllm lm train --config configs/runs/lm_train/tinystories_baseline.yaml
```

The config file controls whether the model uses standard attention or FlashAttention-2.

## Combining with FSDP

FlashAttention-2 reduces attention memory, while FSDP + activation checkpointing reduces distributed training state and saved activations. They can be enabled together:

```yaml
acceleration:
  use_flash_attention: true
  use_fsdp: true
  mixed_precision: null
```

Launch the combined multi-GPU path through the distributed launcher rather than the ordinary single-process command. See [FSDP with Activation Checkpointing](fsdp_activation_checkpointing.md).

## Results

Benchmark and full-training measurements are collected in [Acceleration Results](ACCELERATION_SUMMARY.md). This page intentionally focuses on the project implementation and usage path rather than duplicating result tables.
