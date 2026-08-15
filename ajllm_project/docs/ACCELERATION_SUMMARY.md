# Acceleration Results

This page summarizes measured acceleration results for the TinyStories workflow. It covers the tested model and hardware only; batch size, precision, GPU, context length, and distributed setup all affect the outcome.

Implementation and usage details live in [FlashAttention-2](flash_attention.md) and [FSDP with Activation Checkpointing](fsdp_activation_checkpointing.md).

## Choose a Configuration

| Goal | Recommended configuration |
|---|---|
| Improve CUDA attention efficiency | FlashAttention-2 |
| Reduce per-GPU memory on multiple GPUs | FSDP + activation checkpointing (AC) |
| Best observed fp32 results in these tests | FlashAttention-2 + FSDP + AC |
| Best observed 4090 Lite time and memory | FlashAttention-2 + FSDP + AC with bf16 |

## Full-Training Observations

These are rounded end-to-end observations from TinyStories testing. `GB` is the observed training memory reported during testing. Rows with different batch sizes or precision are operational comparisons, not strict like-for-like speedup measurements.

### 2× T4 (16 GB)

| Configuration | Batch size | Full-training time | Observed memory |
|---|---:|---:|---:|
| Baseline | 32 | 11 h | 12 GB |
| FlashAttention-2 | 32 | 8 h | 8 GB |
| FSDP + AC | 32 | 5 h 30 min | 6 GB |
| FlashAttention-2 + FSDP + AC | 32 | 4 h 30 min | 4 GB |
| FlashAttention-2 + FSDP + AC | 128 | 4 h | 16 GB |

### 2× 4090 Lite (24 GB)

| Configuration | Batch size | Full-training time | Observed memory |
|---|---:|---:|---:|
| Baseline | 64 | 2 h | 24 GB |
| FlashAttention-2 | 64 | 1 h 15 min | 16 GB |
| FlashAttention-2 | 96 | 1 h 07 min | 23 GB |
| FSDP + AC | 64 | 1 h 12 min | 12 GB |
| FSDP + AC | 128 | 1 h 05 min | 23 GB |
| FlashAttention-2 + FSDP + AC, fp32 | 128 | 40 min | 16 GB |
| FlashAttention-2 + FSDP + AC, fp32 | 192 | 37 min | 23 GB |
| FlashAttention-2 + FSDP + AC, bf16 | 128 | 25 min | 10 GB |
| FlashAttention-2 + FSDP + AC, bf16 | 256 | 22 min | 19 GB |

The best observed 4090 Lite configuration was FlashAttention-2 + FSDP + AC with bf16 at batch size 256: 22 minutes and 19 GB observed memory. Confirm loss, perplexity, and stability before using a reduced-precision configuration for another workload.

## Recorded FA2 + FSDP Training Runs

The following completed run folders record two FlashAttention-2 + FSDP + AC training runs. They share the same TinyStories model setup and the same fixed token budget of 327,680,000 tokens.

| Run | Precision | Batch size | Steps | Tokens seen | Elapsed time | Train loss | Validation loss | Best validation loss | Train perplexity | Validation perplexity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [tinystories_fa_fsdp_fp32](../runs/training/tinystories/tinystories_bpe_10k/transformer_baseline/tinystories_fa_fsdp_fp32/20260815T145530Z_5016aa7b/) | fp32 | 128 | 10,000 | 327,680,000 | 43 min 10 s | 1.3700 | 1.3844 | 1.3813 | 3.9354 | 3.9926 |
| [tinystories_fa_fsdp_bf16](../runs/training/tinystories/tinystories_bpe_10k/transformer_baseline/tinystories_fa_fsdp_bf16/20260815T142911Z_e22e4fd5/) | bf16 | 256 | 5,000 | 327,680,000 | 22 min 12 s | 1.3974 | 1.4037 | 1.4097 | 4.0448 | 4.0701 |

Run artifacts:

- fp32: [summary.json](../runs/training/tinystories/tinystories_bpe_10k/transformer_baseline/tinystories_fa_fsdp_fp32/20260815T145530Z_5016aa7b/summary.json), [resolved_config.yaml](../runs/training/tinystories/tinystories_bpe_10k/transformer_baseline/tinystories_fa_fsdp_fp32/20260815T145530Z_5016aa7b/resolved_config.yaml)
- bf16: [summary.json](../runs/training/tinystories/tinystories_bpe_10k/transformer_baseline/tinystories_fa_fsdp_bf16/20260815T142911Z_e22e4fd5/summary.json), [resolved_config.yaml](../runs/training/tinystories/tinystories_bpe_10k/transformer_baseline/tinystories_fa_fsdp_bf16/20260815T142911Z_e22e4fd5/resolved_config.yaml)

The rounded observations above and the exact stored `summary.json` elapsed times are both useful: the table in this section records the exact completed run summaries, while the full-training observations include manually observed configurations that may not all have a corresponding checked-in run folder.

## Step Microbenchmarks

The benchmark uses a 10,000-token vocabulary, `d_model=512`, 4 layers, 16 heads, context length 256, 10 measured steps, and 3 warm-up steps.

| GPU | Benchmark batch | Configuration | World size | Mean step time | Peak allocated memory |
|---|---:|---|---:|---:|---:|
| T4 | 32 | Baseline | 1 | 484.0 ms | 5,804 MiB |
| T4 | 32 | FlashAttention-2 | 1 | 371.1 ms | 3,820 MiB |
| T4 | 32 | FSDP + AC | 2 | 176.5 ms | 1,790 MiB |
| T4 | 32 | FlashAttention-2 + FSDP + AC | 2 | 149.6 ms | 1,294 MiB |
| 4090 Lite | 64 | Baseline | 1 | 181.1 ms | 11,283 MiB |
| 4090 Lite | 64 | FlashAttention-2 | 1 | 103.4 ms | 7,311 MiB |
| 4090 Lite | 64 | FSDP + AC | 2 | 94.6 ms | 3,408 MiB |
| 4090 Lite | 64 | FlashAttention-2 + FSDP + AC | 2 | 90.7 ms | 2,415 MiB |

FSDP rows are rank-local measurements from a two-rank run: the reported time and memory apply to one rank and are not aggregate two-GPU values. Baseline and FlashAttention-2 rows use one GPU, so all rows must not be presented as a single exact global-training speedup comparison.

The benchmark also splits its specified batch across FSDP ranks, whereas the training workflow currently uses `training.batch_size` per rank. This is why benchmark and full-training batch sizes should be interpreted separately.

Raw data: [T4 CSV](../results/benchmark_acceleration_t4.csv) and [4090 Lite CSV](../results/benchmark_acceleration_4090lite.csv).
