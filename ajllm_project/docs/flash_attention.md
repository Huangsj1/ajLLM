# FlashAttention-2

## Overview

FlashAttention-2 is a memory-efficient attention algorithm that avoids materializing the full O(n²) attention matrix. It uses tiling and online softmax to compute attention in blocks, keeping only small tiles in fast SRAM/registers while streaming through the sequence.

## Benefits

**Memory Savings:**
- 4-8× reduction in activation memory for attention layers
- Enables longer sequences or larger batch sizes
- Critical for memory-constrained GPUs

**Speed Improvements:**
- 1.5-2× faster attention computation (memory bandwidth bound)
- No accuracy loss compared to standard attention
- Fused kernel reduces memory traffic

## How It Works

Standard attention materializes the full attention matrix:
```
scores = Q @ K^T / sqrt(d)     # Shape: (batch, heads, n, n)
probs = softmax(scores)         # Materialize full matrix
output = probs @ V              # Shape: (batch, heads, n, d)
```

FlashAttention-2 tiles the computation:
1. Split Q into blocks (query tiles)
2. For each query tile, stream through K/V blocks
3. Maintain running softmax statistics (max, sum) online
4. Never materialize the full n×n attention matrix

This reduces peak memory from O(n²) to O(n) for the attention operation.

## Implementation

We provide two backends:
- **PyTorch**: Pure PyTorch with `torch.compile` for backward pass (works on any device)
- **Triton**: Fused CUDA kernels for maximum performance (CUDA only, optional)

The implementation automatically selects the appropriate backend and falls back gracefully.

## Usage

### Enable in Configuration

Add to your training config:
```yaml
acceleration:
  use_flash_attention: true
  mixed_precision: null  # Optional: fp16 or bf16 for FSDP
```

Note: `mixed_precision` only affects FSDP. FlashAttention2 automatically handles whatever dtype the inputs are in.

### Example Configs

```bash
# TinyStories with FlashAttention2
configs/runs/lm_train/tinystories_baseline.yaml:
  acceleration:
    use_flash_attention: true
    use_fsdp: false
```

### Launch Training

```bash
# Single GPU
uv run ajllm lm train --config configs/runs/lm_train/tinystories_baseline.yaml
```

## Performance Benchmarks

Based on tinystories_baseline model (22.7M params, d_model=512, context=256):

| Configuration | Forward (ms) | Backward (ms) | Peak Memory (MB) |
|---------------|--------------|---------------|------------------|
| Baseline      | 45-60        | 90-120        | ~2,500           |
| FlashAttention2 | 30-40      | 60-80         | ~800             |

**Memory Reduction:** ~3× for this model size and context length

**Speedup:** ~1.5-1.7× overall (compute becomes more efficient)

Memory savings scale with sequence length: longer contexts see larger benefits.

## Limitations

- **CUDA Required**: Full performance requires CUDA-capable GPU
- **CPU Fallback**: Automatically falls back to standard attention on CPU
- **Sequence Length**: Most effective for longer sequences (context > 128)
- **Triton Optional**: Triton kernels provide best performance but are optional

## Technical Details

### Tiling Strategy

- Query tile size: 32 (for d ≤ 64: 64)
- Key tile size: 32 (for d ≤ 64: 64)
- Tiles chosen to fit in GPU L1/SRAM
- Online softmax maintains running maximum and sum

### Numerical Stability

- All accumulations in fp32 regardless of input dtype
- Softmax computed with stable logsumexp
- No accuracy degradation compared to standard attention

### Backward Pass

- Recomputes attention matrix from saved logsumexp
- No need to save full attention matrix (memory savings)
- Compiled with `torch.compile` for performance

## Combining with FSDP

FlashAttention2 and FSDP+AC are complementary:
- FlashAttention reduces activation memory in attention layers
- FSDP+AC reduces parameter and remaining activation memory
- Together they provide 8-15× total memory reduction

Enable both:
```yaml
acceleration:
  use_flash_attention: true
  use_fsdp: true
```

## Troubleshooting

**CUDA Out of Memory:**
- FlashAttention reduces memory but doesn't eliminate limits
- Try smaller batch size or shorter context length
- Enable FSDP for multi-GPU sharding

**Triton Import Error:**
- Triton is optional; PyTorch backend is used automatically
- Install Triton: `pip install triton` (CUDA 11.8+ required)

**Slower than Expected:**
- FlashAttention is memory-bandwidth bound
- Benefits increase with sequence length
- Ensure CUDA is being used (check device in logs)

## References

- FlashAttention-2 Paper: https://arxiv.org/abs/2307.08691
- Original FlashAttention: https://arxiv.org/abs/2205.14135
