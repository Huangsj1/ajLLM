# Native Triton inference kernels

`attention.py` implements paged online-softmax prefill and partitioned decode
with LSE merging. `elementwise.py` implements RMSNorm, residual + RMSNorm,
SwiGLU, and Qwen split-half RoPE fused with paged KV writes.

See [architecture](../../../docs/architecture/architecture.md#stage-3-compute-algorithms)
for layout contracts, launch policy and numerical limits.
