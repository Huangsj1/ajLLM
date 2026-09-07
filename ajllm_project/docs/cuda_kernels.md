# CUDA kernel guide

## Goal and dispatch

`use_flash_attention: true` selects the project's Triton FlashAttention.
`use_cuda_kernels: true` selects Triton RMSNorm, RoPE, SwiGLU, vocabulary Cross
Entropy, AdamW updates, gradient-norm partial reductions, Top-1 MoE
dispatch/combine, and Variable-M grouped GEMM. Turning either switch off uses
the corresponding explicit Torch equation. The MoE router keeps the explicit
PyTorch Softmax equation because it has only a few experts and a standalone
router kernel is slower.

## Kernel inventory and implementation order

| Priority | Operation | Status | Why it matters |
| --- | --- | --- | --- |
| 1 | RMSNorm | implemented | Four Torch operations become one row kernel. |
| 2 | RoPE | implemented | Avoids separate indexing, pair rotation, stack, and flatten launches. |
| 3 | SwiGLU pointwise gate | implemented | Fuses SiLU and multiplication after the projection GEMMs. |
| 4 | Shifted row Softmax utility | implemented, not routed by MoE | Useful only when fused with adjacent work. |
| 5 | Causal FlashAttention | existing Triton implementation | Avoids materializing the attention matrix. |
| 6 | Vocabulary cross entropy | implemented | Fuses stable Softmax, target gather, and negative log likelihood. |
| 7 | MoE Top-1 dispatch + Variable-M Grouped GEMM | implemented | Runs compact unequal expert row ranges without capacity padding. |
| 8 | AdamW / global gradient norm | implemented | Fuses per-tensor updates and norm partial reductions. |

Deferred items are intentionally not replaced by superficial kernels: they need
separate numerical, distributed, and performance validation.

## RMSNorm

Input `[..., H]` is flattened conceptually to `[M, H]`; default `H=768` is
padded to a 1024-lane Triton block.

```text
grid: (M,)
program/block: one normalized row
lanes: columns 0..BLOCK-1, masked when column >= H
```

1. Each lane loads one activation `x[column]` and its gain.
2. The block reduces `x²` across lanes to one mean square for its row.
3. Every lane multiplies by `rsqrt(mean_square + epsilon)` and gain, then stores.
4. Backward uses one row program for `dX`; `[32 rows, 256 columns]` tiles
   atomically accumulate `dWeight`.

This fuses fp32 accumulation, reciprocal square root, normalization, and gain
scaling. The Torch fallback in `layers.py` remains the equation of record.

## RoPE

RoPE input is `[B, heads, T, D]`, with default `D=64`. A program owns one
`[batch, head, token]` row and has one lane per even/odd pair (`D/2` lanes).

```text
grid: (B * heads * T,)
program/block: one token in one attention head
lane i: pair (x[2i], x[2i + 1])
```

The program obtains the absolute position through the position tensor, loads the
pair cosine/sine from the cached table, and writes:

```text
y_even = x_even * cos - x_odd * sin
y_odd  = x_even * sin + x_odd * cos
```

Backward applies the transpose rotation with the same row mapping. This removes
Torch slices, pairwise products, stack, and flatten allocation.

## SwiGLU pointwise gate

Dense FFN projections remain separate. MoE concatenates gate/up weights and
produces two strided `[T, I]` views from `[T, 2I]`. The kernel accepts their row
strides, so it does not materialize contiguous copies. A program handles 256
columns of one logical row.

```text
grid: (rows, ceil(I / 256))
program/block: 256 columns from one row
lane i: gate[row, i], up[row, i], output[row, i]
```

Each lane calculates `sigmoid(gate)`, then `gate * sigmoid(gate) * up`, and
stores one output. The backward kernel writes both `dGate` and `dUp` in the
same launch using the analytic SiLU derivative.

## Shifted row Softmax utility

The utility accepts `[rows, width]`. One program owns a full row; lanes are
columns padded to a power of two.

1. Lanes load row logits and reduce their maximum.
2. Lanes compute `exp(logit - max)` and reduce their sum.
3. Each lane writes its normalized probability.
4. Backward reduces `sum(dy * probability)` once, then writes
   `probability * (dy - dot)` per lane.

The current MoE router deliberately uses `torch.softmax`. With only four
experts, its specialized PyTorch kernel is three to four times faster than a
standalone Triton launch. The Triton utility is retained for future kernels that
can fuse Softmax with routing or another adjacent operation. Top-K and expert
`index_add_` for Top-K greater than one remain explicit Torch operations because
their correct kernel design depends on future capacity and expert-parallel policy.

## Top-1 MoE grouped execution

The default `num_experts=4`, `num_experts_per_token=1` configuration has a
specialized sparse execution path. It does not use a Triton router Softmax:
Top-1 output weights normalize to one, so `argmax(router_logits)` selects the
expert. PyTorch Softmax is calculated only during training for the auxiliary
load-balancing loss.

Tokens are sorted once by expert into `compact[T, D]`. Prefix sums of token
counts form `offsets[E+1]`; expert `e` owns exactly
`compact[offsets[e]:offsets[e+1]]`. No `[E, capacity, D]` allocation exists.

```text
dispatch grid: (num_tokens, ceil(d_model / 256))
program: one routed token and 256 hidden columns
lane: load input[token_indices[position], column] -> compact[position, column]

combine grid: (num_tokens, ceil(d_model / 256))
program: one routed token and 256 hidden columns
lane: load compact[position, column] -> output[token_indices[position], column]
```

Both mappings are unique for Top-1, so they require no atomic operations. Their
backward kernels apply the inverse mapping.

### Forward Variable-M GEMM

`gate_up_proj.weight` is `[E, 2I, D]`; concatenating gate and up turns two expert
projections into one kernel. `down_proj.weight` is `[E, D, I]`. A compact tile
schedule contains only `(expert, first_row)` pairs for real expert tiles:

```text
BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
grid: (sum_e ceil(M_e / 128), ceil(N / 128))
program: one expert's [128 output rows, 128 output columns]
lanes cooperate: load [128, 32] input and [128, 32] weight fragments
loop: advance through K in 32-column fragments
accumulator: fp32 [128, 128], stored in the activation dtype
```

The schedule emits no program for `M_e=0`. Only an expert's final M tile masks
its tail, giving at most 63 alignment rows per nonempty expert rather than
padding every expert to `max(M_e)`. N and K tails are independently masked.

### Backward Variable-M GEMM

The custom Triton forward is paired with explicit local GEMM equations for each
of the few experts:

```text
for expert e:
    dInput[offsets[e]:offsets[e+1]] = dOutput_e @ weight_e
    dWeight[e] = dOutput_e.T @ input_e
```

On the RTX 3080 Ti, a previous one-program Triton `dWeight` reduction serialized
the 26k-row hot-expert interval and made training slower. These equations use
the CUDA GEMM backend's Tensor-Core split-K reduction while retaining compact,
unpadded Variable-M activations and the Triton forward dispatch. Under BF16
autocast, activations and gathered weights remain BF16; autograd accumulates
the corresponding FP32 parameter gradients. `VariableGroupedLinear` remains a
subclass of the local `Linear`, preserving custom FSDP all-gather/reduce-scatter
hooks.
Top-K greater than one remains outside this pre-training implementation until
expert-parallel routing is designed.

## Fused vocabulary Cross Entropy

`training/losses.py` retains an explicit Torch log-sum-exp equation for tests
and benchmarks. Every supported CUDA vocabulary size uses one tiled Triton
implementation; there is no small-vocabulary special kernel.

The input `[B, T, V]` is viewed as `[M, V]`, with `M=B*T`. All vocabulary sizes
use two forward launches and one tiled backward launch:

```text
stage 1 grid: (M, ceil(V / TILE))
program: one [token row, vocabulary tile]
lanes: TILE vocabulary columns, with the final tile masked

stage 2 grid: (M,)
program: one token row, reducing its per-tile statistics

backward grid: (M, ceil(V / TILE))
program: one [token row, vocabulary tile]
```

The launch shape is selected from CUDA compute capability:

| GPU architecture | Tile width | Tile kernel warps | Reason |
| --- | ---: | ---: | --- |
| Hopper or newer (`SM >= 9`) | 2048 | 8 | Uses the larger register file and scheduler capacity. |
| Ampere/Ada (`SM >= 8`) | 1024 | 4 | Balanced occupancy and reduction work; measured on RTX 3080 Ti (SM 8.6). |
| Volta/Turing (`SM >= 7`) | 512 | 4 | Restrains register pressure on older SMs. |
| Older CUDA GPU | 256 | 4 | Conservative fallback. |

The second reduction kernel uses one warp for at most 32 tiles, four warps up
to 1024 tiles, then eight warps. This avoids scheduling idle warps for common
small vocabulary counts while retaining enough reduction capacity for large
vocabularies.

Stage 1 writes two FP32 scalars per tile: local maximum `m_i` and shifted sum
`s_i = sum(exp(logit - m_i))`. Stage 2 merges those stable statistics without
re-reading all logits:

```text
m = max_i(m_i)
s = sum_i(s_i * exp(m_i - m))
loss = m + log(s) - target_logit
```

It saves one global maximum and one global shifted sum per token row for
backward. The tiled backward reloads its tile logits, computes
`exp(logit - m) / s`, and writes the corresponding gradient tile. Temporary
statistics are `[M, ceil(V/TILE)]` FP32 scalars, rather than a materialized
`[M, V]` probability tensor. This design is tested at the default `V=6400` and
non-power-of-two `V=32003`.

## AdamW and global gradient norm

The project's direct `AdamW` equations remain the fallback. Each dense,
contiguous CUDA parameter and its gradient/moment tensors use one 256-lane
Triton program per contiguous element block:

```text
grid: (ceil(parameter_elements / 256),)
program/block: 256 parameter elements
lane i: parameter[i], gradient[i], first_moment[i], second_moment[i]
```

Every lane updates its first moment, second moment, bias-corrected adaptive
parameter step, and decoupled weight decay in one launch. The ordering exactly
matches the pre-existing educational Torch implementation:

```text
m = beta1 * m + (1 - beta1) * gradient
v = beta2 * v + (1 - beta2) * gradient^2
parameter = parameter - adjusted_lr * m / (sqrt(v) + epsilon)
parameter = parameter * (1 - learning_rate * weight_decay)
```

The optimizer selects this kernel only for dense contiguous CUDA tensors and
falls back per optimizer to the Torch equations after a Triton error. This also
works with the project's custom FSDP because FSDP exposes each local shard as a
normal parameter tensor; no full-parameter gather is introduced for updates.

Global gradient clipping first needs one scalar across many unrelated parameter
allocations. A safe pointer-array multi-tensor kernel would add complexity that
obscures this project, so the implementation uses a clear two-level design:

1. One 1024-lane Triton program per gradient block produces an FP32 partial
   `sum(gradient^2)` without allocating `gradient.square()`.
2. A small explicit Torch `cat` and sum combines all parameter partial arrays;
   the existing per-gradient `mul_` performs the optional clip scaling.

When distributed custom FSDP is active, the resulting local squared norm uses
`all_reduce(SUM)` before its square root. Each rank therefore clips against the
same norm of all parameter shards, without gathering full parameters.

This keeps the global/FSDP semantics readable while removing the largest
temporary tensor in norm calculation. Non-contiguous gradients, CPU, disabled
CUDA kernels, and unavailable Triton use the original Torch reference.

## Existing FlashAttention

`flash_attention.py` remains the project's original Torch/Triton
implementation. Its Triton forward grid assigns one program to a query tile and
flattened batch/head item. Within the program, lanes form a query-tile by
head-dimension block; it streams K/V tiles, maintains online softmax max/sum
statistics, applies the causal mask, and accumulates output without materializing
`[T, T]` scores. Backward has separate tiled `dK/dV` and `dQ` passes.

The default head dimension is 64, a power of two required by the current
block-pointer implementation. Non-power-of-two widths use the project's tiled
Torch implementation, then SDPA only if that implementation fails.

On Ampere-class GPUs, Triton's fp32 `tl.dot` can use Tensor Core TF32
accumulation. The FlashAttention comparison therefore allows `4e-3` absolute
error against the project's full-fp32 Torch reference; RMSNorm, RoPE, SwiGLU,
and shifted row Softmax use tighter `2e-6` checks. Fused Cross Entropy is
checked against the explicit Torch fp32-reduction formula at both the actual
`V=6400` BF16 autocast shape and a tiled, non-power-of-two `V=32003` shape.

## Correctness checks

```bash
uv run pytest
```

`tests/test_cuda_kernels.py` compares every Triton forward output and analytic
backward gradient with its current Torch reference, calls
`torch.cuda.synchronize()` to surface asynchronous faults, and is skipped only
on CPU-only hosts. It covers RMSNorm, RoPE, SwiGLU, shifted Softmax, fused
Cross Entropy (including `-100` labels and backward), and existing Triton
FlashAttention. It also compares three fused AdamW updates against the explicit
Torch moments/parameters and checks a BF16 gradient norm with a masked tail
block. Top-1 dispatch/combine are checked against Torch for forward and inverse
gradients, and the complete grouped MoE path is exercised on CUDA. The
Variable-M test includes unequal expert sizes, an empty expert, non-multiple
matrix dimensions, and validates forward, `dInput`, and `dWeight`. CPU tests
also check the equations and custom FSDP expert-weight sharding.

## Simple speed benchmark

The unified benchmark runs the original core kernels and the detailed MoE
breakdown at the current training shape:

```bash
uv run python -m ajllm.utils.benchmark_cuda_kernels \
  --section all --batch-size 16 --sequence-length 512 \
  --warmup 10 --iterations 20
```

The script uses CUDA events after warmup, so compilation time and Python timer
overhead are excluded. `--section core` runs only the original one-operation
comparisons; `--section moe` runs the MoE components and complete paths. The
default dtype is BF16, while `--dtype bf16|fp16|fp32` permits explicit testing.

The shared shape parameters are `--batch-size`, `--sequence-length`,
`--vocabulary-size`, `--hidden-size`, `--num-heads`, and
`--intermediate-size`. MoE additionally accepts `--num-experts` and
`--primary-expert-fraction`; omitting the latter creates balanced routes for any
expert count. For example:

```bash
# Only MoE, with 80% of tokens routed to expert 0.
uv run python -m ajllm.utils.benchmark_cuda_kernels \
  --section moe --num-experts 4 --primary-expert-fraction 0.8 \
  --hidden-size 768 --intermediate-size 2432 --dtype bf16
```

The MoE section reports token dispatch, token combine, Gate/Up Variable-M GEMM,
Down Variable-M GEMM, complete expert MLP forward and forward+backward, fixed-
route Top-1 execution, and the complete router-to-combine MoE layer. Each row
compares the explicit Torch reference against Triton using identical shapes and
weights. These are operation-level measurements, not full-model throughput.

On the project's RTX 3080 Ti with BF16, the measured operation latencies were:

| Operation | Torch | Triton | Speedup |
| --- | ---: | ---: | ---: |
| RMSNorm | 0.357 ms | 0.035 ms | 10.20x |
| RoPE | 0.373 ms | 0.141 ms | 2.64x |
| SwiGLU gate | 0.429 ms | 0.159 ms | 2.69x |
| Shifted Softmax (`B=16`, `T=512`, `V=6400`) | 0.455 ms | 0.277 ms | 1.64x |
| Tiled Cross Entropy (`B=16`, `T=512`, `V=6400`) | 2.706 ms | 0.302 ms | 8.95x |
| AdamW update (`V * H = 6400 * 768` FP32 elements) | 0.524 ms | 0.181 ms | 2.90x |
| Gradient squared norm (same FP32 tensor) | 0.086 ms | 0.031 ms | 2.73x |
| MoE token dispatch (`E=4`, `B=16`, `T=512`) | 0.048 ms | 0.043 ms | 1.11x |
| MoE token combine | 0.042 ms | 0.041 ms | 1.02x |
| MoE Gate/Up Variable-M GEMM | 1.318 ms | 1.388 ms | 0.95x |
| MoE Down Variable-M GEMM | 0.614 ms | 0.665 ms | 0.92x |
| Complete expert MLP forward | 2.187 ms | 2.172 ms | 1.01x |
| Complete expert MLP forward + backward | 8.424 ms | 9.511 ms | 0.89x |
| Fixed-route Top-1 execution forward | 4.054 ms | 3.108 ms | 1.30x |
| Complete MoE layer forward | 4.367 ms | 3.216 ms | 1.36x |
| Complete MoE layer forward + backward | 12.093 ms | 11.154 ms | 1.08x |
| Skewed Top-1 execution forward (6553/547/546/546) | 4.576 ms | 3.162 ms | 1.45x |
| Skewed expert MLP forward + backward | 8.400 ms | 10.072 ms | 0.83x |
| FlashAttention | 87.155 ms | 0.145 ms | 600.03x |

The tiled Cross Entropy path was also measured at `B=4`, `T=128`,
`V=32003`: Torch `1.109 ms`, Triton `0.217 ms` (**5.10x**). This smaller batch
avoids making a micro-benchmark's temporary vocabulary tensors dominate GPU
memory; it is not an end-to-end training claim.

The standalone shifted Softmax is faster at vocabulary width but is still not
used by the current four-expert MoE router. It is retained as a general utility;
Cross Entropy shows the larger benefit of also fusing Softmax with its consumer.
Hardware, dtype, and Triton/PyTorch versions affect these values.

Balanced routing is the default. Tiny dispatch timings are launch-bound and may
fluctuate around parity; the expert MLP is the relevant compute result. The
listed MoE numbers use 50 measured iterations after 10 warmups. The current
handwritten `dWeight` kernel is correct but standalone expert backward remains
slower than cuBLAS at `E=4`. The complete layer nevertheless improves because
it also measures compact dispatch, schedule construction, and the removal of
repeated per-expert execution overhead. `--primary-expert-fraction` controls the
fixed-route rows; complete-layer rows use their separately printed router counts.
