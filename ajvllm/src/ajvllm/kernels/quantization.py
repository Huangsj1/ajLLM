"""INT8 weight-only matrix products with dequantization inside the CUDA kernel."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm(
    X,
    W,
    Scales,
    Bias,
    Out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    scales = tl.load(Scales + cols, cols < N, 0)
    acc = tl.zeros((BM, BN), tl.float32)
    for begin in range(0, K, BK):
        k = begin + ks
        x = tl.load(X + rows[:, None] * K + k[None, :], (rows[:, None] < M) & (k[None, :] < K), 0)
        w = tl.load(W + cols[None, :] * K + k[:, None], (cols[None, :] < N) & (k[:, None] < K), 0)
        weight = (w.to(tl.float32) * scales[None, :]).to(x.dtype)
        acc = tl.dot(x, weight, acc)
    if HAS_BIAS:
        acc += tl.load(Bias + cols, cols < N, 0).to(tl.float32)[None, :]
    tl.store(Out + rows[:, None] * N + cols[None, :], acc, (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _gemv(
    X,
    W,
    Scales,
    Bias,
    Out,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    scales = tl.load(Scales + cols, cols < N, 0)
    acc = tl.zeros((BN, BK), tl.float32)
    for begin in range(0, K, BK):
        k = begin + ks
        x = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
        w = tl.load(W + cols[:, None] * K + k[None, :], (cols[:, None] < N) & (k[None, :] < K), 0)
        weight = (w.to(tl.float32) * scales[:, None]).to(X.dtype.element_ty).to(tl.float32)
        acc += weight * x[None, :]
    result = tl.sum(acc, 1)
    if HAS_BIAS:
        result += tl.load(Bias + cols, cols < N, 0).to(tl.float32)
    tl.store(Out + row * N + cols, result, cols < N)


def linear_w8a16(x, weight, scales, bias):
    rows, width = x.numel() // x.shape[-1], weight.shape[0]
    output = torch.empty((*x.shape[:-1], width), device=x.device, dtype=x.dtype)
    bias_ptr = bias if bias is not None else scales
    if rows <= 8:
        _gemv[(rows, triton.cdiv(width, 4))](
            x, weight, scales, bias_ptr, output, width, x.shape[-1], bias is not None, 4, 512, num_warps=4
        )
    else:
        _gemm[(triton.cdiv(rows, 32), triton.cdiv(width, 64))](
            x,
            weight,
            scales,
            bias_ptr,
            output,
            rows,
            width,
            x.shape[-1],
            bias is not None,
            32,
            64,
            32,
            num_warps=4,
            num_stages=3,
        )
    return output
