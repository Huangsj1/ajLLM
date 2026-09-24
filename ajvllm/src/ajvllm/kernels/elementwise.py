"""Inference fusions adapted from ajllm's row-wise Triton kernels.

Qwen uses split-half RoPE and rounds intermediate elementwise products to the
activation dtype; preserve these semantics rather than copying the training ABI.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm(X, R, W, Y, S, N: tl.constexpr, EPS: tl.constexpr, ADD: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, B)
    x = tl.load(X + row * N + d, d < N, 0).to(tl.float32)
    if ADD:
        # if residual is provided, add it to the input before computing the norm
        x = (x + tl.load(R + row * N + d, d < N, 0).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
        tl.store(S + row * N + d, x, d < N)
    inv = tl.rsqrt(tl.sum(x * x, 0) / N + EPS)
    w = tl.load(W + d, d < N, 0).to(tl.float32)
    y = (x * inv).to(X.dtype.element_ty).to(tl.float32) * w
    tl.store(Y + row * N + d, y, d < N)


def rms_norm(x, weight, eps, residual=None):
    output = torch.empty_like(x)
    summed = torch.empty_like(x) if residual is not None else output
    # grid = (all_tokens_num,)
    _norm[(x.numel() // x.shape[-1],)](
        x,
        residual if residual is not None else x,
        weight,
        output,
        summed,
        x.shape[-1],
        eps,
        residual is not None,
        triton.next_power_of_2(x.shape[-1]),
        enable_fp_fusion=False,
    )
    return (output, summed) if residual is not None else output


@triton.jit
def _swiglu(G, U, Y, N, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    g = tl.load(G + i, i < N, 0).to(tl.float32)
    u = tl.load(U + i, i < N, 0).to(tl.float32)
    silu = (g / (1.0 + tl.exp(-g))).to(G.dtype.element_ty).to(tl.float32)
    tl.store(Y + i, silu * u, i < N)


def swiglu(gate, up):
    out = torch.empty_like(gate)
    # grid = (ceil(all_tokens_num * dff / 256),)
    _swiglu[(triton.cdiv(gate.numel(), 256),)](gate, up, out, gate.numel(), 256)
    return out


@triton.jit
def _rope_cache(
    Q, K, V, C, S, Slots, Out, KC, VC, HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, B: tl.constexpr
):
    t, h = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, B)
    partner = tl.where(d < D // 2, d + D // 2, d - D // 2)
    sign = tl.where(d < D // 2, -1.0, 1.0)
    # load this token's cos/sin factors, shape = (head_dim,)
    c = tl.load(C + t * D + d, d < D, 0).to(tl.float32)
    s = tl.load(S + t * D + d, d < D, 0).to(tl.float32)
    # query head
    if h < HQ:
        base = (t * HQ + h) * D
        # x means current token current head's query, shape = (head_dim,)
        x = tl.load(Q + base + d, d < D, 0).to(tl.float32)
        # z = rotate half x, shape = (head_dim,)
        z = tl.load(Q + base + partner, d < D, 0).to(tl.float32)
        a = (x * c).to(Q.dtype.element_ty).to(tl.float32)
        b = (z * sign * s).to(Q.dtype.element_ty).to(tl.float32)
        # store rotated query to output
        tl.store(Out + base + d, a + b, d < D)
    # kv head
    else:
        head = h - HQ
        base = (t * HK + head) * D
        x = tl.load(K + base + d, d < D, 0).to(tl.float32)
        z = tl.load(K + base + partner, d < D, 0).to(tl.float32)
        a = (x * c).to(K.dtype.element_ty).to(tl.float32)
        b = (z * sign * s).to(K.dtype.element_ty).to(tl.float32)
        # this token's slot
        slot = tl.load(Slots + t)
        target = (slot * HK + head) * D + d
        # store rotated k to k cache, and unrotated v to v cache
        tl.store(KC + target, a + b, (d < D) & (slot >= 0))
        tl.store(VC + target, tl.load(V + base + d, d < D, 0), (d < D) & (slot >= 0))


def rope_and_cache(q, k, v, factors, paged, layer):
    out = torch.empty_like(q)
    # this layer's kv cache tensors, shape = (num_blocks*block_size, num_kv_heads, head_dim)
    kc, vc = paged.storage.layer(layer)
    # grid = (all_tokens_num, num_heads+num_kv_heads), each program processes one token and one head(q or kv)
    _rope_cache[(q.shape[0], q.shape[1] + k.shape[1])](
        q,
        k,
        v,
        *factors,
        paged.slot_mapping,
        out,
        kc,
        vc,
        q.shape[1],
        k.shape[1],
        q.shape[2],
        triton.next_power_of_2(q.shape[2]),     # head_dim
        enable_fp_fusion=False,
    )
    return out
