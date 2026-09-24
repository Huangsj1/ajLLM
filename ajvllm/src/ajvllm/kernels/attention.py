"""Packed paged FlashAttention and split-K decode, forward inference only.

Online-softmax recurrence follows ajllm's flash_attention.py. Addressing, ragged
query tiles, chunk offsets, GQA and split-K reduction are native inference paths.
"""

import torch
import triton
import triton.language as tl

PREFILL_TILE = 32
DECODE_PARTITION = 256


@triton.jit
def _prefill(
    Q,          # packed query tensor, shape = (sum(lengths), num_heads, head_dim)
    K,          # k cache in this layer, shape = (num_slots, num_kv_heads, head_dim)
    V,          # v cache in this layer, shape = (num_slots, num_kv_heads, head_dim)
    Tables,     # block tables for each request, shape = (num_requests, max_num_blocks_for_one_request)
    Tiles,      # (row, start) pairs for each prefill tile
    Starts,     # start indices of each request in the packed token buffer
    Contexts,   # context lengths of each request
    Out,
    WIDTH,      # max_num_blocks_for_one_request
    PAGE: tl.constexpr,     # block size
    HQ: tl.constexpr,       # num_query_heads
    HK: tl.constexpr,       # num_kv_heads
    D: tl.constexpr,        # head_dim
    BD: tl.constexpr,       # next_power_of_2(head_dim)
    BM: tl.constexpr,       # PREFILL_TILE=32
    BN: tl.constexpr,       # 32
):
    tile, head = tl.program_id(0), tl.program_id(1)
    # request row and local token index within the tile
    row = tl.load(Tiles + tile * 2)
    local = tl.load(Tiles + tile * 2 + 1) + tl.arange(0, BM)
    # begin and end indices of this request in the packed token buffer
    begin, end = tl.load(Starts + row), tl.load(Starts + row + 1)
    length = end - begin
    context = tl.load(Contexts + row)
    # token positions in the context, shape = (PREFILL_TILE,)
    pos = context - length + local
    d = tl.arange(0, BD)
    # q.shape = (PREFILL_TILE, BD), valid part = (PREFILL_TILE, D)
    q = tl.load(
        Q + ((begin + local[:, None]) * HQ + head) * D + d[None, :], (local[:, None] < length) & (d[None, :] < D), 0
    )
    # flash attention below: m = max(scores), normalizer = sum(exp(scores-m)), acc = sum(exp(scores-m)*v)
    m = tl.full((BM,), float("-inf"), tl.float32)
    normalizer = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)
    # causal attention: only attend to tokens <= current token, and < context length
    limit = tl.minimum(context, context - length + tl.load(Tiles + tile * 2 + 1) + BM)
    # for each block clolumn of K matrix, load the block's k/v and compute attention for this tile's queries
    for start in range(0, limit, BN):
        n = start + tl.arange(0, BN)
        page = tl.load(Tables + row * WIDTH + n // PAGE, n < context, 0)
        slots = page * PAGE + n % PAGE
        offsets = (slots[:, None] * HK + head // (HQ // HK)) * D + d[None, :]
        k = tl.load(K + offsets, (n[:, None] < context) & (d[None, :] < D), 0)
        v = tl.load(V + offsets, (n[:, None] < context) & (d[None, :] < D), 0)
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * (D**-0.5)
        scores = tl.where((n[None, :] <= pos[:, None]) & (n[None, :] < context), scores, float("-inf"))
        new_m = tl.maximum(m, tl.max(scores, 1))
        alpha = tl.exp(m - new_m)
        p = tl.exp(scores - new_m[:, None])
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc, input_precision="ieee")
        normalizer = normalizer * alpha + tl.sum(p, 1)
        m = new_m
    tl.store(
        Out + ((begin + local[:, None]) * HQ + head) * D + d[None, :],
        acc / normalizer[:, None],
        (local[:, None] < length) & (d[None, :] < D),
    )


@triton.jit
def _decode(
    Q,
    K,
    V,
    Tables,
    Rows,
    Starts,
    Contexts,
    Partial,
    LSE,
    Out,
    WIDTH,
    PAGE: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    PART: tl.constexpr,
    SPLITS: tl.constexpr,
    BN: tl.constexpr,
):
    request, head, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row = tl.load(Rows + request)
    token = tl.load(Starts + row)
    context = tl.load(Contexts + row)
    d = tl.arange(0, BD)
    q = tl.load(Q + (token * HQ + head) * D + d, d < D, 0).to(tl.float32)
    m = tl.full((), float("-inf"), tl.float32)
    normalizer = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BD,), tl.float32)
    for start in range(part * PART, tl.minimum(context, (part + 1) * PART), BN):
        n = start + tl.arange(0, BN)
        page = tl.load(Tables + row * WIDTH + n // PAGE, n < context, 0)
        slots = page * PAGE + n % PAGE
        offsets = (slots[:, None] * HK + head // (HQ // HK)) * D + d[None, :]
        k = tl.load(K + offsets, (n[:, None] < context) & (d[None, :] < D), 0).to(tl.float32)
        v = tl.load(V + offsets, (n[:, None] < context) & (d[None, :] < D), 0).to(tl.float32)
        scores = tl.sum(k * q[None, :], 1) * (D**-0.5)
        scores = tl.where(n < context, scores, float("-inf"))
        new_m = tl.maximum(m, tl.max(scores, 0))
        p = tl.exp(scores - new_m)
        alpha = tl.exp(m - new_m)
        acc = acc * alpha + tl.sum(p[:, None] * v, 0)
        normalizer = normalizer * alpha + tl.sum(p, 0)
        m = new_m
    if SPLITS == 1:
        tl.store(Out + (token * HQ + head) * D + d, acc / tl.maximum(normalizer, 1.0e-20), d < D)
    else:
        index = (request * HQ + head) * SPLITS + part
        tl.store(Partial + index * D + d, acc / tl.maximum(normalizer, 1.0e-20), d < D)
        tl.store(LSE + index, tl.where(normalizer > 0, m + tl.log(normalizer), float("-inf")))


@triton.jit
def _merge(
    Partial,
    LSE,
    Rows,
    Starts,
    Out,
    HQ: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
):
    request, head = tl.program_id(0), tl.program_id(1)
    s, d = tl.arange(0, BS), tl.arange(0, BD)
    index = (request * HQ + head) * SPLITS + s
    lse = tl.load(LSE + index, s < SPLITS, float("-inf"))
    maximum = tl.max(lse, 0)
    weights = tl.exp(lse - tl.where(maximum == float("-inf"), 0.0, maximum))
    weights /= tl.maximum(tl.sum(weights, 0), 1.0e-20)
    values = tl.load(Partial + index[:, None] * D + d[None, :], (s[:, None] < SPLITS) & (d[None, :] < D), 0)
    result = tl.sum(values * weights[:, None], 0)
    token = tl.load(Starts + tl.load(Rows + request))
    tl.store(Out + (token * HQ + head) * D + d, result, d < D)


def paged_attention(q, paged, layer, metadata):
    # q, out shape = (sum(lengths), num_heads, head_dim)
    out = torch.empty_like(q)
    # k,v shape = (num_slots, num_kv_heads, head_dim)
    k, v = paged.storage.layer(layer)
    hq, d = q.shape[1:]
    hk = k.shape[1]
    # args = (max_num_blocks_for_one_request, block_size, num_query_heads, num_kv_heads, head_dim,
    # next_power_of_2(head_dim))
    args = (paged.block_tables.shape[1], paged.storage.block_size, hq, hk, d, max(16, triton.next_power_of_2(d)))
    # 1. prefill
    if metadata.prefill_tiles.shape[0]:
        # grid = (num_prefill_tiles, num_heads), each program processes one tile's one head(one prefill request may
        # have multiple tiles)
        _prefill[(metadata.prefill_tiles.shape[0], hq)](
            q,
            k,
            v,
            paged.block_tables,
            metadata.prefill_tiles,
            metadata.starts,
            metadata.contexts,
            out,
            *args,
            PREFILL_TILE,
            32 if d > 128 else 64,
            num_warps=4,
            num_stages=1 if d > 128 else 3,
        )
    count = metadata.decode_rows.numel()
    if count:
        # Short contexts use one partition; longer ones expose more parallelism.
        part = (
            DECODE_PARTITION
            if metadata.max_decode_context >= 1024
            else triton.next_power_of_2(metadata.max_decode_context)
        )
        splits = triton.cdiv(metadata.max_decode_context, part)
        partial = torch.empty((count, hq, splits, d), device=q.device, dtype=torch.float32) if splits > 1 else out
        lse = torch.empty((count, hq, splits), device=q.device, dtype=torch.float32) if splits > 1 else out
        _decode[(count, hq, splits)](
            q,
            k,
            v,
            paged.block_tables,
            metadata.decode_rows,
            metadata.starts,
            metadata.contexts,
            partial,
            lse,
            out,
            *args,
            part,
            splits,
            64,
            num_warps=4,
        )
        if splits > 1:
            _merge[(count, hq)](
                partial,
                lse,
                metadata.decode_rows,
                metadata.starts,
                out,
                hq,
                d,
                splits,
                triton.next_power_of_2(splits),
                triton.next_power_of_2(d),
                num_warps=4,
            )
    return out
