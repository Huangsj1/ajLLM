"""Fused score transforms and tiled sorted inverse-CDF sampling on CUDA."""

import torch
import triton
import triton.language as tl

TILE = 1024


@triton.jit
def _update_history(Addresses, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    address = tl.load(Addresses + i, i < N, 0).to(tl.pointer_type(tl.int32))
    tl.atomic_add(address, 1, i < N, sem="relaxed")


@triton.jit
def _transform(Logits, Params, Metadata, Scores, Invalid, V: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    i = tile * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(Logits + row * V + i, i < V, float("-inf")).to(tl.float32)
    bad = (x != x) | (x == float("inf"))
    address = tl.load(Metadata + row * 3)
    if address != 0:
        history = address.to(tl.pointer_type(tl.int32))
        seen = tl.load(history + i, i < V, 0) > 0
        counts = tl.load(history + V + i, i < V, 0).to(tl.float32)
        rep = tl.load(Params + row * 5)
        presence = tl.load(Params + row * 5 + 1)
        frequency = tl.load(Params + row * 5 + 2)
        penalized = tl.where(x > 0, x / rep, x * rep)
        x = tl.where(seen, penalized, x) - presence * (counts > 0) - frequency * counts
        bad |= (x != x) | (x == float("inf"))
        if tl.load(Metadata + row * 3 + 1) != 0:
            stopped = tl.load(history + 2 * V + i, i < V, 0) > 0
            x = tl.where(stopped, float("-inf"), x)
    tl.store(Scores + row * V + i, x, i < V)
    tl.store(Invalid + row * TILES + tile, tl.max((bad & (i < V)).to(tl.int32), 0))


@triton.jit
def _cdf_tiles(Values, Params, Metadata, CDF, Sums, V: tl.constexpr, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    i = tile * BLOCK + tl.arange(0, BLOCK)
    maximum = tl.load(Values + row * V)
    x = tl.load(Values + row * V + i, i < V, float("-inf"))
    temperature = tl.maximum(tl.load(Params + row * 5 + 3), 1.1754943508222875e-38)
    limit = tl.load(Metadata + row * 3 + 2)
    # Unnormalized weights avoid materializing softmax and renormalized probabilities.
    weight = tl.exp((x - maximum) / temperature)
    weight = tl.where((i < V) & (i < limit) & (x != float("-inf")), weight, 0.0)
    prefix = tl.cumsum(weight, 0)
    tl.store(CDF + row * V + i, prefix, i < V)
    tl.store(Sums + row * TILES + tile, tl.sum(tl.where(tl.arange(0, BLOCK) == BLOCK - 1, prefix, 0.0), 0))


@triton.jit
def _select(
    Values,
    IDs,
    Params,
    CDF,
    Sums,
    Invalid,
    Draws,
    Result,
    V: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
    BT: tl.constexpr,
):
    row = tl.program_id(0)
    t = tl.arange(0, BT)
    sums = tl.load(Sums + row * TILES + t, t < TILES, 0)
    prefixes = tl.cumsum(sums, 0)
    total = tl.sum(tl.where(t == TILES - 1, prefixes, 0.0), 0)
    p = tl.load(Params + row * 5 + 4)
    lanes = tl.arange(0, BLOCK)
    cutoff = V - 1
    mass = total
    if p < 1.0:
        # Preserve which side of a token boundary p lies on; FP32 multiplication
        # can round p * total back onto that boundary. Weights/scans remain FP32.
        threshold = p.to(tl.float64) * total.to(tl.float64)
        tile = tl.min(tl.where((t < TILES) & (prefixes >= threshold), t, TILES - 1), 0)
        before = tl.sum(tl.where(t == tile - 1, prefixes, 0.0), 0)
        indices = tile * BLOCK + lanes
        prefix = tl.load(CDF + row * V + indices, indices < V, float("inf")) + before
        end = tl.minimum((tile + 1) * BLOCK, V) - 1
        cutoff = tl.min(tl.where((indices < V) & (prefix >= threshold), indices, end), 0)
        mass = tl.sum(tl.where(indices == cutoff, prefix, 0.0), 0)
    target = tl.load(Draws + row) * mass
    tile = tl.min(tl.where((t < TILES) & (prefixes > target), t, TILES - 1), 0)
    before = tl.sum(tl.where(t == tile - 1, prefixes, 0.0), 0)
    indices = tile * BLOCK + lanes
    prefix = tl.load(CDF + row * V + indices, indices < V, float("inf")) + before
    # A rounded target at the upper endpoint must never select a zero-mass tail.
    local = tl.load(CDF + row * V + indices, indices < V, 0.0)
    previous = tl.load(CDF + row * V + indices - 1, (lanes > 0) & (indices < V), 0.0)
    last = tl.max(tl.where((indices < V) & (local > previous), indices, tile * BLOCK), 0)
    chosen = tl.min(tl.where((indices < V) & (prefix > target), indices, last), 0)
    chosen = tl.minimum(chosen, cutoff)
    token = tl.load(IDs + row * V + chosen)
    maximum = tl.load(Values + row * V)
    score = tl.load(Values + row * V + chosen)
    temperature = tl.maximum(tl.load(Params + row * 5 + 3), 1.1754943508222875e-38)
    logprob = (score - maximum) / temperature - tl.log(mass)
    invalid = tl.max(tl.load(Invalid + row * TILES + t, t < TILES, 0), 0) != 0
    invalid |= (logprob != logprob) | (tl.abs(logprob) == float("inf"))
    tl.store(Result + row * 3, token.to(tl.float64))
    tl.store(Result + row * 3 + 1, logprob.to(tl.float64))
    tl.store(Result + row * 3 + 2, invalid.to(tl.float64))


def update_history(addresses):
    if addresses.numel():
        _update_history[(triton.cdiv(addresses.numel(), 256),)](addresses, addresses.numel(), 256)


def sample_sorted(logits, params, metadata, draws):
    batch, vocab = logits.shape
    tiles = triton.cdiv(vocab, TILE)
    scores = torch.empty((batch, vocab), device=logits.device, dtype=torch.float32)
    invalid = torch.empty((batch, tiles), device=logits.device, dtype=torch.int32)
    # 1.penalty logits
    _transform[(batch, tiles)](logits, params, metadata, scores, invalid, vocab, tiles, TILE, enable_fp_fusion=False)
    # 2.sorted
    values, ids = scores.sort(dim=1, descending=True, stable=True)
    cdf = torch.empty_like(values)
    sums = torch.empty((batch, tiles), device=logits.device, dtype=torch.float32)
    # 3.temperature, top-k, tiled CDF
    _cdf_tiles[(batch, tiles)](values, params, metadata, cdf, sums, vocab, tiles, TILE, enable_fp_fusion=False)
    result = torch.empty((batch, 3), device=logits.device, dtype=torch.float64)
    # 4.sample
    _select[(batch,)](
        values,
        ids,
        params,
        cdf,
        sums,
        invalid,
        draws,
        result,
        vocab,
        tiles,
        TILE,
        triton.next_power_of_2(tiles),
        enable_fp_fusion=False,
    )
    return result
