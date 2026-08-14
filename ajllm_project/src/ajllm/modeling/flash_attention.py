"""FlashAttention-2 implementation with PyTorch and optional Triton kernels.

This module provides memory-efficient attention through tiling, avoiding the O(n²)
attention matrix materialization. Falls back to standard attention on CPU.
"""

from __future__ import annotations

import math

import torch
from einops import rearrange

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False

MASK_BIAS = -1e6  # Large negative for masked positions


def _flatten_batch(tensor: torch.Tensor) -> torch.Tensor:
    """Collapse leading dimensions into one batch dimension."""
    return rearrange(tensor, "... s d -> (...) s d")


def _causal_bias(q_index: torch.Tensor, k_index: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Additive causal mask: 0 where query can attend, MASK_BIAS elsewhere."""
    allowed = q_index[:, None] >= k_index[None, :]
    return torch.where(allowed, 0.0, MASK_BIAS).to(dtype)


def _tile_sizes_for(d: int) -> tuple[int, int]:
    """Default (B_q, B_k) tile sizes based on head dimension."""
    if d <= 64:
        return 64, 64
    return 32, 32


def flash_forward_pytorch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = False,
    q_tile_size: int = 32,
    k_tile_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tiled FlashAttention-2 forward pass in PyTorch.

    Returns (O, L) where O is output and L is logsumexp for backward.
    """
    *batch_shape, n_queries, d = Q.shape
    n_keys = K.shape[-2]
    scale = 1.0 / math.sqrt(d)

    # Flatten batch dimensions and upcast to fp32 for stable online softmax
    q_flat = _flatten_batch(Q).float()
    k_flat = _flatten_batch(K).float()
    v_flat = _flatten_batch(V).float()
    batch = q_flat.shape[0]

    out = torch.empty(batch, n_queries, d, device=Q.device, dtype=torch.float32)
    lse = torch.empty(batch, n_queries, device=Q.device, dtype=torch.float32)

    # Outer loop over query tiles
    for q_start in range(0, n_queries, q_tile_size):
        q_stop = min(q_start + q_tile_size, n_queries)
        q_tile = q_flat[:, q_start:q_stop]
        bq = q_stop - q_start

        # Initialize accumulators
        o_acc = torch.zeros(batch, bq, d, device=Q.device, dtype=torch.float32)
        l_acc = torch.zeros(batch, bq, device=Q.device, dtype=torch.float32)
        m_acc = torch.full((batch, bq), -float("inf"), device=Q.device, dtype=torch.float32)

        # Inner loop over key tiles (streaming)
        for k_start in range(0, n_keys, k_tile_size):
            k_stop = min(k_start + k_tile_size, n_keys)
            k_tile = k_flat[:, k_start:k_stop]
            v_tile = v_flat[:, k_start:k_stop]

            # Compute scores for this tile
            scores = torch.matmul(q_tile, k_tile.transpose(-1, -2)) * scale
            if is_causal:
                q_index = torch.arange(q_start, q_stop, device=Q.device)
                k_index = torch.arange(k_start, k_stop, device=Q.device)
                scores = scores + _causal_bias(q_index, k_index, scores.dtype)

            # Online softmax update
            m_new = torch.maximum(m_acc, scores.amax(dim=-1))
            p_tile = torch.exp(scores - m_new.unsqueeze(-1))
            rescale = torch.exp(m_acc - m_new)

            l_acc = rescale * l_acc + p_tile.sum(dim=-1)
            o_acc = rescale.unsqueeze(-1) * o_acc + torch.matmul(p_tile, v_tile)
            m_acc = m_new

        # Normalize and store
        out[:, q_start:q_stop] = o_acc / l_acc.unsqueeze(-1)
        lse[:, q_start:q_stop] = m_acc + torch.log(l_acc)

    out = out.reshape(*batch_shape, n_queries, d).to(Q.dtype)
    lse = lse.reshape(*batch_shape, n_queries)
    return out, lse


def flash_backward_pytorch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    out: torch.Tensor,
    dO: torch.Tensor,
    L: torch.Tensor,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass via recomputation (no saved attention matrix)."""
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    input_dtype = Q.dtype

    q = Q.float()
    k = K.float()
    v = V.float()
    do = dO.float()

    # Recompute scores and softmax
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if is_causal:
        n_queries, n_keys = Q.shape[-2], K.shape[-2]
        q_index = torch.arange(n_queries, device=Q.device)
        k_index = torch.arange(n_keys, device=Q.device)
        scores = scores + _causal_bias(q_index, k_index, scores.dtype)

    p = torch.exp(scores - L.float().unsqueeze(-1))

    # Backward through attention
    dv = torch.matmul(p.transpose(-1, -2), do)
    dp = torch.matmul(do, v.transpose(-1, -2))

    # Softmax backward
    row_sum = (out.float() * do).sum(dim=-1)
    ds = p * (dp - row_sum.unsqueeze(-1))

    dq = torch.matmul(ds, k) * scale
    dk = torch.matmul(ds.transpose(-1, -2), q) * scale
    return dq.to(input_dtype), dk.to(input_dtype), dv.to(input_dtype)


_compiled_backward = None


def _compiled_flash_backward(*args):
    """torch.compile wrapper for backward with eager fallback."""
    global _compiled_backward
    if _compiled_backward is None:
        _compiled_backward = torch.compile(flash_backward_pytorch, dynamic=False)
    try:
        return _compiled_backward(*args)
    except Exception:
        return flash_backward_pytorch(*args)


class FlashAttention2PyTorch(torch.autograd.Function):
    """FlashAttention-2 with tiled PyTorch forward and compiled backward."""

    Q_TILE_SIZE = 32
    K_TILE_SIZE = 32

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        out, lse = flash_forward_pytorch(
            Q, K, V, is_causal,
            q_tile_size=FlashAttention2PyTorch.Q_TILE_SIZE,
            k_tile_size=FlashAttention2PyTorch.K_TILE_SIZE,
        )
        ctx.save_for_backward(Q, K, V, out, lse)
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, out, lse = ctx.saved_tensors
        dq, dk, dv = _compiled_flash_backward(Q, K, V, out, dO.contiguous(), lse, ctx.is_causal)
        return dq, dk, dv, None


# Triton kernels (optional, CUDA only)
if TRITON_AVAILABLE:

    @triton.jit
    def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS, scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        """Fused forward kernel: one program per query tile."""
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        # Block pointers for this query tile
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        # Load query tile once
        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Initialize accumulators
        o_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        l_acc = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        m_acc = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

        query_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

        # Causal masking: early stop for key tiles beyond diagonal
        n_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
        if is_causal:
            n_key_tiles = tl.minimum(n_key_tiles, tl.cdiv((query_tile_index + 1) * Q_TILE_SIZE, K_TILE_SIZE))

        # Stream key/value tiles
        for key_tile_index in range(n_key_tiles):
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            # Compute scores
            scores = tl.dot(q, tl.trans(k)) * scale
            if is_causal:
                key_offsets = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
                scores += tl.where(query_offsets[:, None] >= key_offsets[None, :], 0.0, -1e6)

            # Online softmax update
            m_new = tl.maximum(m_acc, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            rescale = tl.exp(m_acc - m_new)
            l_acc = rescale * l_acc + tl.sum(p, axis=1)

            o_acc = o_acc * rescale[:, None]
            o_acc = tl.dot(p.to(v.dtype), v, acc=o_acc)
            m_acc = m_new

            # Advance to next key tile
            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        # Normalize and store
        o_acc = o_acc / l_acc[:, None]
        lse = m_acc + tl.log(l_acc)
        tl.store(O_block_ptr, o_acc.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
        tl.store(L_block_ptr, lse, boundary_check=(0,))

    @triton.jit
    def flash_bwd_dkdv_kernel(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, Delta_ptr, dK_ptr, dV_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_dob, stride_doq, stride_dod,
        stride_lb, stride_lq,
        stride_db, stride_dq_,
        stride_dkb, stride_dkk, stride_dkd,
        stride_dvb, stride_dvk, stride_dvd,
        N_QUERIES, N_KEYS, scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        """Backward kernel: compute dK and dV per key tile."""
        key_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        # Causal: skip fully masked query tiles
        first_query_tile = 0
        if is_causal:
            first_query_tile = (key_tile_index * K_TILE_SIZE) // Q_TILE_SIZE
        query_offset = first_query_tile * Q_TILE_SIZE

        # Load key and value tiles once
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        dk_acc = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
        dv_acc = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

        key_offsets = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        n_query_tiles = tl.cdiv(N_QUERIES, Q_TILE_SIZE)

        # Loop over query tiles
        for query_tile_index in range(first_query_tile, n_query_tiles):
            Q_block_ptr = tl.make_block_ptr(
                Q_ptr + batch_index * stride_qb,
                shape=(N_QUERIES, D),
                strides=(stride_qq, stride_qd),
                offsets=(query_tile_index * Q_TILE_SIZE, 0),
                block_shape=(Q_TILE_SIZE, D),
                order=(1, 0),
            )
            dO_block_ptr = tl.make_block_ptr(
                dO_ptr + batch_index * stride_dob,
                shape=(N_QUERIES, D),
                strides=(stride_doq, stride_dod),
                offsets=(query_tile_index * Q_TILE_SIZE, 0),
                block_shape=(Q_TILE_SIZE, D),
                order=(1, 0),
            )
            L_block_ptr = tl.make_block_ptr(
                L_ptr + batch_index * stride_lb,
                shape=(N_QUERIES,),
                strides=(stride_lq,),
                offsets=(query_tile_index * Q_TILE_SIZE,),
                block_shape=(Q_TILE_SIZE,),
                order=(0,),
            )
            Delta_block_ptr = tl.make_block_ptr(
                Delta_ptr + batch_index * stride_db,
                shape=(N_QUERIES,),
                strides=(stride_dq_,),
                offsets=(query_tile_index * Q_TILE_SIZE,),
                block_shape=(Q_TILE_SIZE,),
                order=(0,),
            )

            q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
            do = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
            lse = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
            delta = tl.load(Delta_block_ptr, boundary_check=(0,), padding_option="zero")

            # Recompute P
            scores = tl.dot(q, tl.trans(k)) * scale
            if is_causal:
                query_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
                scores += tl.where(query_offsets[:, None] >= key_offsets[None, :], 0.0, -1e6)
            p = tl.exp(scores - lse[:, None])

            # Accumulate gradients
            dv_acc = tl.dot(tl.trans(p).to(do.dtype), do, acc=dv_acc)
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None])
            dk_acc = tl.dot(tl.trans(ds).to(q.dtype), q, acc=dk_acc)

        # Store
        dK_block_ptr = tl.make_block_ptr(
            dK_ptr + batch_index * stride_dkb,
            shape=(N_KEYS, D),
            strides=(stride_dkk, stride_dkd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        dV_block_ptr = tl.make_block_ptr(
            dV_ptr + batch_index * stride_dvb,
            shape=(N_KEYS, D),
            strides=(stride_dvk, stride_dvd),
            offsets=(key_tile_index * K_TILE_SIZE, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        tl.store(dK_block_ptr, (dk_acc * scale).to(dK_block_ptr.type.element_ty), boundary_check=(0, 1))
        tl.store(dV_block_ptr, dv_acc.to(dV_block_ptr.type.element_ty), boundary_check=(0, 1))

    @triton.jit
    def flash_bwd_dq_kernel(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, Delta_ptr, dQ_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_dob, stride_doq, stride_dod,
        stride_lb, stride_lq,
        stride_db, stride_dq_,
        stride_dqb, stride_dqq, stride_dqd,
        N_QUERIES, N_KEYS, scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        """Backward kernel: compute dQ per query tile."""
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        # Load query tile and associated gradients
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        dO_block_ptr = tl.make_block_ptr(
            dO_ptr + batch_index * stride_dob,
            shape=(N_QUERIES, D),
            strides=(stride_doq, stride_dod),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )
        Delta_block_ptr = tl.make_block_ptr(
            Delta_ptr + batch_index * stride_db,
            shape=(N_QUERIES,),
            strides=(stride_dq_,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )

        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        do = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        lse = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        delta = tl.load(Delta_block_ptr, boundary_check=(0,), padding_option="zero")

        dq_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

        query_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        n_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)
        if is_causal:
            n_key_tiles = tl.minimum(n_key_tiles, tl.cdiv((query_tile_index + 1) * Q_TILE_SIZE, K_TILE_SIZE))

        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        # Loop over key tiles
        for key_tile_index in range(n_key_tiles):
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            # Recompute P
            scores = tl.dot(q, tl.trans(k)) * scale
            if is_causal:
                key_offsets = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
                scores += tl.where(query_offsets[:, None] >= key_offsets[None, :], 0.0, -1e6)
            p = tl.exp(scores - lse[:, None])

            # Accumulate dQ
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None])
            dq_acc = tl.dot(ds.to(k.dtype), k, acc=dq_acc)

            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        # Store
        dQ_block_ptr = tl.make_block_ptr(
            dQ_ptr + batch_index * stride_dqb,
            shape=(N_QUERIES, D),
            strides=(stride_dqq, stride_dqd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        tl.store(dQ_block_ptr, (dq_acc * scale).to(dQ_block_ptr.type.element_ty), boundary_check=(0, 1))

    def _check_triton_inputs(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> None:
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available")
        if not (Q.is_cuda and K.is_cuda and V.is_cuda):
            raise ValueError("Triton FlashAttention requires CUDA tensors")
        if not (Q.shape[-1] == K.shape[-1] == V.shape[-1]):
            raise ValueError("Q, K, V must share the same head dimension")

    def _resolve_tile_sizes(n_queries: int, n_keys: int, d: int, q_tile: int | None, k_tile: int | None) -> tuple[int, int]:
        """Pick tile sizes that divide sequence lengths and are >= 16."""
        default_q, default_k = _tile_sizes_for(d)
        q_tile = q_tile or default_q
        k_tile = k_tile or default_k
        q_tile = max(16, min(q_tile, n_queries))
        k_tile = max(16, min(k_tile, n_keys))
        if n_keys % k_tile != 0:
            raise ValueError(f"K_TILE_SIZE={k_tile} must divide n_keys={n_keys}")
        return q_tile, k_tile

    class FlashAttention2Triton(torch.autograd.Function):
        """FlashAttention-2 with Triton kernels for forward and backward."""

        Q_TILE_SIZE: int | None = None
        K_TILE_SIZE: int | None = None
        USE_TRITON_BACKWARD = True

        @staticmethod
        def forward(ctx, Q, K, V, is_causal=False):
            _check_triton_inputs(Q, K, V)

            *batch_shape, n_queries, d = Q.shape
            n_keys = K.shape[-2]
            q = _flatten_batch(Q).contiguous()
            k = _flatten_batch(K).contiguous()
            v = _flatten_batch(V).contiguous()
            batch = q.shape[0]

            q_tile, k_tile = _resolve_tile_sizes(n_queries, n_keys, d, FlashAttention2Triton.Q_TILE_SIZE, FlashAttention2Triton.K_TILE_SIZE)

            out = torch.empty_like(q)
            lse = torch.empty((batch, n_queries), device=q.device, dtype=torch.float32)

            grid = (triton.cdiv(n_queries, q_tile), batch)
            flash_fwd_kernel[grid](
                q, k, v, out, lse,
                q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                lse.stride(0), lse.stride(1),
                N_QUERIES=n_queries, N_KEYS=n_keys, scale=1.0 / math.sqrt(d),
                D=d, Q_TILE_SIZE=q_tile, K_TILE_SIZE=k_tile, is_causal=is_causal,
            )

            out = out.reshape(*batch_shape, n_queries, d)
            lse = lse.reshape(*batch_shape, n_queries)
            ctx.save_for_backward(Q, K, V, out, lse)
            ctx.is_causal = is_causal
            return out

        @staticmethod
        def backward(ctx, dO):
            Q, K, V, out, lse = ctx.saved_tensors
            if not FlashAttention2Triton.USE_TRITON_BACKWARD:
                dq, dk, dv = _compiled_flash_backward(Q, K, V, out, dO.contiguous(), lse, ctx.is_causal)
                return dq, dk, dv, None

            *batch_shape, n_queries, d = Q.shape
            n_keys = K.shape[-2]
            q = _flatten_batch(Q).contiguous()
            k = _flatten_batch(K).contiguous()
            v = _flatten_batch(V).contiguous()
            o = _flatten_batch(out).contiguous()
            do = _flatten_batch(dO).contiguous()
            lse_flat = lse.reshape(-1, n_queries).contiguous()
            batch = q.shape[0]

            q_tile, k_tile = _resolve_tile_sizes(n_queries, n_keys, d, FlashAttention2Triton.Q_TILE_SIZE, FlashAttention2Triton.K_TILE_SIZE)
            if n_queries % q_tile != 0:
                raise ValueError(f"Q_TILE_SIZE={q_tile} must divide n_queries={n_queries}")

            delta = (o.float() * do.float()).sum(dim=-1)

            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            scale = 1.0 / math.sqrt(d)

            # Pass 1: dK and dV
            flash_bwd_dkdv_kernel[(triton.cdiv(n_keys, k_tile), batch)](
                q, k, v, do, lse_flat, delta, dk, dv,
                q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                do.stride(0), do.stride(1), do.stride(2),
                lse_flat.stride(0), lse_flat.stride(1),
                delta.stride(0), delta.stride(1),
                dk.stride(0), dk.stride(1), dk.stride(2),
                dv.stride(0), dv.stride(1), dv.stride(2),
                N_QUERIES=n_queries, N_KEYS=n_keys, scale=scale,
                D=d, Q_TILE_SIZE=q_tile, K_TILE_SIZE=k_tile, is_causal=ctx.is_causal,
            )

            # Pass 2: dQ
            flash_bwd_dq_kernel[(triton.cdiv(n_queries, q_tile), batch)](
                q, k, v, do, lse_flat, delta, dq,
                q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                do.stride(0), do.stride(1), do.stride(2),
                lse_flat.stride(0), lse_flat.stride(1),
                delta.stride(0), delta.stride(1),
                dq.stride(0), dq.stride(1), dq.stride(2),
                N_QUERIES=n_queries, N_KEYS=n_keys, scale=scale,
                D=d, Q_TILE_SIZE=q_tile, K_TILE_SIZE=k_tile, is_causal=ctx.is_causal,
            )

            return (
                dq.reshape(*batch_shape, n_queries, d),
                dk.reshape(*batch_shape, n_keys, d),
                dv.reshape(*batch_shape, n_keys, d),
                None,
            )

    flash_attention_triton = FlashAttention2Triton.apply
else:
    flash_attention_triton = None


# Main entry point
flash_attention_pytorch = FlashAttention2PyTorch.apply


def flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = False,
    use_triton: bool = False,
) -> torch.Tensor:
    """FlashAttention-2 entry point with automatic backend selection.

    Args:
        Q: Query tensor (..., n_queries, d)
        K: Key tensor (..., n_keys, d)
        V: Value tensor (..., n_keys, d)
        is_causal: Whether to apply causal masking
        use_triton: Use Triton kernels if available (CUDA only)

    Returns:
        Output tensor (..., n_queries, d)
    """
    if use_triton and TRITON_AVAILABLE and Q.is_cuda:
        return flash_attention_triton(Q, K, V, is_causal)
    return flash_attention_pytorch(Q, K, V, is_causal)
