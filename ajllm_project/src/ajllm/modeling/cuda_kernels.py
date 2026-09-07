"""Small Triton kernels used by the CUDA decoder.

The kernels in this file intentionally cover only operations whose inputs and
outputs are already materialized. Variable-M MoE GEMMs are the exception: they
select expert-weight tiles directly from compact routed token ranges.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _next_power_of_two(value: int) -> int:
    """Return Triton's required power-of-two block width."""
    return 1 << (value - 1).bit_length()


_CROSS_ENTROPY_TILE = 1024
_CROSS_ENTROPY_MAX_TILES = 65536
_MOE_TRANSFER_SIZE = 1024
# Larger Ampere Tensor Core tiles amortize schedule loads for long expert ranges.
_VARIABLE_GROUPED_BLOCK_M = 128
_VARIABLE_GROUPED_BLOCK_N = 128
_VARIABLE_GROUPED_BLOCK_K = 32
# dWeight has a reduction over routed rows; a narrower output tile preserves
# occupancy while the forward path benefits from the wider N tile above.


if True:

    @triton.jit
    def _rmsnorm_forward_kernel(X, W, Y, RSTD, N, EPS: tl.constexpr, BLOCK: tl.constexpr):
        """Normalize one row per program and multiply it by the learned weight."""
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        mask = columns < N
        x = tl.load(X + row * N + columns, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(W + columns, mask=mask, other=0.0).to(tl.float32)

        # All lanes reduce their x^2 values to one scalar inverse RMS for the row.
        inverse_rms = tl.rsqrt(tl.sum(x * x, axis=0) / N + EPS)
        tl.store(Y + row * N + columns, x * inverse_rms * weight, mask=mask)
        tl.store(RSTD + row, inverse_rms)


    @triton.jit
    def _rmsnorm_input_backward_kernel(DY, X, W, RSTD, DX, N, BLOCK: tl.constexpr):
        """Compute the complete RMSNorm input gradient for one row per program."""
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        mask = columns < N
        x = tl.load(X + row * N + columns, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + row * N + columns, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(W + columns, mask=mask, other=0.0).to(tl.float32)
        inverse_rms = tl.load(RSTD + row)

        # d(x * r) / dx = r * (I - x x^T r^2 / N).
        scaled_gradient = dy * weight
        row_dot = tl.sum(scaled_gradient * x, axis=0) / N
        dx = inverse_rms * (scaled_gradient - x * inverse_rms * inverse_rms * row_dot)
        tl.store(DX + row * N + columns, dx, mask=mask)


    @triton.jit
    def _rmsnorm_weight_backward_kernel(DY, X, RSTD, DW, M, N, BLOCK_ROWS: tl.constexpr, BLOCK_N: tl.constexpr):
        """Accumulate one tile of dW; row tiles atomically add into the shared weight gradient."""
        row_block = tl.program_id(0)
        column_block = tl.program_id(1)
        rows = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        columns = column_block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < M) & (columns[None, :] < N)
        offsets = rows[:, None] * N + columns[None, :]
        x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + offsets, mask=mask, other=0.0).to(tl.float32)
        inverse_rms = tl.load(RSTD + rows, mask=rows < M, other=0.0).to(tl.float32)

        # Every program owns a [BLOCK_ROWS, BLOCK_N] tile, then sums it over rows.
        partial_dw = tl.sum(dy * x * inverse_rms[:, None], axis=0)
        tl.atomic_add(DW + columns, partial_dw, mask=columns < N)


    @triton.jit
    def _rope_forward_kernel(X, COS, SIN, POSITIONS, Y, H, T, D2, POS_B, POS_H, POS_T, BLOCK: tl.constexpr):
        """Rotate all even/odd pairs for one [batch, head, token] row per program."""
        row = tl.program_id(0)
        pairs = tl.arange(0, BLOCK)
        pair_mask = pairs < D2
        token = row % T
        head = (row // T) % H
        batch = row // (H * T)
        position = tl.load(POSITIONS + batch * POS_B + head * POS_H + token * POS_T)
        cosine = tl.load(COS + position * D2 + pairs, mask=pair_mask, other=0.0).to(tl.float32)
        sine = tl.load(SIN + position * D2 + pairs, mask=pair_mask, other=0.0).to(tl.float32)
        even = tl.load(X + row * (2 * D2) + 2 * pairs, mask=pair_mask, other=0.0).to(tl.float32)
        odd = tl.load(X + row * (2 * D2) + 2 * pairs + 1, mask=pair_mask, other=0.0).to(tl.float32)

        # Adjacent lanes hold an RoPE pair; no intermediate stack/flatten tensors are created.
        tl.store(Y + row * (2 * D2) + 2 * pairs, even * cosine - odd * sine, mask=pair_mask)
        tl.store(Y + row * (2 * D2) + 2 * pairs + 1, even * sine + odd * cosine, mask=pair_mask)


    @triton.jit
    def _rope_backward_kernel(DY, COS, SIN, POSITIONS, DX, H, T, D2, POS_B, POS_H, POS_T, BLOCK: tl.constexpr):
        """Apply the transpose of the 2D rotation to one gradient row per program."""
        row = tl.program_id(0)
        pairs = tl.arange(0, BLOCK)
        pair_mask = pairs < D2
        token = row % T
        head = (row // T) % H
        batch = row // (H * T)
        position = tl.load(POSITIONS + batch * POS_B + head * POS_H + token * POS_T)
        cosine = tl.load(COS + position * D2 + pairs, mask=pair_mask, other=0.0).to(tl.float32)
        sine = tl.load(SIN + position * D2 + pairs, mask=pair_mask, other=0.0).to(tl.float32)
        dy_even = tl.load(DY + row * (2 * D2) + 2 * pairs, mask=pair_mask, other=0.0).to(tl.float32)
        dy_odd = tl.load(DY + row * (2 * D2) + 2 * pairs + 1, mask=pair_mask, other=0.0).to(tl.float32)

        tl.store(DX + row * (2 * D2) + 2 * pairs, dy_even * cosine + dy_odd * sine, mask=pair_mask)
        tl.store(DX + row * (2 * D2) + 2 * pairs + 1, -dy_even * sine + dy_odd * cosine, mask=pair_mask)


    @triton.jit
    def _swiglu_forward_kernel(
        GATE, UP, OUTPUT, WIDTH, GATE_ROW_STRIDE, UP_ROW_STRIDE, BLOCK: tl.constexpr
    ):
        """Fuse SiLU and the gate/up product, including strided projection views."""
        row = tl.program_id(0)
        columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = columns < WIDTH
        gate = tl.load(GATE + row * GATE_ROW_STRIDE + columns, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(UP + row * UP_ROW_STRIDE + columns, mask=mask, other=0.0).to(tl.float32)
        sigmoid = 1.0 / (1.0 + tl.exp(-gate))
        tl.store(OUTPUT + row * WIDTH + columns, gate * sigmoid * up, mask=mask)


    @triton.jit
    def _swiglu_backward_kernel(
        GATE, UP, DOUTPUT, DGATE, DUP, WIDTH, GATE_ROW_STRIDE, UP_ROW_STRIDE, BLOCK: tl.constexpr
    ):
        """Compute gradients for both inputs in the same elementwise program."""
        row = tl.program_id(0)
        columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = columns < WIDTH
        gate = tl.load(GATE + row * GATE_ROW_STRIDE + columns, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(UP + row * UP_ROW_STRIDE + columns, mask=mask, other=0.0).to(tl.float32)
        offsets = row * WIDTH + columns
        doutput = tl.load(DOUTPUT + offsets, mask=mask, other=0.0).to(tl.float32)
        sigmoid = 1.0 / (1.0 + tl.exp(-gate))
        silu = gate * sigmoid
        dsilu = sigmoid + gate * sigmoid * (1.0 - sigmoid)
        tl.store(DGATE + offsets, doutput * up * dsilu, mask=mask)
        tl.store(DUP + offsets, doutput * silu, mask=mask)


    @triton.jit
    def _adamw_update_kernel(
        PARAMETERS,
        GRADIENTS,
        FIRST_MOMENTS,
        SECOND_MOMENTS,
        N_ELEMENTS,
        BETA1,
        BETA2,
        ADJUSTED_LR,
        EPSILON,
        LEARNING_RATE,
        WEIGHT_DECAY,
        BLOCK: tl.constexpr,
    ):
        """Fuse one AdamW parameter tensor's moment and decoupled-decay update."""
        block = tl.program_id(0)
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N_ELEMENTS
        parameter = tl.load(PARAMETERS + offsets, mask=mask, other=0.0).to(tl.float32)
        gradient = tl.load(GRADIENTS + offsets, mask=mask, other=0.0).to(tl.float32)
        first_moment = tl.load(FIRST_MOMENTS + offsets, mask=mask, other=0.0).to(tl.float32)
        second_moment = tl.load(SECOND_MOMENTS + offsets, mask=mask, other=0.0).to(tl.float32)

        first_moment = BETA1 * first_moment + (1.0 - BETA1) * gradient
        second_moment = BETA2 * second_moment + (1.0 - BETA2) * gradient * gradient
        parameter = parameter - ADJUSTED_LR * first_moment / (tl.sqrt(second_moment) + EPSILON)
        # Preserve this project's existing update order: adaptive step, then decay.
        parameter = parameter * (1.0 - LEARNING_RATE * WEIGHT_DECAY)

        tl.store(PARAMETERS + offsets, parameter, mask=mask)
        tl.store(FIRST_MOMENTS + offsets, first_moment, mask=mask)
        tl.store(SECOND_MOMENTS + offsets, second_moment, mask=mask)


    @triton.jit
    def _squared_norm_kernel(INPUTS, PARTIAL_SUMS, N_ELEMENTS, BLOCK: tl.constexpr):
        """Accumulate one FP32 squared-norm partial for a contiguous tensor block."""
        block = tl.program_id(0)
        offsets = block * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(INPUTS + offsets, mask=offsets < N_ELEMENTS, other=0.0).to(tl.float32)
        tl.store(PARTIAL_SUMS + block, tl.sum(values * values, axis=0))


    @triton.jit
    def _moe_top1_dispatch_kernel(SOURCE, TOKEN_INDICES, DESTINATION, WIDTH, BLOCK: tl.constexpr):
        """Gather routed rows into a compact expert-sorted token matrix."""
        position = tl.program_id(0)
        columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = columns < WIDTH
        token = tl.load(TOKEN_INDICES + position)
        values = tl.load(SOURCE + token * WIDTH + columns, mask=mask, other=0.0)
        tl.store(DESTINATION + position * WIDTH + columns, values, mask=mask)


    @triton.jit
    def _moe_top1_combine_kernel(SOURCE, TOKEN_INDICES, DESTINATION, WIDTH, BLOCK: tl.constexpr):
        """Scatter compact expert rows back to their original token positions."""
        position = tl.program_id(0)
        columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = columns < WIDTH
        token = tl.load(TOKEN_INDICES + position)
        values = tl.load(SOURCE + position * WIDTH + columns, mask=mask, other=0.0)
        tl.store(DESTINATION + token * WIDTH + columns, values, mask=mask)


    @triton.jit
    def _variable_grouped_gemm_forward_kernel(
        A,
        WEIGHT,
        OFFSETS,
        TILE_EXPERTS,
        TILE_ROWS,
        C,
        K,
        N,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Compute one real Variable-M expert tile; no global expert capacity exists."""
        m_tile = tl.program_id(0)
        n_tile = tl.program_id(1)
        expert = tl.load(TILE_EXPERTS + m_tile)
        row_start = tl.load(TILE_ROWS + m_tile)
        row_end = tl.load(OFFSETS + expert + 1)
        rows = row_start + tl.arange(0, BLOCK_M)
        columns = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        reduction = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        k_start = 0
        while k_start < K:
            a = tl.load(
                A + rows[:, None] * K + k_start + reduction[None, :],
                mask=(rows[:, None] < row_end) & (k_start + reduction[None, :] < K),
                other=0.0,
            )
            weight = tl.load(
                WEIGHT + expert * N * K + columns[:, None] * K + k_start + reduction[None, :],
                mask=(columns[:, None] < N) & (k_start + reduction[None, :] < K),
                other=0.0,
            )
            accumulator += tl.dot(a, tl.trans(weight))
            k_start += BLOCK_K
        tl.store(
            C + rows[:, None] * N + columns[None, :],
            accumulator,
            mask=(rows[:, None] < row_end) & (columns[None, :] < N),
        )


    @triton.jit
    def _softmax_forward_kernel(X, Y, N, BLOCK: tl.constexpr):
        """Compute one stable shifted Softmax row per program."""
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        mask = columns < N
        logits = tl.load(X + row * N + columns, mask=mask, other=-float("inf")).to(tl.float32)

        # A single program performs max, exp, sum, and normalization for one token.
        shifted = logits - tl.max(logits, axis=0)
        exponentials = tl.exp(shifted)
        tl.store(Y + row * N + columns, exponentials / tl.sum(exponentials, axis=0), mask=mask)


    @triton.jit
    def _softmax_backward_kernel(PROBABILITIES, DOUTPUT, DINPUT, N, BLOCK: tl.constexpr):
        """Apply dSoftmax = p * (dy - sum(dy * p)) for one row per program."""
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        mask = columns < N
        probabilities = tl.load(PROBABILITIES + row * N + columns, mask=mask, other=0.0).to(tl.float32)
        doutput = tl.load(DOUTPUT + row * N + columns, mask=mask, other=0.0).to(tl.float32)
        dot = tl.sum(probabilities * doutput, axis=0)
        tl.store(DINPUT + row * N + columns, probabilities * (doutput - dot), mask=mask)


    @triton.jit
    def _cross_entropy_partial_stats_kernel(
        LOGITS, PARTIAL_MAXIMUMS, PARTIAL_SUMS, VOCABULARY, NUM_TILES, TILE: tl.constexpr
    ):
        """Write max and shifted-exp sum for one vocabulary tile of one token row."""
        row = tl.program_id(0)
        tile = tl.program_id(1)
        columns = tile * TILE + tl.arange(0, TILE)
        mask = columns < VOCABULARY
        logits = tl.load(LOGITS + row * VOCABULARY + columns, mask=mask, other=-float("inf")).to(tl.float32)

        # Each tile has its own stable statistics, which can later be merged exactly.
        local_maximum = tl.max(logits, axis=0)
        local_sum = tl.sum(tl.exp(logits - local_maximum), axis=0)
        offset = row * NUM_TILES + tile
        tl.store(PARTIAL_MAXIMUMS + offset, local_maximum)
        tl.store(PARTIAL_SUMS + offset, local_sum)


    @triton.jit
    def _cross_entropy_reduce_tiles_kernel(
        LOGITS,
        TARGETS,
        PARTIAL_MAXIMUMS,
        PARTIAL_SUMS,
        LOSSES,
        ROW_MAXIMUMS,
        ROW_SUMS,
        VOCABULARY,
        NUM_TILES,
        BLOCK: tl.constexpr,
    ):
        """Merge tile statistics into one stable loss and reusable row statistics."""
        row = tl.program_id(0)
        tiles = tl.arange(0, BLOCK)
        tile_mask = tiles < NUM_TILES
        offsets = row * NUM_TILES + tiles
        partial_maximums = tl.load(PARTIAL_MAXIMUMS + offsets, mask=tile_mask, other=-float("inf"))
        partial_sums = tl.load(PARTIAL_SUMS + offsets, mask=tile_mask, other=0.0)
        maximum = tl.max(partial_maximums, axis=0)
        normalizer = tl.sum(partial_sums * tl.exp(partial_maximums - maximum), axis=0)
        target = tl.load(TARGETS + row)
        valid = target != -100
        safe_target = tl.where(valid, target, 0)
        target_logit = tl.load(LOGITS + row * VOCABULARY + safe_target).to(tl.float32)
        loss = maximum + tl.log(normalizer) - target_logit

        tl.store(LOSSES + row, tl.where(valid, loss, 0.0))
        tl.store(ROW_MAXIMUMS + row, maximum)
        tl.store(ROW_SUMS + row, normalizer)


    @triton.jit
    def _cross_entropy_tiled_backward_kernel(
        LOGITS,
        TARGETS,
        DLOSSES,
        ROW_MAXIMUMS,
        ROW_SUMS,
        DLOGITS,
        VOCABULARY,
        TILE: tl.constexpr,
    ):
        """Write the gradient for one vocabulary tile, using saved global statistics."""
        row = tl.program_id(0)
        tile = tl.program_id(1)
        columns = tile * TILE + tl.arange(0, TILE)
        mask = columns < VOCABULARY
        logits = tl.load(LOGITS + row * VOCABULARY + columns, mask=mask, other=-float("inf")).to(tl.float32)
        target = tl.load(TARGETS + row)
        valid = target != -100
        safe_target = tl.where(valid, target, 0)
        maximum = tl.load(ROW_MAXIMUMS + row)
        normalizer = tl.load(ROW_SUMS + row)
        upstream = tl.load(DLOSSES + row).to(tl.float32)
        probabilities = tl.exp(logits - maximum) / normalizer
        target_mask = columns == safe_target
        gradient = tl.where(valid, upstream * (probabilities - target_mask.to(tl.float32)), 0.0)
        tl.store(DLOGITS + row * VOCABULARY + columns, gradient, mask=mask)


class _RMSNormTriton(torch.autograd.Function):
    """Autograd wrapper around the RMSNorm forward and backward kernels."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
        rows, width = inputs.numel() // inputs.shape[-1], inputs.shape[-1]
        block = _next_power_of_two(width)
        inputs_2d = inputs.contiguous().view(rows, width)
        output = torch.empty_like(inputs_2d)
        inverse_rms = torch.empty(rows, device=inputs.device, dtype=torch.float32)
        _rmsnorm_forward_kernel[(rows,)](inputs_2d, weight, output, inverse_rms, width, EPS=epsilon, BLOCK=block)
        ctx.save_for_backward(inputs_2d, weight, inverse_rms)
        ctx.width = width
        return output.view_as(inputs)

    @staticmethod
    def backward(ctx, doutput: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        inputs, weight, inverse_rms = ctx.saved_tensors
        rows, width = inputs.shape
        block = _next_power_of_two(width)
        doutput_2d = doutput.contiguous().view(rows, width)
        dinputs = torch.empty_like(inputs)
        dweight = torch.zeros_like(weight, dtype=torch.float32)
        _rmsnorm_input_backward_kernel[(rows,)](
            doutput_2d, inputs, weight, inverse_rms, dinputs, width, BLOCK=block
        )
        _rmsnorm_weight_backward_kernel[(triton.cdiv(rows, 32), triton.cdiv(width, 256))](
            doutput_2d, inputs, inverse_rms, dweight, rows, width, BLOCK_ROWS=32, BLOCK_N=256
        )
        return dinputs.view_as(doutput), dweight.to(weight.dtype), None


class _RoPETriton(torch.autograd.Function):
    """Autograd wrapper for RoPE; cosine and sine tables are constant buffers."""

    @staticmethod
    def forward(
        ctx, inputs: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        batch, heads, sequence, width = inputs.shape
        pair_width = width // 2
        contiguous_inputs = inputs.contiguous()
        output = torch.empty_like(contiguous_inputs)
        _rope_forward_kernel[(batch * heads * sequence,)](
            contiguous_inputs,
            cosine,
            sine,
            positions,
            output,
            heads,
            sequence,
            pair_width,
            positions.stride(0),
            positions.stride(1),
            positions.stride(2),
            BLOCK=_next_power_of_two(pair_width),
        )
        ctx.save_for_backward(cosine, sine, positions)
        ctx.shape = (batch, heads, sequence, width)
        return output

    @staticmethod
    def backward(ctx, doutput: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        cosine, sine, positions = ctx.saved_tensors
        batch, heads, sequence, width = ctx.shape
        pair_width = width // 2
        dinputs = torch.empty_like(doutput.contiguous())
        _rope_backward_kernel[(batch * heads * sequence,)](
            doutput.contiguous(),
            cosine,
            sine,
            positions,
            dinputs,
            heads,
            sequence,
            pair_width,
            positions.stride(0),
            positions.stride(1),
            positions.stride(2),
            BLOCK=_next_power_of_two(pair_width),
        )
        return dinputs, None, None, None


class _SwiGLUTriton(torch.autograd.Function):
    """Autograd wrapper for the fused SwiGLU pointwise gate."""

    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if gate.shape != up.shape or gate.stride(-1) != 1 or up.stride(-1) != 1:
            raise ValueError("SwiGLU inputs must have equal shapes and contiguous last dimensions")
        width = gate.shape[-1]
        rows = gate.numel() // width
        gate_row_stride = gate.stride(-2) if gate.ndim > 1 else width
        up_row_stride = up.stride(-2) if up.ndim > 1 else width
        output = torch.empty(gate.shape, device=gate.device, dtype=gate.dtype)
        _swiglu_forward_kernel[(rows, triton.cdiv(width, 256))](
            gate, up, output, width, gate_row_stride, up_row_stride, BLOCK=256
        )
        ctx.save_for_backward(gate, up)
        ctx.width = width
        ctx.rows = rows
        ctx.gate_row_stride = gate_row_stride
        ctx.up_row_stride = up_row_stride
        return output

    @staticmethod
    def backward(ctx, doutput: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate, up = ctx.saved_tensors
        dgate = torch.empty(gate.shape, device=gate.device, dtype=gate.dtype)
        dup = torch.empty(up.shape, device=up.device, dtype=up.dtype)
        _swiglu_backward_kernel[(ctx.rows, triton.cdiv(ctx.width, 256))](
            gate,
            up,
            doutput.contiguous(),
            dgate,
            dup,
            ctx.width,
            ctx.gate_row_stride,
            ctx.up_row_stride,
            BLOCK=256,
        )
        return dgate, dup


class _SoftmaxTriton(torch.autograd.Function):
    """Autograd wrapper for the standalone row-wise Softmax utility."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
        rows, width = inputs.shape
        output = torch.empty_like(inputs)
        warps = 1 if width <= 32 else 4
        _softmax_forward_kernel[(rows,)](
            inputs, output, width, BLOCK=_next_power_of_two(width), num_warps=warps
        )
        ctx.save_for_backward(output)
        ctx.warps = warps
        return output

    @staticmethod
    def backward(ctx, doutput: torch.Tensor) -> torch.Tensor:
        (probabilities,) = ctx.saved_tensors
        dinputs = torch.empty_like(probabilities)
        rows, width = probabilities.shape
        _softmax_backward_kernel[(rows,)](
            probabilities,
            doutput.contiguous(),
            dinputs,
            width,
            BLOCK=_next_power_of_two(width),
            num_warps=ctx.warps,
        )
        return dinputs


class _CrossEntropyTiledTriton(torch.autograd.Function):
    """Architecture-tuned tiled cross entropy for every supported vocabulary size."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        rows, vocabulary = logits.shape
        tile, tile_warps = _cross_entropy_launch_config(logits.device)
        tiles = (vocabulary + tile - 1) // tile
        partial_maximums = torch.empty((rows, tiles), device=logits.device, dtype=torch.float32)
        partial_sums = torch.empty_like(partial_maximums)
        _cross_entropy_partial_stats_kernel[(rows, tiles)](
            logits,
            partial_maximums,
            partial_sums,
            vocabulary,
            tiles,
            TILE=tile,
            num_warps=tile_warps,
        )
        losses = torch.empty(rows, device=logits.device, dtype=torch.float32)
        row_maximums = torch.empty_like(losses)
        row_sums = torch.empty_like(losses)
        reduction_block = _next_power_of_two(tiles)
        _cross_entropy_reduce_tiles_kernel[(rows,)](
            logits,
            targets,
            partial_maximums,
            partial_sums,
            losses,
            row_maximums,
            row_sums,
            vocabulary,
            tiles,
            BLOCK=reduction_block,
            num_warps=_cross_entropy_reduction_warps(reduction_block),
        )
        ctx.save_for_backward(logits, targets, row_maximums, row_sums)
        ctx.tiles, ctx.tile, ctx.tile_warps = tiles, tile, tile_warps
        return losses

    @staticmethod
    def backward(ctx, dlosses: torch.Tensor) -> tuple[torch.Tensor, None]:
        logits, targets, row_maximums, row_sums = ctx.saved_tensors
        rows, vocabulary = logits.shape
        dlogits = torch.empty_like(logits)
        _cross_entropy_tiled_backward_kernel[(rows, ctx.tiles)](
            logits,
            targets,
            dlosses.contiguous(),
            row_maximums,
            row_sums,
            dlogits,
            vocabulary,
            TILE=ctx.tile,
            num_warps=ctx.tile_warps,
        )
        return dlogits, None


class _MoETop1DispatchTriton(torch.autograd.Function):
    """Autograd wrapper for Top-1 gather into compact expert-sorted storage."""

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        _, width = inputs.shape
        output = torch.empty_like(inputs)
        _moe_top1_dispatch_kernel[(token_indices.numel(), triton.cdiv(width, 256))](
            inputs, token_indices, output, width, BLOCK=256, num_warps=4
        )
        ctx.save_for_backward(token_indices)
        return output

    @staticmethod
    def backward(ctx, doutput: torch.Tensor) -> tuple[torch.Tensor, None]:
        (token_indices,) = ctx.saved_tensors
        tokens, width = doutput.shape
        dinputs = torch.empty_like(doutput)
        _moe_top1_combine_kernel[(token_indices.numel(), triton.cdiv(width, 256))](
            doutput.contiguous(), token_indices, dinputs, width, BLOCK=256, num_warps=4
        )
        return dinputs, None


class _MoETop1CombineTriton(torch.autograd.Function):
    """Autograd wrapper for Top-1 scatter from compact expert-sorted storage."""

    @staticmethod
    def forward(
        ctx, grouped_outputs: torch.Tensor, token_indices: torch.Tensor
    ) -> torch.Tensor:
        _, width = grouped_outputs.shape
        output = torch.empty_like(grouped_outputs)
        _moe_top1_combine_kernel[(token_indices.numel(), triton.cdiv(width, _MOE_TRANSFER_SIZE))](
                    grouped_outputs.contiguous(), token_indices, output, width, BLOCK=_MOE_TRANSFER_SIZE
                )
        ctx.save_for_backward(token_indices)
        return output

    @staticmethod
    def backward(ctx, doutput: torch.Tensor) -> tuple[torch.Tensor, None]:
        (token_indices,) = ctx.saved_tensors
        _, width = doutput.shape
        dgrouped = torch.empty_like(doutput)
        _moe_top1_dispatch_kernel[(token_indices.numel(), triton.cdiv(width, _MOE_TRANSFER_SIZE))](
                    doutput.contiguous(), token_indices, dgrouped, width, BLOCK=_MOE_TRANSFER_SIZE
                )
        return dgrouped, None


class _VariableGroupedGEMMTriton(torch.autograd.Function):
    """Autograd wrapper for compact Variable-M expert matrix multiplication."""

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        expert_offsets: torch.Tensor,
        tile_experts: torch.Tensor,
        tile_rows: torch.Tensor,
    ) -> torch.Tensor:
        tokens, input_width = inputs.shape
        experts, output_width, weight_width = weight.shape
        if input_width != weight_width:
            raise ValueError("Grouped GEMM input and weight dimensions do not match")
        output = torch.empty((tokens, output_width), device=inputs.device, dtype=inputs.dtype)
        block_m = _VARIABLE_GROUPED_BLOCK_M
        block_n = _VARIABLE_GROUPED_BLOCK_N
        block_k = _VARIABLE_GROUPED_BLOCK_K
        # (tile_num, ceil(dff / N))
        grid = (tile_experts.numel(), triton.cdiv(output_width, block_n))
        _variable_grouped_gemm_forward_kernel[grid](
            inputs,
            weight,
            expert_offsets,
            tile_experts,
            tile_rows,
            output,
            input_width,
            output_width,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=3,
        )
        ctx.save_for_backward(inputs, weight, expert_offsets, tile_experts, tile_rows)
        ctx.dimensions = (experts, output_width, input_width)
        return output

    @staticmethod
    def backward(ctx, doutput: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:
        inputs, weight, expert_offsets, tile_experts, tile_rows = ctx.saved_tensors
        experts, _, _ = ctx.dimensions
        doutput = doutput.contiguous()
        dinputs = torch.empty_like(inputs)
        dweight = torch.empty_like(weight)
        # The forward kernel benefits from Variable-M scheduling.  Its former
        # dWeight kernel instead serialized a long routed-row reduction in one
        # program. Four explicit GEMM equations let cuBLAS use Tensor Core
        # split-K reductions for the small fixed expert count.
        offset_values = expert_offsets.detach().cpu()
        for expert in range(experts):
            start, end = int(offset_values[expert]), int(offset_values[expert + 1])
            expert_inputs = inputs[start:end]
            expert_doutput = doutput[start:end]
            expert_weight = weight[expert]
            dinputs[start:end] = expert_doutput @ expert_weight
            dweight[expert] = expert_doutput.transpose(0, 1) @ expert_inputs
        return dinputs, dweight, None, None, None


def _cross_entropy_launch_config(device: torch.device) -> tuple[int, int]:
    """Choose a vocabulary-tile shape that matches the available CUDA architecture.

    Larger, newer GPUs can amortize launch overhead over more vocabulary values.
    Older architectures use smaller reductions to protect occupancy and registers.
    The Ampere setting is measured on the project's RTX 3080 Ti (SM 8.6).
    """
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 9:  # Hopper or newer: more register file and scheduling capacity.
        return 2048, 8
    if major >= 8:  # Ampere/Ada: balanced 1024-value reduction.
        return _CROSS_ENTROPY_TILE, 4
    if major >= 7:  # Turing/Volta: prefer a lower register footprint.
        return 512, 4
    return 256, 4


def _cross_entropy_reduction_warps(block: int) -> int:
    """Avoid assigning idle warps when a row has only a few vocabulary tiles."""
    if block <= 32:
        return 1
    return 4 if block <= 1024 else 8


def adamw_update_(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    first_moment: torch.Tensor,
    second_moment: torch.Tensor,
    beta1: float,
    beta2: float,
    adjusted_learning_rate: float,
    epsilon: float,
    learning_rate: float,
    weight_decay: float,
) -> None:
    """Apply this project's AdamW equation to one dense parameter tensor in one launch."""
    elements = parameter.numel()
    _adamw_update_kernel[(triton.cdiv(elements, 256),)](
        parameter,
        gradient,
        first_moment,
        second_moment,
        elements,
        beta1,
        beta2,
        adjusted_learning_rate,
        epsilon,
        learning_rate,
        weight_decay,
        BLOCK=256,
    )


def squared_norm_partials(inputs: torch.Tensor) -> torch.Tensor:
    """Return FP32 block partials for a tensor's squared norm without a square tensor."""
    elements, block = inputs.numel(), 1024
    partials = torch.empty(triton.cdiv(elements, block), device=inputs.device, dtype=torch.float32)
    _squared_norm_kernel[(partials.numel(),)](inputs, partials, elements, BLOCK=block, num_warps=4)
    return partials


def moe_top1_dispatch(inputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
    """Gather Top-1 tokens into one compact expert-sorted matrix."""
    return _MoETop1DispatchTriton.apply(inputs, token_indices)


def moe_top1_combine(grouped_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
    """Restore uniquely routed Top-1 expert outputs to original token order."""
    return _MoETop1CombineTriton.apply(grouped_outputs, token_indices)


def build_variable_m_schedule(expert_offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map each real M tile to its expert and first compact-token row.

    Only the final tile of each expert can contain masked alignment rows. There
    is no maximum-capacity dimension and no tile is emitted for an empty expert.
    """
    # the number of tokens for each expert, e.g. [3,2,4,1] 
    counts = expert_offsets[1:] - expert_offsets[:-1]
    # e.g. [2,1,2,1] for 4 experts with 3,2,4,1 rows, suppose M=2
    tiles_per_expert = (counts + _VARIABLE_GROUPED_BLOCK_M - 1) // _VARIABLE_GROUPED_BLOCK_M
    # e.g. [0,1,2,3] for 4 experts
    experts = torch.arange(counts.numel(), device=expert_offsets.device, dtype=torch.int64)
    # repeat each expert index according to how many tiles it has, e.g. [0,0,1,2,2,3]
    tile_experts = torch.repeat_interleave(experts, tiles_per_expert)
    # compute the first tile for each expert, e.g. [0,2,3,5] for 4 experts with 2,1,2,1 tiles
    first_tile = torch.cumsum(tiles_per_expert, dim=0) - tiles_per_expert
    local_tiles = torch.arange(tile_experts.numel(), device=expert_offsets.device) - first_tile[tile_experts]
    # compute the first token row of each tile, e.g. [0,2,3,5,7,8] for 4 experts with 3,2,4,1 rows and M=2
    tile_rows = expert_offsets[tile_experts] + local_tiles * _VARIABLE_GROUPED_BLOCK_M
    return tile_experts.to(torch.int32), tile_rows.to(torch.int32)


def variable_grouped_gemm(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    expert_offsets: torch.Tensor,
    tile_experts: torch.Tensor,
    tile_rows: torch.Tensor,
) -> torch.Tensor:
    """Multiply compact Variable-M rows by the corresponding expert weights."""
    return _VariableGroupedGEMMTriton.apply(inputs, weight, expert_offsets, tile_experts, tile_rows)


def rms_norm(inputs: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Run fused RMSNorm."""
    return _RMSNormTriton.apply(inputs, weight, epsilon)


def rope(inputs: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Run fused RoPE for a four-dimensional ``[B, heads, T, D]`` tensor."""
    return _RoPETriton.apply(inputs, cosine, sine, positions)


def swiglu_gate(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Run fused ``SiLU(gate) * up`` with its paired backward kernel."""
    return _SwiGLUTriton.apply(gate, up)


def row_softmax(inputs: torch.Tensor) -> torch.Tensor:
    """Run a stable shifted row-wise Softmax utility kernel."""
    return _SoftmaxTriton.apply(inputs)


def fused_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return mean causal cross entropy from fused vocabulary-row kernels.

    The implementation accepts a contiguous ``[rows, vocabulary]`` view and
    ignores target ``-100``. Every vocabulary size uses stable tile statistics
    followed by one row reduction program.
    """
    vocabulary = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocabulary).contiguous()
    flat_targets = targets.reshape(-1).contiguous()
    valid = flat_targets != -100
    if not torch.any(valid):
        return flat_logits.sum() * 0.0
    losses = _CrossEntropyTiledTriton.apply(flat_logits, flat_targets)
    return torch.sum(losses) / torch.sum(valid)
