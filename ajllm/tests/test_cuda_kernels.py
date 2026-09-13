"""CUDA-only numerical checks for Triton kernels against their Torch references."""

from __future__ import annotations

import pytest
import torch

from ajllm.modeling.cuda_kernels import (
    adamw_update_,
    build_variable_m_schedule,
    fused_cross_entropy,
    moe_top1_combine,
    moe_top1_dispatch,
    rms_norm,
    rope,
    row_softmax,
    squared_norm_partials,
    swiglu_gate,
    variable_grouped_gemm,
)
from ajllm.modeling.flash_attention import flash_attention, flash_attention_pytorch
from ajllm.modeling.moe import MoELayer
from ajllm.training.losses import torch_cross_entropy

CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernel checks")


@CUDA_REQUIRED
def test_rmsnorm_triton_matches_torch_forward_and_backward() -> None:
    torch.manual_seed(0)
    inputs = torch.randn(3, 5, 64, device="cuda", requires_grad=True)
    weight = torch.randn(64, device="cuda", requires_grad=True)
    doutput = torch.randn_like(inputs)
    output = rms_norm(inputs, weight, 1e-6)
    output.backward(doutput)
    input_gradient, weight_gradient = inputs.grad.clone(), weight.grad.clone()

    reference_inputs = inputs.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    reference = reference_inputs.float() * torch.rsqrt(reference_inputs.float().square().mean(-1, keepdim=True) + 1e-6)
    reference = reference * reference_weight.float()
    reference.backward(doutput)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(input_gradient, reference_inputs.grad, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(weight_gradient, reference_weight.grad, atol=2e-6, rtol=2e-6)


@CUDA_REQUIRED
def test_rope_triton_matches_torch_forward_and_backward() -> None:
    torch.manual_seed(1)
    inputs = torch.randn(2, 4, 8, 64, device="cuda", requires_grad=True)
    positions = torch.arange(8, device="cuda").view(1, 1, 8).expand(2, 4, 8)
    table_positions = torch.arange(32, device="cuda").unsqueeze(1)
    dimensions = torch.arange(0, 64, 2, device="cuda").float()
    angles = table_positions * (10_000.0 ** (-dimensions / 64))
    cosine, sine = torch.cos(angles), torch.sin(angles)
    doutput = torch.randn_like(inputs)
    output = rope(inputs, cosine, sine, positions)
    output.backward(doutput)
    input_gradient = inputs.grad.clone()

    reference_inputs = inputs.detach().clone().requires_grad_()
    even, odd = reference_inputs[..., ::2], reference_inputs[..., 1::2]
    reference = torch.stack(
        (even * cosine[positions] - odd * sine[positions], even * sine[positions] + odd * cosine[positions]), -1
    )
    reference = reference.flatten(-2)
    reference.backward(doutput)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(input_gradient, reference_inputs.grad, atol=2e-6, rtol=2e-6)


@CUDA_REQUIRED
def test_swiglu_triton_matches_torch_forward_and_backward() -> None:
    torch.manual_seed(2)
    gate = torch.randn(2, 3, 128, device="cuda", requires_grad=True)
    up = torch.randn_like(gate, requires_grad=True)
    doutput = torch.randn_like(gate)
    output = swiglu_gate(gate, up)
    output.backward(doutput)
    gate_gradient, up_gradient = gate.grad.clone(), up.grad.clone()

    reference_gate, reference_up = gate.detach().clone().requires_grad_(), up.detach().clone().requires_grad_()
    reference = reference_gate * torch.sigmoid(reference_gate) * reference_up
    reference.backward(doutput)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(gate_gradient, reference_gate.grad, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(up_gradient, reference_up.grad, atol=2e-6, rtol=2e-6)


@CUDA_REQUIRED
@pytest.mark.parametrize("width", (4, 6400))
def test_shifted_softmax_triton_matches_torch_forward_and_backward(width: int) -> None:
    torch.manual_seed(3)
    logits = torch.randn(128, width, device="cuda", requires_grad=True)
    doutput = torch.randn_like(logits)
    probabilities = row_softmax(logits)
    probabilities.backward(doutput)
    gradient = logits.grad.clone()

    reference_logits = logits.detach().clone().requires_grad_()
    reference = torch.softmax(reference_logits, dim=-1)
    reference.backward(doutput)
    torch.cuda.synchronize()
    torch.testing.assert_close(probabilities, reference, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(gradient, reference_logits.grad, atol=2e-6, rtol=2e-6)


@CUDA_REQUIRED
def test_fused_cross_entropy_triton_matches_torch_forward_and_backward() -> None:
    """Check regular and ignored labels, including the full vocabulary gradient."""
    torch.manual_seed(4)
    # 6400 is the default tokenizer vocabulary; BF16 mirrors autocast training.
    logits = torch.randn(2, 3, 6400, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    targets = torch.tensor([[1, 32, -100], [6399, 7, 17]], device="cuda")
    loss = fused_cross_entropy(logits, targets)
    loss.backward()
    gradient = logits.grad.clone()

    reference_logits = logits.detach().clone().requires_grad_()
    reference = torch_cross_entropy(reference_logits, targets)
    reference.backward()
    torch.cuda.synchronize()
    torch.testing.assert_close(loss, reference, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(gradient, reference_logits.grad, atol=2e-5, rtol=2e-5)


@CUDA_REQUIRED
def test_tiled_cross_entropy_triton_matches_torch_forward_and_backward() -> None:
    """Exercise a non-power-of-two 32k vocabulary through both tiled stages."""
    torch.manual_seed(5)
    vocabulary = 32_003
    logits = torch.randn(1, 2, vocabulary, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    targets = torch.tensor([[1, vocabulary - 1]], device="cuda")
    loss = fused_cross_entropy(logits, targets)
    loss.backward()
    gradient = logits.grad.clone()

    reference_logits = logits.detach().clone().requires_grad_()
    reference = torch_cross_entropy(reference_logits, targets)
    reference.backward()
    torch.cuda.synchronize()
    torch.testing.assert_close(loss, reference, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(gradient, reference_logits.grad, atol=2e-5, rtol=2e-5)


@CUDA_REQUIRED
def test_fused_adamw_triton_matches_explicit_torch_updates() -> None:
    """Compare parameters and both moments across several bias-corrected updates."""
    torch.manual_seed(6)
    parameter = torch.randn(513, device="cuda")
    first_moment, second_moment = torch.zeros_like(parameter), torch.zeros_like(parameter)
    reference_parameter = parameter.clone()
    reference_first, reference_second = torch.zeros_like(parameter), torch.zeros_like(parameter)
    beta1, beta2, learning_rate, epsilon, weight_decay = 0.9, 0.95, 3e-4, 1e-8, 0.1

    for step in range(1, 4):
        gradient = torch.randn_like(parameter)
        adjusted_learning_rate = learning_rate * (1 - beta2**step) ** 0.5 / (1 - beta1**step)
        adamw_update_(
            parameter,
            gradient,
            first_moment,
            second_moment,
            beta1,
            beta2,
            adjusted_learning_rate,
            epsilon,
            learning_rate,
            weight_decay,
        )
        reference_first.mul_(beta1).add_(gradient, alpha=1 - beta1)
        reference_second.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
        reference_parameter.addcdiv_(
            reference_first, reference_second.sqrt().add_(epsilon), value=-adjusted_learning_rate
        )
        reference_parameter.mul_(1 - learning_rate * weight_decay)

    torch.cuda.synchronize()
    torch.testing.assert_close(parameter, reference_parameter, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(first_moment, reference_first, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(second_moment, reference_second, atol=3e-6, rtol=3e-6)


@CUDA_REQUIRED
def test_squared_norm_partials_triton_matches_torch() -> None:
    """Exercise a masked final reduction block used by global gradient clipping."""
    torch.manual_seed(7)
    gradient = torch.randn(12_345, device="cuda", dtype=torch.bfloat16)
    squared_norm = torch.sum(squared_norm_partials(gradient))
    reference = torch.sum(gradient.float().square())
    torch.cuda.synchronize()
    torch.testing.assert_close(squared_norm, reference, atol=1e-2, rtol=2e-6)


@CUDA_REQUIRED
def test_moe_top1_dispatch_and_combine_triton_match_torch_with_backward() -> None:
    """Verify compact sorted token movement and both inverse gradients."""
    torch.manual_seed(8)
    tokens, width = 17, 64
    routes = torch.tensor([3, 0, 1, 3, 2, 0, 1, 3, 1, 2, 0, 3, 2, 1, 0, 2, 3], device="cuda")
    order = torch.argsort(routes)
    token_indices = torch.arange(tokens, device="cuda").index_select(0, order)
    inputs = torch.randn(tokens, width, device="cuda", requires_grad=True)
    dispatched = moe_top1_dispatch(inputs, token_indices)
    dispatch_gradient = torch.randn_like(dispatched)
    dispatched.backward(dispatch_gradient)
    input_gradient = inputs.grad.clone()

    reference_inputs = inputs.detach().clone().requires_grad_()
    reference_dispatched = reference_inputs.index_select(0, token_indices)
    reference_dispatched.backward(dispatch_gradient)
    torch.testing.assert_close(dispatched, reference_dispatched, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(input_gradient, reference_inputs.grad, atol=2e-6, rtol=2e-6)

    grouped = torch.randn(tokens, width, device="cuda", requires_grad=True)
    combined = moe_top1_combine(grouped, token_indices)
    combine_gradient = torch.randn_like(combined)
    combined.backward(combine_gradient)
    grouped_gradient = grouped.grad.clone()

    reference_grouped = grouped.detach().clone().requires_grad_()
    reference_combined = torch.empty((tokens, width), device="cuda")
    reference_combined.index_copy_(0, token_indices, reference_grouped)
    reference_combined.backward(combine_gradient)
    torch.cuda.synchronize()
    torch.testing.assert_close(combined, reference_combined, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(grouped_gradient, reference_grouped.grad, atol=2e-6, rtol=2e-6)


@CUDA_REQUIRED
def test_variable_m_grouped_gemm_matches_unpadded_torch_with_backward() -> None:
    """Cover unequal M values, an empty expert, tail tiles, and every gradient."""
    torch.manual_seed(9)
    offsets = torch.tensor([0, 3, 3, 20, 29], device="cuda", dtype=torch.int64)
    tile_experts, tile_rows = build_variable_m_schedule(offsets)
    inputs = torch.randn(29, 48, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(4, 70, 48, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    doutput = torch.randn(29, 70, device="cuda", dtype=torch.bfloat16)
    output = variable_grouped_gemm(inputs, weight, offsets, tile_experts, tile_rows)
    output.backward(doutput)
    input_gradient, weight_gradient = inputs.grad.clone(), weight.grad.clone()

    reference_inputs = inputs.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    rows = []
    for expert in range(4):
        start, end = offsets[expert].item(), offsets[expert + 1].item()
        rows.append(reference_inputs[start:end] @ reference_weight[expert].transpose(0, 1))
    reference = torch.cat(rows)
    reference.backward(doutput)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference, atol=0.15, rtol=0.02)
    torch.testing.assert_close(input_gradient, reference_inputs.grad, atol=0.15, rtol=0.02)
    torch.testing.assert_close(weight_gradient, reference_weight.grad, atol=0.15, rtol=0.02)


@CUDA_REQUIRED
def test_grouped_moe_top1_executes_cuda_dispatch_and_grouped_gemms() -> None:
    """Exercise the complete balanced Top-1 grouped MoE path with autograd."""
    torch.manual_seed(10)
    layer = MoELayer(64, 128, num_experts=4).cuda()
    inputs = torch.randn(32, 64, device="cuda", requires_grad=True)
    routes = torch.arange(32, device="cuda") % 4
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = layer._forward_grouped_top1(inputs, routes)
    outputs.square().mean().backward()
    assert inputs.grad is not None
    assert layer.grouped_experts.gate_up_proj.weight.grad is not None
    assert layer.grouped_experts.gate_up_proj.weight.grad.dtype == torch.float32


@CUDA_REQUIRED
def test_flash_attention_triton_matches_project_torch_implementation() -> None:
    torch.manual_seed(10)
    queries = torch.randn(1, 2, 64, 64, device="cuda", requires_grad=True)
    keys = torch.randn_like(queries, requires_grad=True)
    values = torch.randn_like(queries, requires_grad=True)
    doutput = torch.randn_like(queries)
    output = flash_attention(queries, keys, values, is_causal=True)
    output.backward(doutput)
    query_gradient = queries.grad.clone()

    reference_queries = queries.detach().clone().requires_grad_()
    reference = flash_attention_pytorch(reference_queries, keys.detach(), values.detach(), True)
    reference.backward(doutput)
    torch.cuda.synchronize()
    # Ampere Tensor Cores use TF32 for Triton's fp32 dot products; compare with
    # an absolute tolerance that captures this expected accumulation difference.
    torch.testing.assert_close(output, reference, atol=4e-3, rtol=1e-2)
    torch.testing.assert_close(query_gradient, reference_queries.grad, atol=4e-3, rtol=1e-2)
