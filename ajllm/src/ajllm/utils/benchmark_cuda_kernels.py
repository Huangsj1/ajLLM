"""Benchmark the project's Torch and Triton CUDA paths at configurable shapes."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ajllm.modeling.activations import VariableGroupedSwiGLUExperts, silu
from ajllm.modeling.cuda_kernels import (
    adamw_update_,
    build_variable_m_schedule,
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
from ajllm.training.losses import cross_entropy, torch_cross_entropy

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@dataclass(frozen=True)
class BenchmarkConfig:
    """Shapes and timing controls shared by the core and MoE benchmarks."""

    batch_size: int = 16
    sequence_length: int = 512
    vocabulary_size: int = 6400
    hidden_size: int = 768
    num_heads: int = 12
    intermediate_size: int = 2432
    num_experts: int = 4
    primary_expert_fraction: float | None = None
    dtype: torch.dtype = torch.bfloat16
    warmup: int = 5
    iterations: int = 20
    section: str = "all"

    def validate(self) -> None:
        """Reject shapes that do not match the kernels' basic contracts."""
        positive = (
            self.batch_size,
            self.sequence_length,
            self.vocabulary_size,
            self.hidden_size,
            self.num_heads,
            self.intermediate_size,
            self.num_experts,
            self.iterations,
        )
        if any(value <= 0 for value in positive) or self.warmup < 0:
            raise ValueError("benchmark dimensions and iterations must be positive; warmup may be zero")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden-size must divide evenly by num-heads")
        if (self.hidden_size // self.num_heads) % 2:
            raise ValueError("the RoPE head dimension must be even")
        minimum_fraction = 1.0 / self.num_experts
        if self.primary_expert_fraction is not None and not minimum_fraction <= self.primary_expert_fraction <= 1.0:
            raise ValueError(f"primary-expert-fraction must be in [{minimum_fraction:g}, 1]")


def _measure(function: Callable[[], object], warmup: int, iterations: int) -> float:
    """Return mean forward latency in milliseconds after warmup."""
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def _measure_training(function: Callable[[], None], warmup: int, iterations: int) -> float:
    """Return mean forward-plus-backward latency with graph creation per call."""
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def _initialize_cublas(device: torch.device, dtype: torch.dtype) -> None:
    """Initialize CUDA's primary context before any timed backward iteration."""
    matrix = torch.ones((8, 8), device=device, dtype=dtype, requires_grad=True)
    torch.matmul(matrix, matrix).sum().backward()
    torch.cuda.synchronize(device)


def _print_result(name: str, torch_ms: float, triton_ms: float) -> None:
    """Print one stable, copyable comparison row."""
    speedup = torch_ms / triton_ms if triton_ms else float("inf")
    print(f"{name:30} torch={torch_ms:8.3f} ms  triton={triton_ms:8.3f} ms  speedup={speedup:5.2f}x")


def _torch_rmsnorm(inputs: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Reference equation from ``RMSNorm.forward``."""
    float_inputs = inputs.float()
    rms = torch.sqrt(torch.mean(float_inputs.square(), dim=-1, keepdim=True) + epsilon)
    return ((float_inputs / rms) * weight.float()).to(inputs.dtype)


def _torch_rope(
    inputs: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Reference equation from ``RotaryPositionalEmbedding.forward``."""
    cosine, sine = cosine[positions].to(inputs.dtype), sine[positions].to(inputs.dtype)
    even, odd = inputs[..., ::2], inputs[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


def _torch_adamw_update(
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
) -> torch.Tensor:
    """Reference sequence from the explicit ``AdamW.step`` method."""
    first_moment.mul_(beta1).add_(gradient, alpha=1 - beta1)
    second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
    parameter.addcdiv_(first_moment, second_moment.sqrt().add_(epsilon), value=-adjusted_learning_rate)
    parameter.mul_(1 - learning_rate * weight_decay)
    return parameter


def _benchmark_core(config: BenchmarkConfig, device: torch.device) -> None:
    """Run the original one-operation kernel comparisons."""
    batch, sequence = config.batch_size, config.sequence_length
    hidden_size, intermediate_size = config.hidden_size, config.intermediate_size
    heads, head_dimension = config.num_heads, hidden_size // config.num_heads
    vocabulary_size, dtype = config.vocabulary_size, config.dtype
    warmup, iterations = config.warmup, config.iterations
    epsilon = 1e-6

    print("\n[Core kernels]")
    hidden = torch.randn(batch, sequence, hidden_size, device=device, dtype=dtype)
    norm_weight = torch.ones(hidden_size, device=device)
    _print_result(
        "RMSNorm",
        _measure(lambda: _torch_rmsnorm(hidden, norm_weight, epsilon), warmup, iterations),
        _measure(lambda: rms_norm(hidden, norm_weight, epsilon), warmup, iterations),
    )

    attention = torch.randn(batch, heads, sequence, head_dimension, device=device, dtype=dtype)
    positions = torch.arange(sequence, device=device).view(1, 1, -1).expand(batch, heads, -1)
    table_positions = torch.arange(sequence, device=device).unsqueeze(1)
    dimensions = torch.arange(0, head_dimension, 2, device=device).float()
    angles = table_positions * (1_000_000.0 ** (-dimensions / head_dimension))
    cosine, sine = torch.cos(angles), torch.sin(angles)
    _print_result(
        "RoPE",
        _measure(lambda: _torch_rope(attention, cosine, sine, positions), warmup, iterations),
        _measure(lambda: rope(attention, cosine, sine, positions), warmup, iterations),
    )

    gate = torch.randn(batch, sequence, intermediate_size, device=device, dtype=dtype)
    up = torch.randn_like(gate)
    _print_result(
        "SwiGLU gate",
        _measure(lambda: gate * torch.sigmoid(gate) * up, warmup, iterations),
        _measure(lambda: swiglu_gate(gate, up), warmup, iterations),
    )

    softmax_logits = torch.randn(batch * sequence, vocabulary_size, device=device, dtype=dtype)
    _print_result(
        "Shifted Softmax",
        _measure(lambda: torch.softmax(softmax_logits, dim=-1), warmup, iterations),
        _measure(lambda: row_softmax(softmax_logits), warmup, iterations),
    )

    vocabulary_logits = torch.randn(batch, sequence, vocabulary_size, device=device, dtype=dtype)
    targets = torch.randint(vocabulary_size, (batch, sequence), device=device)
    _print_result(
        "Cross Entropy",
        _measure(lambda: torch_cross_entropy(vocabulary_logits, targets), warmup, iterations),
        _measure(lambda: cross_entropy(vocabulary_logits, targets), warmup, iterations),
    )

    adamw_elements = vocabulary_size * hidden_size
    adamw_parameter = torch.randn(adamw_elements, device=device)
    adamw_gradient = torch.randn_like(adamw_parameter)
    adamw_first, adamw_second = torch.zeros_like(adamw_parameter), torch.zeros_like(adamw_parameter)
    torch_parameter = adamw_parameter.clone()
    torch_first, torch_second = torch.zeros_like(torch_parameter), torch.zeros_like(torch_parameter)
    beta1, beta2, learning_rate, weight_decay, optimizer_epsilon = 0.9, 0.95, 3e-4, 0.1, 1e-8
    adjusted_learning_rate = learning_rate * (1 - beta2) ** 0.5 / (1 - beta1)
    _print_result(
        "AdamW update",
        _measure(
            lambda: _torch_adamw_update(
                torch_parameter,
                adamw_gradient,
                torch_first,
                torch_second,
                beta1,
                beta2,
                adjusted_learning_rate,
                optimizer_epsilon,
                learning_rate,
                weight_decay,
            ),
            warmup,
            iterations,
        ),
        _measure(
            lambda: adamw_update_(
                adamw_parameter,
                adamw_gradient,
                adamw_first,
                adamw_second,
                beta1,
                beta2,
                adjusted_learning_rate,
                optimizer_epsilon,
                learning_rate,
                weight_decay,
            ),
            warmup,
            iterations,
        ),
    )
    _print_result(
        "Gradient norm",
        _measure(lambda: torch.sum(adamw_gradient.float().square()), warmup, iterations),
        _measure(lambda: torch.sum(squared_norm_partials(adamw_gradient)), warmup, iterations),
    )

    queries = torch.randn(batch, heads, sequence, head_dimension, device=device, dtype=dtype)
    keys, values = torch.randn_like(queries), torch.randn_like(queries)
    _print_result(
        "FlashAttention",
        _measure(lambda: flash_attention_pytorch(queries, keys, values, True), warmup, iterations),
        _measure(lambda: flash_attention(queries, keys, values, is_causal=True), warmup, iterations),
    )


def _make_routes(config: BenchmarkConfig, device: torch.device) -> torch.Tensor:
    """Create deterministic routes with a configurable load on expert zero."""
    tokens = config.batch_size * config.sequence_length
    primary_fraction = config.primary_expert_fraction or 1.0 / config.num_experts
    primary = int(tokens * primary_fraction)
    if config.num_experts == 1:
        return torch.zeros(tokens, device=device, dtype=torch.int64)
    tail = torch.arange(tokens - primary, device=device) % (config.num_experts - 1) + 1
    return torch.cat((torch.zeros(primary, device=device, dtype=torch.int64), tail))


def _routing_layout(
    routes: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sorted token IDs, offsets, and the compact Variable-M tile schedule."""
    token_indices = torch.arange(routes.numel(), device=routes.device)
    sorted_tokens = token_indices.index_select(0, torch.argsort(routes))
    counts = torch.bincount(routes, minlength=num_experts)
    expert_offsets = torch.cat((counts.new_zeros(1), torch.cumsum(counts, dim=0)))
    tile_experts, tile_rows = build_variable_m_schedule(expert_offsets)
    return sorted_tokens, expert_offsets, tile_experts, tile_rows


def _torch_dispatch(inputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
    """Reference compact token gather."""
    return inputs.index_select(0, token_indices)


def _torch_combine(compact_outputs: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
    """Reference compact token scatter."""
    output = torch.empty_like(compact_outputs)
    output.index_copy_(0, token_indices, compact_outputs)
    return output


def _torch_grouped_linear(
    inputs: torch.Tensor, weight: torch.Tensor, expert_offsets: tuple[int, ...]
) -> torch.Tensor:
    """Evaluate Variable-M weights as explicit per-expert Torch GEMMs."""
    outputs = []
    for expert in range(weight.shape[0]):
        start, end = expert_offsets[expert], expert_offsets[expert + 1]
        outputs.append(inputs[start:end] @ weight[expert].transpose(0, 1))
    return torch.cat(outputs)


def _torch_expert_mlp(
    inputs: torch.Tensor, expert_offsets: tuple[int, ...], experts: VariableGroupedSwiGLUExperts
) -> torch.Tensor:
    """Run the complete compact expert MLP through explicit Torch equations."""
    gate_up = _torch_grouped_linear(inputs, experts.gate_up_proj.weight, expert_offsets)
    gate, up = gate_up.chunk(2, dim=-1)
    hidden = silu(gate) * up
    return _torch_grouped_linear(hidden, experts.down_proj.weight, expert_offsets)


def _torch_top1_execution(
    inputs: torch.Tensor, routes: torch.Tensor, experts: VariableGroupedSwiGLUExperts
) -> torch.Tensor:
    """Reference Top-1 execution without CPU synchronization for route lengths."""
    output = torch.empty_like(inputs)
    for expert in range(experts.gate_up_proj.num_experts):
        token_indices = torch.nonzero(routes == expert, as_tuple=False).flatten()
        expert_inputs = inputs.index_select(0, token_indices)
        gate_up = expert_inputs @ experts.gate_up_proj.weight[expert].transpose(0, 1)
        gate, up = gate_up.chunk(2, dim=-1)
        expert_outputs = (silu(gate) * up) @ experts.down_proj.weight[expert].transpose(0, 1)
        output.index_copy_(0, token_indices, expert_outputs)
    return output


def _benchmark_moe(config: BenchmarkConfig, device: torch.device) -> None:
    """Compare individual MoE components and complete Top-1 execution paths."""
    tokens = config.batch_size * config.sequence_length
    dtype, warmup, iterations = config.dtype, config.warmup, config.iterations
    routes = _make_routes(config, device)
    sorted_tokens, offsets, tile_experts, tile_rows = _routing_layout(routes, config.num_experts)
    counts = offsets[1:] - offsets[:-1]
    # Keep device-to-host boundary resolution outside every timed reference call.
    offset_values = tuple(offsets.tolist())
    inputs = torch.randn(tokens, config.hidden_size, device=device, dtype=dtype)
    compact_inputs = _torch_dispatch(inputs, sorted_tokens)
    experts = VariableGroupedSwiGLUExperts(config.num_experts, config.hidden_size, config.intermediate_size).to(
        device=device, dtype=dtype
    )

    print("\n[MoE components]")
    print(f"expert token counts: {counts.tolist()}")
    _print_result(
        "Token dispatch",
        _measure(lambda: _torch_dispatch(inputs, sorted_tokens), warmup, iterations),
        _measure(lambda: moe_top1_dispatch(inputs, sorted_tokens), warmup, iterations),
    )
    compact_outputs = torch.randn_like(compact_inputs)
    _print_result(
        "Token combine",
        _measure(lambda: _torch_combine(compact_outputs, sorted_tokens), warmup, iterations),
        _measure(lambda: moe_top1_combine(compact_outputs, sorted_tokens), warmup, iterations),
    )
    _print_result(
        "Gate/Up Variable-M GEMM",
        _measure(
            lambda: _torch_grouped_linear(compact_inputs, experts.gate_up_proj.weight, offset_values),
            warmup,
            iterations,
        ),
        _measure(
            lambda: variable_grouped_gemm(
                compact_inputs, experts.gate_up_proj.weight, offsets, tile_experts, tile_rows
            ),
            warmup,
            iterations,
        ),
    )
    intermediate = torch.randn(tokens, config.intermediate_size, device=device, dtype=dtype)
    _print_result(
        "Down Variable-M GEMM",
        _measure(
            lambda: _torch_grouped_linear(intermediate, experts.down_proj.weight, offset_values), warmup, iterations
        ),
        _measure(
            lambda: variable_grouped_gemm(intermediate, experts.down_proj.weight, offsets, tile_experts, tile_rows),
            warmup,
            iterations,
        ),
    )
    _print_result(
        "Complete expert MLP fwd",
        _measure(lambda: _torch_expert_mlp(compact_inputs, offset_values, experts), warmup, iterations),
        _measure(lambda: experts(compact_inputs, offsets, tile_experts, tile_rows), warmup, iterations),
    )

    output_gradient = torch.randn(tokens, config.hidden_size, device=device, dtype=dtype)

    def torch_expert_forward_backward() -> None:
        experts.zero_grad(set_to_none=True)
        training_inputs = compact_inputs.detach().requires_grad_()
        _torch_expert_mlp(training_inputs, offset_values, experts).backward(output_gradient)

    def triton_expert_forward_backward() -> None:
        experts.zero_grad(set_to_none=True)
        training_inputs = compact_inputs.detach().requires_grad_()
        experts(training_inputs, offsets, tile_experts, tile_rows).backward(output_gradient)

    _print_result(
        "Complete expert MLP fwd+bwd",
        _measure_training(torch_expert_forward_backward, warmup, iterations),
        _measure_training(triton_expert_forward_backward, warmup, iterations),
    )

    triton_layer = MoELayer(
        config.hidden_size,
        config.intermediate_size,
        config.num_experts,
        num_experts_per_token=1,
    ).to(device=device, dtype=dtype).eval()
    assert triton_layer.grouped_experts is not None
    _print_result(
        "Top-1 execution fwd",
        _measure(
            lambda: _torch_top1_execution(inputs, routes, triton_layer.grouped_experts), warmup, iterations
        ),
        _measure(lambda: triton_layer._forward_grouped_top1(inputs, routes), warmup, iterations),
    )

    layer_inputs = inputs.view(config.batch_size, config.sequence_length, config.hidden_size)
    with torch.inference_mode():
        natural_routes = torch.argmax(triton_layer.router(inputs), dim=-1)
        natural_counts = torch.bincount(natural_routes, minlength=config.num_experts)
    print(f"router-produced counts: {natural_counts.tolist()}")
    _print_result(
        "Complete MoE layer fwd",
        _measure(
            lambda: _torch_top1_execution(
                inputs, torch.argmax(triton_layer.router(inputs), dim=-1), triton_layer.grouped_experts
            ).view_as(layer_inputs),
            warmup,
            iterations,
        ),
        _measure(lambda: triton_layer(layer_inputs), warmup, iterations),
    )

    layer_gradient = torch.randn_like(layer_inputs)

    def torch_layer_forward_backward() -> None:
        triton_layer.zero_grad(set_to_none=True)
        training_inputs = layer_inputs.detach().requires_grad_()
        flat_inputs = training_inputs.reshape(-1, config.hidden_size)
        routes = torch.argmax(triton_layer.router(flat_inputs), dim=-1)
        _torch_top1_execution(flat_inputs, routes, triton_layer.grouped_experts).view_as(training_inputs).backward(
            layer_gradient
        )

    def triton_layer_forward_backward() -> None:
        triton_layer.zero_grad(set_to_none=True)
        training_inputs = layer_inputs.detach().requires_grad_()
        triton_layer(training_inputs).backward(layer_gradient)

    _print_result(
        "Complete MoE layer fwd+bwd",
        _measure_training(torch_layer_forward_backward, warmup, iterations),
        _measure_training(triton_layer_forward_backward, warmup, iterations),
    )


def run(config: BenchmarkConfig) -> None:
    """Run the selected benchmark sections."""
    config.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    torch.manual_seed(0)
    device = torch.device("cuda")
    _initialize_cublas(device, config.dtype)
    head_dimension = config.hidden_size // config.num_heads
    print(f"device: {torch.cuda.get_device_name(device)}")
    print(
        f"shape: batch={config.batch_size}, sequence={config.sequence_length}, hidden={config.hidden_size}, "
        f"heads={config.num_heads}, head_dim={head_dimension}, intermediate={config.intermediate_size}, "
        f"vocabulary={config.vocabulary_size}, experts={config.num_experts}, dtype={config.dtype}"
    )
    print(f"warmup={config.warmup}, iterations={config.iterations}; CUDA-event operation latency")
    if config.section in {"all", "core"}:
        _benchmark_core(config, device)
    if config.section in {"all", "moe"}:
        _benchmark_moe(config, device)


def main() -> None:
    """Parse benchmark shapes and section selection from the command line."""
    parser = argparse.ArgumentParser(description="Benchmark ajLLM Torch and Triton CUDA paths")
    parser.add_argument("--section", choices=("all", "core", "moe"), default="all")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--vocabulary-size", type=int, default=6400)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--intermediate-size", type=int, default=2432)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument(
        "--primary-expert-fraction",
        type=float,
        default=None,
        help="fraction routed to expert 0; defaults to balanced 1 / num-experts",
    )
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    run(
        BenchmarkConfig(
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            vocabulary_size=args.vocabulary_size,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            intermediate_size=args.intermediate_size,
            num_experts=args.num_experts,
            primary_expert_fraction=args.primary_expert_fraction,
            dtype=_DTYPES[args.dtype],
            warmup=args.warmup,
            iterations=args.iterations,
            section=args.section,
        )
    )


if __name__ == "__main__":
    main()
