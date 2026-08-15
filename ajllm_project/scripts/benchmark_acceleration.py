#!/usr/bin/env python3
"""Benchmark acceleration techniques: Baseline, FlashAttention2, FSDP+AC, and combinations.

Measures forward/backward/optimizer time, peak memory, and throughput for each configuration.

Examples:
    # Quick local test (single GPU)
    python scripts/benchmark_acceleration.py --device cuda --steps 3 --warmup 1

    # Full benchmark on 2 GPUs
    python scripts/benchmark_acceleration.py --device cuda --world-size 2 --steps 10

    # Only test specific configurations
    python scripts/benchmark_acceleration.py --configs baseline flash_attention
"""

import argparse
import csv
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

# Import model components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ajllm.modeling.transformer import TransformerLM
from ajllm.training.losses import cross_entropy


CONFIGS = ("baseline", "flash_attention", "fsdp_ac", "flash_attention_fsdp_ac")
PHASES = ("forward", "backward", "optimizer", "total")


@dataclass
class PhaseStats:
    """Statistics for one training phase."""
    mean_ms: float
    std_ms: float
    peak_memory_mb: float


@dataclass
class BenchmarkResult:
    config: str
    device: str
    world_size: int
    rank: int
    vocab_size: int
    d_model: int
    num_layers: int
    num_heads: int
    batch_size: int
    context_length: int
    steps: int
    warmup_steps: int
    forward: PhaseStats
    backward: PhaseStats
    optimizer: PhaseStats
    total: PhaseStats
    parameter_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--world-size", type=int, default=1, help="Number of GPUs for FSDP")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--master-port", type=str, default="29501")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def _use_cuda(device_arg: str) -> bool:
    return device_arg == "cuda"


def _rank_device(args: argparse.Namespace, rank: int) -> torch.device:
    if not _use_cuda(args.device):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device_index = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_index)
    return torch.device("cuda", device_index)


def _init_process_group(args: argparse.Namespace, rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = args.master_port
    backend = "nccl" if _use_cuda(args.device) else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def _cleanup_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _sync(device: torch.device) -> None:
    """Synchronize device and process group."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif dist.is_available() and dist.is_initialized():
        dist.barrier()


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: torch.device) -> float:
    """Return peak allocated memory in MiB."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024
    return 0.0


def _make_batch(args: argparse.Namespace, device: torch.device, world_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate batch with size adjusted for world size."""
    torch.manual_seed(1234)
    local_batch_size = args.batch_size // world_size
    tokens = torch.randint(0, args.vocab_size, (local_batch_size, args.context_length), device=device)
    return tokens, tokens


def _make_model(args: argparse.Namespace, device: torch.device, config: str) -> nn.Module:
    """Build model based on config."""
    torch.manual_seed(42)

    use_flash = "flash_attention" in config
    use_fsdp = "fsdp" in config

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_model * 4,
        use_flash_attention=use_flash,
    )

    # Move model to device before FSDP wrapping (FSDP broadcasts tensors during init)
    model = model.to(device)

    if use_fsdp:
        from ajllm.training.distributed import FullyShardedDataParallel
        compute_dtype = torch.float16 if device.type == "cuda" else None
        model = FullyShardedDataParallel(model, compute_dtype=compute_dtype)

    return model


def _stats(samples: list[float], peaks: list[float]) -> PhaseStats:
    return PhaseStats(
        mean_ms=statistics.mean(samples),
        std_ms=statistics.stdev(samples) if len(samples) > 1 else 0.0,
        peak_memory_mb=max(peaks) if peaks else 0.0,
    )


def _benchmark_worker(args: argparse.Namespace, config: str, rank: int, world_size: int) -> BenchmarkResult:
    """Run benchmark for one configuration."""
    distributed = world_size > 1
    device = _rank_device(args, rank)

    if distributed:
        _init_process_group(args, rank, world_size)

    try:
        model = _make_model(args, device, config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        tokens, targets = _make_batch(args, device, world_size)

        parameter_count = sum(p.numel() for p in model.parameters())

        # Collect timing and memory stats
        timings: dict[str, list[float]] = {p: [] for p in PHASES}
        peaks: dict[str, list[float]] = {p: [] for p in PHASES}

        for step in range(args.warmup + args.steps):
            optimizer.zero_grad(set_to_none=True)
            collect = step >= args.warmup

            # Total step timing
            _sync(device)
            _reset_peak_memory(device)
            total_start = time.perf_counter()

            # Forward
            _reset_peak_memory(device)
            t0 = time.perf_counter()
            logits = model(tokens)
            loss = cross_entropy(logits, targets)
            _sync(device)
            fwd_ms = (time.perf_counter() - t0) * 1000
            fwd_peak = _peak_memory_mb(device)

            # Backward
            _reset_peak_memory(device)
            t0 = time.perf_counter()
            loss.backward()
            if hasattr(model, 'finish_gradient_synchronization'):
                model.finish_gradient_synchronization()
            _sync(device)
            bwd_ms = (time.perf_counter() - t0) * 1000
            bwd_peak = _peak_memory_mb(device)

            # Optimizer
            _reset_peak_memory(device)
            t0 = time.perf_counter()
            optimizer.step()
            _sync(device)
            opt_ms = (time.perf_counter() - t0) * 1000
            opt_peak = _peak_memory_mb(device)

            # Total
            total_ms = (time.perf_counter() - total_start) * 1000
            total_peak = max(fwd_peak, bwd_peak, opt_peak)

            if collect:
                timings["forward"].append(fwd_ms)
                timings["backward"].append(bwd_ms)
                timings["optimizer"].append(opt_ms)
                timings["total"].append(total_ms)
                peaks["forward"].append(fwd_peak)
                peaks["backward"].append(bwd_peak)
                peaks["optimizer"].append(opt_peak)
                peaks["total"].append(total_peak)

        return BenchmarkResult(
            config=config,
            device=str(device),
            world_size=world_size,
            rank=rank,
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            batch_size=args.batch_size,
            context_length=args.context_length,
            steps=args.steps,
            warmup_steps=args.warmup,
            forward=_stats(timings["forward"], peaks["forward"]),
            backward=_stats(timings["backward"], peaks["backward"]),
            optimizer=_stats(timings["optimizer"], peaks["optimizer"]),
            total=_stats(timings["total"], peaks["total"]),
            parameter_count=parameter_count,
        )
    finally:
        _cleanup_process_group()


def _run_rank_for_spawn(rank: int, args: argparse.Namespace, config: str, q: mp.SimpleQueue) -> None:
    try:
        result = _benchmark_worker(args, config, rank, args.world_size)
        q.put(asdict(result))
    except Exception as e:
        import traceback
        print(f"Rank {rank} error: {e}", flush=True)
        traceback.print_exc()
        raise


def _spawn_config(args: argparse.Namespace, config: str) -> list[BenchmarkResult]:
    """Run config across multiple GPUs."""
    # FSDP requires distributed
    needs_distributed = "fsdp" in config
    world_size = args.world_size if needs_distributed else 1

    if world_size == 1:
        return [_benchmark_worker(args, config, rank=0, world_size=1)]

    queue: mp.SimpleQueue = mp.get_context("spawn").SimpleQueue()
    mp.spawn(_run_rank_for_spawn, args=(args, config, queue), nprocs=world_size, join=True)
    results = [BenchmarkResult(**_restore_result(queue.get())) for _ in range(world_size)]
    return sorted(results, key=lambda r: r.rank)


def _restore_result(data: dict) -> dict:
    """Restore PhaseStats objects."""
    for phase in PHASES:
        data[phase] = PhaseStats(**data[phase])
    return data


def _write_outputs(args: argparse.Namespace, results: list[BenchmarkResult]) -> None:
    """Write results to JSON, CSV, and plots."""
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = [asdict(result) for result in results]

    # JSON output
    json_path = args.output_dir / "benchmark_acceleration.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults written to {json_path}")

    # CSV output
    csv_path = args.output_dir / "benchmark_acceleration.csv"
    fieldnames = [
        "config", "rank", "device", "world_size", "parameter_count", "phase",
        "mean_ms", "std_ms", "peak_memory_mb",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for phase in PHASES:
                stat: PhaseStats = getattr(result, phase)
                writer.writerow({
                    "config": result.config,
                    "rank": result.rank,
                    "device": result.device,
                    "world_size": result.world_size,
                    "parameter_count": result.parameter_count,
                    "phase": phase,
                    "mean_ms": stat.mean_ms,
                    "std_ms": stat.std_ms,
                    "peak_memory_mb": stat.peak_memory_mb,
                })
    print(f"CSV written to {csv_path}")

    # Generate plots
    _generate_plots(args, results)

    # Print summary in CONFIGS order
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Get configs that actually ran, preserving CONFIGS order
    seen_configs = [c for c in CONFIGS if any(r.config == c for r in results)]

    for config in seen_configs:
        config_results = [r for r in results if r.config == config and r.rank == 0]
        if not config_results:
            continue
        r = config_results[0]
        print(f"\n{config.upper().replace('_', ' ')}:")
        print(f"  Forward:   {r.forward.mean_ms:7.2f} ms  |  Peak Memory: {r.forward.peak_memory_mb:8.2f} MB")
        print(f"  Backward:  {r.backward.mean_ms:7.2f} ms  |  Peak Memory: {r.backward.peak_memory_mb:8.2f} MB")
        print(f"  Optimizer: {r.optimizer.mean_ms:7.2f} ms  |  Peak Memory: {r.optimizer.peak_memory_mb:8.2f} MB")
        print(f"  Total:     {r.total.mean_ms:7.2f} ms  |  Peak Memory: {r.total.peak_memory_mb:8.2f} MB")


def _generate_plots(args: argparse.Namespace, results: list[BenchmarkResult]) -> None:
    """Generate comparison plots for time and memory in reference project style.

    Single figure with 2×4 line-plot grid showing every rank.
    - rows: time (top) / peak memory (bottom)
    - cols: forward / backward / optimizer / total
    - x-axis: configs in CONFIGS order
    - Each rank is one line + marker series
    """
    if not results:
        return

    # Build sorted config list preserving CONFIGS order
    seen_configs: list[str] = []
    for config in CONFIGS:
        if any(r.config == config for r in results):
            seen_configs.append(config)

    config_x = {c: i for i, c in enumerate(seen_configs)}
    x_ticks = list(range(len(seen_configs)))

    # Collect all ranks present
    all_ranks = sorted({r.rank for r in results})

    # Config display labels
    config_labels = {
        "baseline": "baseline",
        "flash_attention": "flash_attention",
        "fsdp_ac": "fsdp_ac",
        "flash_attention_fsdp_ac": "flash_attention_fsdp_ac",
    }

    # Rank styles: one color + marker per rank
    rank_styles: list[dict] = [
        {"color": "#3b82f6", "marker": "o"},   # rank 0 – blue / circle
        {"color": "#f59e0b", "marker": "^"},   # rank 1 – amber / triangle-up
        {"color": "#10b981", "marker": "s"},   # rank 2 – green / square
        {"color": "#ef4444", "marker": "D"},   # rank 3 – red / diamond
        {"color": "#8b5cf6", "marker": "P"},   # rank 4 – purple / plus
        {"color": "#ec4899", "marker": "*"},   # rank 5 – pink / star
        {"color": "#14b8a6", "marker": "X"},   # rank 6 – teal / x-filled
        {"color": "#f97316", "marker": "v"},   # rank 7 – orange / triangle-down
    ]

    def rank_style(rank: int) -> dict:
        return rank_styles[rank % len(rank_styles)]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), constrained_layout=True)
    fig.suptitle(
        f"Acceleration Benchmark — all ranks | {args.batch_size}×{args.context_length} tokens, "
        f"{args.d_model}d/{args.num_layers}L/{args.num_heads}H, {args.steps} steps",
        fontsize=13
    )

    legend_handles: list = []
    legend_labels: list[str] = []

    for col, phase in enumerate(PHASES):
        ax_t = axes[0, col]  # time subplot
        ax_m = axes[1, col]  # memory subplot

        ax_t.set_title(f"{phase} — time (ms)")
        ax_t.set_ylabel("ms")
        ax_m.set_title(f"{phase} — peak memory (MiB)")
        ax_m.set_ylabel("MiB")

        for ax in (ax_t, ax_m):
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([config_labels.get(c, c) for c in seen_configs], rotation=20, ha="right")
            ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

        for rank in all_ranks:
            # Gather (x, time, mem) for this rank, in config order
            rank_results = {r.config: r for r in results if r.rank == rank}
            xs, times, mems = [], [], []
            for config in seen_configs:
                if config in rank_results:
                    stat: PhaseStats = getattr(rank_results[config], phase)
                    xs.append(config_x[config])
                    times.append(stat.mean_ms)
                    mems.append(stat.peak_memory_mb)

            if not xs:
                continue

            style = rank_style(rank)
            color, marker = style["color"], style["marker"]
            ms = 7  # marker size

            # Time line
            (line,) = ax_t.plot(
                xs, times, color=color, marker=marker,
                markersize=ms, linewidth=1.5, label=f"rank {rank}",
            )
            for xi, val in zip(xs, times):
                ax_t.annotate(
                    f"{val:.1f}",
                    (xi, val),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", va="bottom", fontsize=6.5, color=color,
                )

            # Memory line
            ax_m.plot(
                xs, mems, color=color, marker=marker,
                markersize=ms, linewidth=1.5, label=f"rank {rank}",
            )
            for xi, val in zip(xs, mems):
                ax_m.annotate(
                    f"{val:.1f}",
                    (xi, val),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", va="bottom", fontsize=6.5, color=color,
                )

            # Collect legend entry once (from col 0 only)
            if col == 0:
                legend_handles.append(line)
                legend_labels.append(f"rank {rank}")

    # Single legend below the figure
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="lower center", ncol=len(all_ranks),
            bbox_to_anchor=(0.5, -0.04), fontsize=9,
        )

    # Save plot
    plot_path = args.output_dir / "benchmark_acceleration.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    print(f"Benchmark plot saved to {plot_path}")
    plt.close()


def _plot_speedup_comparison(args: argparse.Namespace, config_results: dict[str, BenchmarkResult],
                            configs: list[str], method_labels: dict[str, str],
                            method_colors: dict[str, str]) -> None:
    """Generate speedup and memory reduction comparison plot."""
    baseline_result = config_results["baseline"]
    baseline_time = baseline_result.total.mean_ms
    baseline_memory = baseline_result.total.peak_memory_mb

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Speedup and Memory Reduction vs Baseline", fontsize=12, fontweight="bold")

    plot_configs = [c for c in configs if c != "baseline"]
    xs = range(len(plot_configs))

    # Collect data
    speedups = []
    memory_reductions = []
    colors = []

    for config in plot_configs:
        result = config_results[config]
        speedup = baseline_time / result.total.mean_ms
        memory_reduction = (1 - result.total.peak_memory_mb / baseline_memory) * 100

        speedups.append(speedup)
        memory_reductions.append(memory_reduction)
        colors.append(method_colors.get(config, "#888888"))

    # Speedup plot
    bars1 = ax1.bar(xs, speedups, color=colors, alpha=0.8, width=0.6)
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label="Baseline (1.0×)")
    ax1.set_ylabel("Speedup (×)", fontweight="bold")
    ax1.set_title("Training Speed Improvement")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([method_labels.get(c, c) for c in plot_configs], fontsize=9)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    ax1.set_ylim(bottom=0)

    # Add value labels
    for bar, val in zip(bars1, speedups):
        height = bar.get_height()
        ax1.annotate(
            f"{val:.2f}×",
            (bar.get_x() + bar.get_width() / 2., height),
            textcoords="offset points", xytext=(0, 3),
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    # Memory reduction plot
    bars2 = ax2.bar(xs, memory_reductions, color=colors, alpha=0.8, width=0.6)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_ylabel("Memory Reduction (%)", fontweight="bold")
    ax2.set_title("Peak Memory Savings")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([method_labels.get(c, c) for c in plot_configs], fontsize=9)
    ax2.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars2, memory_reductions):
        height = bar.get_height()
        ax2.annotate(
            f"{val:.1f}%",
            (bar.get_x() + bar.get_width() / 2., height),
            textcoords="offset points", xytext=(0, 3 if val > 0 else -12),
            ha="center", va="bottom" if val > 0 else "top",
            fontsize=9, fontweight="bold"
        )

    plt.tight_layout()

    # Save plot
    plot_path = args.output_dir / "benchmark_speedup.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Speedup comparison plot saved to {plot_path}")
    plt.close()


def main():
    args = parse_args()

    if _use_cuda(args.device) and not torch.cuda.is_available():
        print("Error: CUDA requested but not available", file=sys.stderr)
        sys.exit(1)

    if _use_cuda(args.device) and args.world_size > torch.cuda.device_count():
        print(f"Error: Requested {args.world_size} GPUs but only {torch.cuda.device_count()} available", file=sys.stderr)
        sys.exit(1)

    results: list[BenchmarkResult] = []
    for config in args.configs:
        print(f"\nBenchmarking: {config}")
        results.extend(_spawn_config(args, config))

    _write_outputs(args, results)


if __name__ == "__main__":
    main()
