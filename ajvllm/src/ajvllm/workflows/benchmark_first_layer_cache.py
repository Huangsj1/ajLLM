"""CUDA microbenchmark of embedding -> first RMSNorm/QKV versus resident lookup.

No engine, KV pool, attention or full model is constructed. Large token counts are
stress tests of this local boundary, not claims about scheduler forward sizes.
"""

import argparse
import gc
import json
import statistics
import time
import tomllib
from functools import partial
from pathlib import Path

import torch

from ajvllm.modeling.qwen2.first_layer_cache import FirstLayerFrontEnd, build_cache, load_table


def _event_time(call, iterations):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def measure(call, *, graph, trials):
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    captured = None
    unroll = 64 if graph else 1
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            for _ in range(unroll):
                outputs = call()
        run = captured.replay
    else:
        run = call
    estimate = _event_time(run, 3)
    iterations = min(200, max(3, int(30 / max(estimate, 0.001))))
    times = [_event_time(run, iterations) / unroll for _ in range(trials)]
    if captured is not None:
        del captured, outputs
    return dict(ms=statistics.median(times), min_ms=min(times), max_ms=max(times), iterations=iterations)


def write_report(report, directory):
    import csv

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = report["rows"]
    if rows:
        with (directory / "summary.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        lines = [
            "# First-layer cache benchmark", "",
            "Both paths include embedding. GPU-resident BF16 tables; no full model or engine.", "",
            "| Model | Pattern | Packed tokens | Mode | Compute (ms) | Lookup (ms) | Speedup |",
            "|---|---|---:|---|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {Path(row['model']).name} | {row['pattern']} | {row['tokens']} | {row['timing']} | "
                f"{row['compute_ms']:.6f} | {row['lookup_ms']:.6f} | {row['speedup']:.2f}x |"
            )
        (directory / "summary.md").write_text("\n".join(lines) + "\n")


def plot(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(r["model"] for r in report["rows"]))
    fig, axes = plt.subplots(len(models), 2, figsize=(13, 4.5 * len(models)), squeeze=False, layout="constrained")
    for index, model in enumerate(models):
        for pattern in report["config"]["patterns"]:
            rows = [r for r in report["rows"] if r["model"] == model and r["pattern"] == pattern]
            x = [r["tokens"] for r in rows]
            if not rows:
                continue
            axes[index, 0].loglog(x, [r["compute_ms"] for r in rows], ".-", label=f"RMSNorm + QKV ({pattern})")
            axes[index, 0].loglog(x, [r["lookup_ms"] for r in rows], ".--", label=f"QKV lookup ({pattern})")
            axes[index, 1].semilogx(x, [r["speedup"] for r in rows], ".-", label=pattern)
        axes[index, 0].set(title=f"{Path(model).name}: both paths include embedding", ylabel="CUDA time (ms)")
        axes[index, 1].set(title="Compute time / lookup time", ylabel="Speedup (x)")
        axes[index, 1].axhline(1, color="gray", linestyle=":")
        for ax in axes[index]:
            ax.set_xlabel("Packed input tokens")
            ax.grid(alpha=0.25)
            ax.legend()
    fig.suptitle(
        f"First-layer boundary only; BF16; graph replay up to {report['config']['graph_max_tokens']} tokens\n"
        "GPU-resident tables; steady-state timing; no disk/PCIe transfer in forward"
    )
    fig.savefig(directory / "comparison.png", dpi=160)
    plt.close(fig)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["model/Qwen2.5-0.5B-Instruct", "model/Qwen2.5-1.5B-Instruct"])
    parser.add_argument("--config", type=Path, default=Path("configs/engine/benchmark.toml"))
    parser.add_argument("--directory", default="cache/first_layer_qkv")
    parser.add_argument("--tokens", type=int, nargs="+", help="Default: powers of two through seqs * model length")
    parser.add_argument("--patterns", nargs="+", choices=["uniform", "repeated"], default=["uniform", "repeated"])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--graph-max-tokens", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bandwidth-gbps", type=float, default=912.0, help="Theoretical 3080 Ti DRAM bandwidth")
    parser.add_argument(
        "--dense-tflops", type=float, default=68.2,
        help="3080 Ti dense BF16 Tensor Core peak with FP32 accumulation",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/first-layer-cache"))
    args = parser.parse_args()
    if args.trials < 1 or args.graph_max_tokens < 0 or min(args.bandwidth_gbps, args.dense_tflops) <= 0:
        parser.error("trials and hardware rates must be positive; graph-max-tokens must be nonnegative")
    if (args.output / "report.json").exists():
        parser.error("choose a new output directory")
    engine = tomllib.loads(args.config.read_text())["engine"]
    maximum = engine["max_num_seqs"] * engine["max_model_len"]
    sizes = sorted(set(args.tokens or [2**i for i in range(maximum.bit_length())] + [maximum]))
    if sizes[0] < 1:
        parser.error("token counts must be positive")
    torch.cuda.set_device(args.device)
    torch.manual_seed(0)
    report = dict(
        config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, scheduler=engine, graph_unroll=64,
        rows=[], models=[], skipped=[],
    )
    for path in args.models:
        module, metadata = FirstLayerFrontEnd.from_directory(path, device=args.device)
        start = time.perf_counter()
        artifact = build_cache(module, metadata, args.directory)
        build_or_reuse_s = time.perf_counter() - start
        module.table = load_table(artifact, metadata, device=args.device, dtype=torch.bfloat16)
        h, w, vocab = module.config.hidden_size, module.qkv.out_features, module.config.vocab_size
        # Validate independent packed shapes, including the GEMV-like single-token case.
        errors = []
        for count in [1, 17, 257]:
            ids = torch.randint(vocab, (count,), device=args.device)
            expected, actual = module(ids)[1].float(), module.lookup(ids)[1].float()
            diff = actual - expected
            relative = (diff.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8)).item()
            if not torch.isfinite(actual).all() or relative > 0.01:
                raise RuntimeError(f"cache validation failed for {path}, tokens={count}, relative RMS={relative}")
            errors.append(dict(tokens=count, max_abs=diff.abs().max().item(), relative_rms=relative))
            del expected, actual, diff, ids
        report["models"].append(dict(
            path=path, hidden_size=h, qkv_width=w, vocab_size=vocab, manifest=metadata,
            artifact=str(artifact), cache_bytes=module.table.numel() * 2,
            build_or_reuse_s=build_or_reuse_s, validation=errors,
        ))
        for pattern in args.patterns:
            for count in sizes:
                gc.collect()
                torch.cuda.empty_cache()
                free, _ = torch.cuda.mem_get_info()
                # Residual + normalized rows + QKV output, plus workspace/safety.
                needed = count * (2 * h + w) * 2 + 512 * 1024**2
                if needed > free * 0.9:
                    report["skipped"].append(dict(model=path, pattern=pattern, tokens=count, reason="memory estimate"))
                    write_report(report, args.output)
                    continue
                ids = torch.randint(vocab, (count,), device=args.device)
                if pattern == "repeated":
                    dictionary = torch.randint(vocab, (128,), device=args.device)
                    ids = dictionary[torch.arange(count, device=args.device) % 128]
                use_graph = count <= args.graph_max_tokens
                torch.cuda.reset_peak_memory_stats()
                functions = [partial(module, ids), partial(module.lookup, ids)]
                # Alternate path order to reduce systematic clock/order bias.
                order = [0, 1] if sizes.index(count) % 2 == 0 else [1, 0]
                measured = {}
                try:
                    for which in order:
                        measured[which] = measure(functions[which], graph=use_graph, trials=args.trials)
                except torch.cuda.OutOfMemoryError:
                    report["skipped"].append(dict(model=path, pattern=pattern, tokens=count, reason="CUDA OOM"))
                    del functions, ids
                    torch.cuda.empty_cache()
                    write_report(report, args.output)
                    break
                compute, lookup = measured[0], measured[1]
                report["rows"].append(dict(
                    model=path, hidden_size=h, qkv_width=w, tokens=count, pattern=pattern,
                    timing="graph" if use_graph else "direct", compute_ms=compute["ms"], lookup_ms=lookup["ms"],
                    speedup=compute["ms"] / lookup["ms"],
                    compute_min_ms=compute["min_ms"], compute_max_ms=compute["max_ms"],
                    lookup_min_ms=lookup["min_ms"], lookup_max_ms=lookup["max_ms"],
                    qkv_read_lower_bound_ms=count * w * 2 / (args.bandwidth_gbps * 1e6),
                    lookup_read_write_lower_bound_ms=2 * count * (h + w) * 2 / (args.bandwidth_gbps * 1e6),
                    gemm_lower_bound_ms=2 * count * h * w / (args.dense_tflops * 1e9),
                    peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                ))
                print(f"{Path(path).name} {pattern} N={count}: compute {compute['ms']:.4f} ms, "
                      f"lookup {lookup['ms']:.4f} ms, {compute['ms'] / lookup['ms']:.2f}x", flush=True)
                del functions, ids
                write_report(report, args.output)
        del module
        gc.collect()
        torch.cuda.empty_cache()
    if report["rows"]:
        plot(report, args.output)
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
