"""Controlled, serial CUDA service sweeps of token and sequence budgets."""

import argparse
import copy
import csv
import hashlib
import json
import statistics
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

from tokenizers import Tokenizer

from ajvllm.config import EngineConfig
from ajvllm.workflows.compare import measure, server


def sweep_config(base, tokens, seqs, graph_sizes):
    config = copy.deepcopy(base)
    engine = config["engine"]
    engine.update(max_num_batched_tokens=tokens, max_num_seqs=seqs, enable_chunked_prefill=True)
    engine.pop("max_prefill_chunk_size", None)
    engine.pop("max_prefill_tokens_per_step", None)
    EngineConfig(**engine)
    memory = config.setdefault("memory", {})
    memory.update(backend="paged", enable_prefix_cache=False)
    memory.pop("num_blocks", None)
    # Keep the same capture buckets and memory allowance across all sweep points.
    if config.get("graphs", {}).get("enabled"):
        config["graphs"]["batch_sizes"] = graph_sizes
    return config


def write_config(config, path):
    path.write_text(
        "\n".join(
            f"[{section}]\n" + "\n".join(f"{key} = {json.dumps(value)}" for key, value in values.items()) + "\n"
            for section, values in config.items()
        )
    )


def health(url):
    with urlopen(url + "/health", timeout=10) as response:
        return json.load(response)


def result_row(phase, value, repeat, result, before, after):
    kv = after["kv_cache"]
    row = dict(
        phase=phase,
        value=value,
        repeat=repeat,
        status="ok" if not result["failed"] else "failed",
        concurrency=result["concurrency"],
        completed=result["completed"],
        failed=result["failed"],
        input_tokens_per_s=result["input_tokens_per_s"],
        output_tokens_per_s=result["output_tokens_per_s"],
    )
    for name in ("ttft_s", "tpot_s", "latency_s"):
        for stat in ("mean", "p95"):
            row[f"{name}_{stat}"] = (result["latency"][name] or {}).get(stat)
    row.update(
        gpu_utilization_percent=(result["gpu"]["utilization_percent"] or {}).get("mean"),
        gpu_peak_mib=(result["gpu"]["memory_used_mib"] or {}).get("max"),
        kv_pool_mib=kv["pool_bytes"] / 1024**2,
        kv_peak_mib=kv["peak_used_bytes"] / 1024**2,
        kv_peak_percent=100 * kv["peak_used_bytes"] / kv["pool_bytes"],
    )
    for key in ("preemptions", "admission_waits", "prefix_hits"):
        row[key] = kv[key] - before["kv_cache"][key]
    for key in ("captures", "replays", "fallbacks"):
        row[f"graph_{key}"] = after["graphs"].get(key, 0) - before["graphs"].get(key, 0)
    return row


def save_report(report, output):
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = report["rows"]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    columns = [
        "input_tokens_per_s",
        "output_tokens_per_s",
        "ttft_s_mean",
        "tpot_s_mean",
        "ttft_s_p95",
        "tpot_s_p95",
        "gpu_utilization_percent",
        "gpu_peak_mib",
        "kv_pool_mib",
        "kv_peak_percent",
        "admission_waits",
        "preemptions",
        "graph_fallbacks",
    ]
    lines = [
        "# Budget sweep",
        "",
        "Successful repeats: mean ± run standard deviation; latency in seconds.",
        "KV high-water marks include excluded warmup. GPU memory includes other device users.",
        "Decode measurements include HTTP and short prefills; they are not isolated kernel timings.",
        "",
    ]
    for phase in ("prefill", "decode"):
        lines += [
            f"## {phase.capitalize()}",
            "",
            "| Budget | Runs | " + " | ".join(columns) + " |",
            "|---|---|" + "---|" * len(columns),
        ]
        for value in sorted({row["value"] for row in rows if row["phase"] == phase}):
            group = [row for row in rows if row["phase"] == phase and row["value"] == value]
            good = [row for row in group if row["status"] == "ok"]
            cells = []
            for key in columns:
                samples = [row[key] for row in good if row.get(key) is not None]
                cells.append(f"{statistics.mean(samples):.4f} ± {statistics.pstdev(samples):.4f}" if samples else "—")
            lines.append(f"| {value} | {len(good)}/{len(group)} | " + " | ".join(cells) + " |")
        lines.append("")
    for row in rows:
        if row["status"] != "ok":
            lines.append(
                f"- Failed {row['phase']}={row['value']}: {row.get('error', 'request failures; see report.json')}"
            )
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def plot_report(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = [
        ("input_tokens_per_s", "Input throughput (tokens/s)"),
        ("output_tokens_per_s", "Output throughput (tokens/s)"),
        ("ttft_s_mean", "Mean TTFT (s)"),
        ("tpot_s_mean", "Mean TPOT (s)"),
        ("gpu_utilization_percent", "Mean GPU utilization (%)"),
        ("gpu_peak_mib", "Peak device memory (MiB)"),
        ("kv_pool_mib", "KV pool (MiB)"),
        ("kv_peak_percent", "KV high-water usage (%)"),
    ]
    for phase in ("prefill", "decode"):
        good = [row for row in rows if row["phase"] == phase and row["status"] == "ok"]
        if not good:
            continue
        values = sorted({row["value"] for row in good})
        fig, axes = plt.subplots(2, 4, figsize=(17, 8), layout="constrained")
        for ax, (metric, label) in zip(axes.flat, metrics, strict=True):
            samples = [
                [row[metric] for row in good if row["value"] == value and row[metric] is not None] for value in values
            ]
            ax.errorbar(
                values,
                [np.mean(s) if s else float("nan") for s in samples],
                yerr=[np.std(s) if s else 0 for s in samples],
                marker="o",
                capsize=4,
            )
            ax.set(title=label, xlabel="Token budget" if phase == "prefill" else "Sequence limit", ylim=(0, None))
            upper = [float(np.mean(s) + np.std(s)) for s in samples if s]
            if upper:
                ax.set_ylim(0, max(max(upper), 1e-6) * 1.12)
            ax.set_xscale("log", base=2)
            ax.set_xticks(values, labels=[str(v) for v in values])
            ax.grid(alpha=0.25)
        fig.suptitle(f"{phase.capitalize()} budget sweep — mean ± run standard deviation; KV peak includes warmup")
        fig.savefig(output / f"{phase}.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/engine/benchmark.toml"))
    parser.add_argument("--model", type=Path, default=Path("model/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--dataset", type=Path, default=Path("benchmarks/datasets/long.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/budget"))
    parser.add_argument("--phase", choices=("prefill", "decode", "both"), default="both")
    parser.add_argument("--token-budgets", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--sequence-limits", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--prefill-concurrency", type=int, default=8)
    parser.add_argument("--decode-token-budget", type=int, default=8192)
    parser.add_argument("--prefill-prompt-tokens", type=int, default=4096)
    parser.add_argument("--decode-prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-output-tokens", type=int, default=128)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=8135)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--startup-timeout", type=float, default=300)
    args = parser.parse_args()
    for name in (
        "prefill_concurrency",
        "decode_token_budget",
        "prefill_prompt_tokens",
        "decode_prompt_tokens",
        "requests",
        "repeats",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.decode_output_tokens < 2 or any(v < 1 for v in args.token_budgets + args.sequence_limits):
        parser.error("output tokens must be >= 2 and sweep values positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu-memory-utilization must be in (0, 1]")
    base = tomllib.loads(args.config.read_text())
    graph_max = max(args.sequence_limits + [args.prefill_concurrency])
    graph_sizes = sorted(
        {1 << i for i in range(graph_max.bit_length())} | set(args.sequence_limits) | {args.prefill_concurrency}
    )
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    source = [
        tokenizer.encode(json.loads(line)["prompt"], add_special_tokens=False).ids
        for line in args.dataset.read_text().splitlines()
        if line.strip()
    ]
    cases = []
    if args.phase in ("both", "prefill"):
        cases += [
            ("prefill", budget, args.prefill_concurrency, args.prefill_prompt_tokens, 2)
            for budget in sorted(set(args.token_budgets))
        ]
    if args.phase in ("both", "decode"):
        cases += [
            ("decode", args.decode_token_budget, seqs, args.decode_prompt_tokens, args.decode_output_tokens)
            for seqs in sorted(set(args.sequence_limits))
        ]
    for phase, _, seqs, prompt, generated in cases:
        if prompt + generated > base["engine"]["max_model_len"]:
            parser.error(f"{phase} workload exceeds configured max_model_len")
        if not any(len(tokens) >= prompt for tokens in source):
            parser.error(f"dataset has no prompt with {prompt} tokens; choose a shorter prompt or larger dataset")
        if args.requests < seqs:
            parser.error("requests must be >= every tested concurrency")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        parser.error("output already contains a report; select a new directory")
    report = dict(
        created_at=datetime.now(UTC).isoformat(),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        base_config=base,
        dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        rows=[],
        runs=[],
    )
    for phase, budget, seqs, prompt, generated in cases:
        value = budget if phase == "prefill" else seqs
        label = f"{phase}-{value}"
        config = sweep_config(base, budget, seqs, graph_sizes)
        config_path = output / f"{label}.toml"
        write_config(config, config_path)
        dataset = [tokens[:prompt] for tokens in source if len(tokens) >= prompt]
        run_args = copy.copy(args)
        run_args.max_tokens = generated
        for repeat in range(args.repeats):
            print(f"{label}, repeat {repeat + 1}/{args.repeats}: starting", flush=True)
            command = [
                sys.executable,
                "-m",
                "ajvllm.workflows.serve",
                "--model",
                str(args.model),
                "--config",
                str(config_path),
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--max-pending-requests",
                str(max(128, seqs * 2)),
            ]
            try:
                with server(command, output / f"{label}-r{repeat}.log", args) as url:
                    startup = health(url)
                    warmup = measure("ajvllm", url, dataset, seqs, run_args, warmup=True)
                    if warmup["failed"]:
                        raise RuntimeError(f"warmup failed: {warmup['requests']}")
                    before = health(url)
                    result = measure("ajvllm", url, dataset, seqs, run_args)
                    after = health(url)
                row = result_row(phase, value, repeat, result, before, after)
                report["runs"].append(
                    dict(
                        phase=phase,
                        value=value,
                        repeat=repeat,
                        startup=startup,
                        before=before,
                        after=after,
                        result=result,
                    )
                )
            except (RuntimeError, TimeoutError, OSError) as exc:
                row = dict(phase=phase, value=value, repeat=repeat, status="failed", error=str(exc))
            report["rows"].append(row)
            save_report(report, output)
            print(json.dumps(row), flush=True)
            if row["status"] == "failed":
                break
    plot_report(report["rows"], output)
    print(f"Report, CSV, candidate configurations and plots: {output}")


if __name__ == "__main__":
    main()
