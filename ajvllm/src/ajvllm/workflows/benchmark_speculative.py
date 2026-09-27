"""Compare one target checkpoint with and without a draft model on the same GPU."""

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import statistics
import sys
import tomllib
from pathlib import Path

from tokenizers import Tokenizer

from ajvllm.workflows.benchmark_budget_config import health, write_config
from ajvllm.workflows.compare import measure, server


def summarize(report, output):
    rows = []
    for k in report["draft_tokens"]:
        for concurrency in report["concurrency"]:
            runs = [
                r
                for r in report["runs"]
                if r["draft_tokens"] == k and r["concurrency"] == concurrency and not r["result"]["failed"]
            ]
            if not runs:
                continue
            row = dict(draft_tokens=k, concurrency=concurrency, repeats=len(runs))
            for metric in ("ttft_s", "tpot_s", "latency_s"):
                row[metric] = statistics.mean(r["result"]["latency"][metric]["mean"] for r in runs)
            row["tokens_per_s"] = statistics.mean(r["result"]["output_tokens_per_s"] for r in runs)
            row["tokens_per_s_std"] = statistics.pstdev(r["result"]["output_tokens_per_s"] for r in runs)
            row["tpot_s_std"] = statistics.pstdev(r["result"]["latency"]["tpot_s"]["mean"] for r in runs)
            row["gpu_peak_mib"] = max((r["result"]["gpu"]["memory_used_mib"] or {}).get("max", 0) for r in runs)
            row["gpu_utilization_percent"] = statistics.mean(
                (r["result"]["gpu"]["utilization_percent"] or {}).get("mean", float("nan")) for r in runs
            )
            row["acceptance_rate"] = statistics.mean(r["acceptance_rate"] for r in runs)
            row["tokens_per_round"] = statistics.mean(r["tokens_per_round"] for r in runs)
            row["kv_pool_mib"] = statistics.mean(r["after"]["memory"]["pool_bytes"] / 1024**2 for r in runs)
            rows.append(row)
    report["summary"] = rows
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not rows:
        return
    with (output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Speculative decoding comparison",
        "",
        "| Draft tokens | Concurrency | Tokens/s | TTFT (ms) | TPOT (ms) | Accepted (%) | Tokens/round |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['draft_tokens']} | {r['concurrency']} | {r['tokens_per_s']:.2f} | "
            f"{r['ttft_s'] * 1000:.2f} | {r['tpot_s'] * 1000:.2f} | "
            f"{r['acceptance_rate'] * 100:.1f} | {r['tokens_per_round']:.2f} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def plot(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("tokens_per_s", "Output throughput (tokens/s)", 1),
        ("ttft_s", "Mean TTFT (ms)", 1000),
        ("tpot_s", "Mean TPOT (ms)", 1000),
        ("gpu_peak_mib", "Peak device memory (MiB)", 1),
        ("gpu_utilization_percent", "Mean GPU utilization (%)", 1),
        ("acceptance_rate", "Accepted draft tokens (%)", 100),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for ax, (metric, title, scale) in zip(axes.flat, panels, strict=True):
        for k in report["draft_tokens"]:
            rows = [r for r in report["summary"] if r["draft_tokens"] == k]
            if rows:
                ax.plot(
                    [r["concurrency"] for r in rows],
                    [r[metric] * scale for r in rows],
                    marker="o",
                    label="Target only" if not k else f"Draft K={k}",
                )
        ax.set(title=title, xlabel="Concurrency", ylim=(0, None), xticks=report["concurrency"])
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Same target and workload; stochastic sampling; prefix cache disabled")
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/engine/benchmark.toml"))
    parser.add_argument("--model", type=Path, default=Path("model/Qwen2.5-1.5B-Instruct"))
    parser.add_argument("--draft-model", type=Path, default=Path("model/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--dataset", type=Path, default=Path("benchmarks/datasets/long.jsonl"))
    parser.add_argument("--draft-tokens", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--prompt-tokens", type=int, default=256, help="0 keeps each complete prompt")
    parser.add_argument(
        "--chat", action="store_true", help="Apply the target chat template; requires --prompt-tokens 0"
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--profile-steps", action="store_true", help="Diagnostic synchronized timing, not throughput comparison"
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=8137)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/speculative"))
    args = parser.parse_args()
    if (
        min(*args.concurrency, args.requests, args.repeats) < 1
        or args.prompt_tokens < 0
        or min(args.draft_tokens) < 0
        or args.max_tokens < 2
        or args.requests < max(args.concurrency)
    ):
        parser.error("invalid request, draft, concurrency, or output counts")
    base = tomllib.loads(args.config.read_text())
    if args.prompt_tokens + args.max_tokens > base["engine"]["max_model_len"]:
        parser.error("workload exceeds configured context")
    if base.get("memory", {}).get("num_blocks") is not None:
        parser.error("omit num_blocks: this workflow compares automatic pools at the same memory utilization")
    if (args.output / "report.json").exists():
        parser.error("choose a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    source = [json.loads(line)["prompt"] for line in args.dataset.read_text().splitlines() if line.strip()]
    if args.chat:
        if args.prompt_tokens:
            parser.error("chat workloads require --prompt-tokens 0 to preserve the complete chat template")
        from ajvllm.tokenization.qwen2 import Qwen2Tokenizer

        tokenizer = Qwen2Tokenizer(args.model)
        encoded = [tokenizer.encode_chat([{"role": "user", "content": text}]) for text in source]
    else:
        tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
        encoded = [tokenizer.encode(text, add_special_tokens=False).ids for text in source]
    dataset = [
        ids[: args.prompt_tokens] if args.prompt_tokens else ids for ids in encoded if len(ids) >= args.prompt_tokens
    ]
    if not dataset or max(map(len, dataset)) + args.max_tokens > base["engine"]["max_model_len"]:
        parser.error("dataset is empty, too short, or exceeds the context limit")
    ks = sorted({0, *args.draft_tokens})
    report = dict(
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        draft_tokens=ks,
        concurrency=sorted(set(args.concurrency)),
        base_config=base,
        versions={name: importlib.metadata.version(name) for name in ("torch", "triton")},
        dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        token_inputs_sha256=hashlib.sha256(json.dumps(dataset).encode()).hexdigest(),
        runs=[],
        failures=[],
    )
    (args.output / "inputs.json").write_text(json.dumps(dataset))
    for repeat in range(args.repeats):
        for k in ks if repeat % 2 == 0 else list(reversed(ks)):
            settings = copy.deepcopy(base)
            settings.setdefault("memory", {})["enable_prefix_cache"] = False
            settings["speculative"] = dict(enabled=bool(k), draft_model=str(args.draft_model), num_draft_tokens=k or 1)
            config_path = args.output / f"draft-{k}.toml"
            write_config(settings, config_path)
            command = [
                sys.executable,
                "-m",
                "ajvllm.workflows.serve",
                "--config",
                str(config_path),
                "--model",
                str(args.model),
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
            ]
            if args.profile_steps:
                command.append("--profile-steps")
            print(f"Starting K={k}, repeat {repeat + 1}", flush=True)
            try:
                with server(command, args.output / f"draft-{k}-r{repeat}.log", args) as url:
                    for concurrency in report["concurrency"]:
                        warm = measure("ajvllm", url, dataset, concurrency, args, warmup=True)
                        if warm["failed"]:
                            raise RuntimeError(f"warmup failed: {warm['requests']}")
                        before = health(url)
                        result = measure("ajvllm", url, dataset, concurrency, args)
                        after = health(url)
                        delta = {
                            key: after["speculative"].get(key, 0) - before["speculative"].get(key, 0)
                            for key in ("proposed_tokens", "accepted_tokens", "emitted_tokens", "rounds")
                        }
                        report["runs"].append(
                            dict(
                                draft_tokens=k,
                                concurrency=concurrency,
                                repeat=repeat,
                                result=result,
                                before=before,
                                after=after,
                                acceptance_rate=delta["accepted_tokens"] / max(1, delta["proposed_tokens"]),
                                tokens_per_round=delta["emitted_tokens"] / max(1, delta["rounds"]),
                            )
                        )
                        summarize(report, args.output)
                        print(
                            f"K={k} c={concurrency}: {result['output_tokens_per_s']:.2f} tokens/s; "
                            f"failed={result['failed']}",
                            flush=True,
                        )
                        if result["failed"]:
                            raise RuntimeError("request failure; see report")
            except (RuntimeError, TimeoutError, OSError) as exc:
                report["failures"].append(dict(draft_tokens=k, repeat=repeat, error=str(exc)))
                print(str(exc), flush=True)
            finally:
                summarize(report, args.output)
    if report["summary"]:
        plot(report, args.output)
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
