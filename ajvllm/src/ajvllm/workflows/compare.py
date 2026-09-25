"""Serial single-GPU HTTP comparison against vLLM, using identical token inputs."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

from ajvllm.workflows.benchmark import distribution, monitor_gpu


def stream_request(engine, url, tokens, args, seed):
    sampling = dict(temperature=0.8, top_p=0.9, seed=seed, max_tokens=args.max_tokens, ignore_eos=True)
    if engine == "ajvllm":
        endpoint = "/generate"
        body = dict(token_ids=tokens, stream=True, sampling=sampling)
    else:
        endpoint = "/v1/completions"
        body = dict(
            model="comparison",
            prompt=tokens,
            stream=True,
            return_token_ids=True,
            stream_options={"include_usage": True},
            **sampling,
        )
    started = time.perf_counter()
    arrivals, count, finish, usage = [], 0, None, None
    try:
        request = Request(url + endpoint, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urlopen(request, timeout=args.timeout) as response:
            for line in response:
                if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                    continue
                event = json.loads(line[6:])
                now = time.perf_counter()
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                if engine == "ajvllm":
                    ids = event.get("new_token_ids", [])
                    finish = event.get("finish_reason") or finish
                else:
                    choices = event.get("choices", [])
                    ids = choices[0].get("token_ids", []) if choices else []
                    finish = (choices[0].get("finish_reason") if choices else None) or finish
                    usage = event.get("usage") or usage
                if ids:
                    count += len(ids)
                    arrivals.append({"time_s": now - started, "tokens": len(ids)})
        if count != args.max_tokens or finish != "length":
            raise RuntimeError(f"expected {args.max_tokens} tokens and length finish, got {count}, {finish}")
        if usage and (usage["completion_tokens"] != count or usage["prompt_tokens"] != len(tokens)):
            raise RuntimeError("server token usage differs from the submitted workload")
        return dict(
            prompt_tokens=len(tokens),
            output_tokens=count,
            ttft_s=arrivals[0]["time_s"],
            tpot_s=(arrivals[-1]["time_s"] - arrivals[0]["time_s"]) / (count - 1),
            latency_s=time.perf_counter() - started,
            arrivals=arrivals,
        )
    except Exception as exc:
        return {"error": str(exc)}


def measure(engine, url, dataset, concurrency, args, *, warmup=False):
    count = max(concurrency * 2, len(dataset)) if warmup else args.requests
    samples, errors, stop = [], [], threading.Event()
    monitor = threading.Thread(target=monitor_gpu, args=(stop, samples, errors, args.gpu, 0.25), daemon=True)
    if not warmup:
        monitor.start()
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(
                pool.map(lambda i: stream_request(engine, url, dataset[i % len(dataset)], args, i), range(count))
            )
        elapsed = time.perf_counter() - started
    finally:
        stop.set()
        if not warmup:
            monitor.join(timeout=4)
    good = [row for row in rows if "error" not in row]
    return dict(
        engine=engine,
        concurrency=concurrency,
        elapsed_s=elapsed,
        completed=len(good),
        failed=count - len(good),
        output_tokens_per_s=sum(row["output_tokens"] for row in good) / elapsed,
        requests_per_s=len(good) / elapsed,
        input_tokens_per_s=sum(row["prompt_tokens"] for row in good) / elapsed,
        latency={key: distribution(row[key] for row in good) for key in ("ttft_s", "tpot_s", "latency_s")},
        gpu={key: distribution(row[key] for row in samples) for key in ("utilization_percent", "memory_used_mib")},
        gpu_errors=errors,
        requests=rows,
    )


@contextmanager
def server(command, logfile, args):
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        "MAX_JOBS": "2",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "VLLM_NO_USAGE_STATS": "1",
    }
    url = f"http://127.0.0.1:{args.port}"
    # Refuse to benchmark or terminate a pre-existing server.
    try:
        with urlopen(url + "/health", timeout=1):
            raise RuntimeError(f"port {args.port} already hosts a service")
    except OSError:
        pass
    with logfile.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited ({process.returncode}); see {logfile}")
                try:
                    with urlopen(url + "/health", timeout=2):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"server startup timeout; see {logfile}")
                    time.sleep(1)
            yield url
        finally:
            # Kill only the process group created by this workflow, including vLLM workers.
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def commands(args, output):
    cfg = json.loads((args.model / "config.json").read_text())
    context = max(args.prompt_lengths) + args.max_tokens
    max_seqs = max(args.concurrency)
    # vLLM reserves a null block; a single-slot pool needs one spare page to finish decode.
    blocks = max_seqs * ((context + 15) // 16) + int(max_seqs == 1)
    kv_bytes = (
        blocks
        * 16
        * cfg["num_hidden_layers"]
        * 2
        * cfg["num_key_value_heads"]
        * (cfg["hidden_size"] // cfg["num_attention_heads"])
        * 2
    )
    config = output / "ajvllm.toml"
    config.write_text(f"""[engine]
max_num_seqs = {max_seqs}
max_num_batched_tokens = {args.token_budget}
max_model_len = {context}
enable_chunked_prefill = true
max_prefill_chunk_size = {args.token_budget}
max_prefill_tokens_per_step = {args.token_budget}
[memory]
backend = "paged"
block_size = 16
num_blocks = {blocks}
enable_prefix_cache = false
[compute]
backend = "triton"
[graphs]
enabled = {str(args.graphs).lower()}
batch_sizes = {sorted(set(args.concurrency))}
max_graphs = 32
memory_limit_mb = 512
[quantization]
mode = "none"
""")
    common = ["--host", "127.0.0.1", "--port", str(args.port), "--dtype", "bfloat16", "--gpu-memory-utilization", "0.4"]
    aj = [sys.executable, "-m", "ajvllm.workflows.serve", "--model", str(args.model), "--config", str(config), *common]
    vl = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(args.model),
        "--served-model-name",
        "comparison",
        "--max-model-len",
        str(context),
        "--max-num-seqs",
        str(max_seqs),
        "--max-num-batched-tokens",
        str(args.token_budget),
        "--block-size",
        "16",
        "--kv-cache-memory-bytes",
        str(kv_bytes),
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--generation-config",
        "vllm",
        "--no-enable-log-requests",
        *common,
    ]
    if args.graphs:
        vl += [
            "--compilation-config",
            json.dumps(
                {
                    "mode": 0,
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                    "cudagraph_capture_sizes": sorted(set(args.concurrency)),
                }
            ),
        ]
    else:
        vl += ["--enforce-eager"]
    return {"ajvllm": aj, "vllm": vl}, {"context": context, "max_seqs": max_seqs, "kv_pool_bytes": kv_bytes}


def summarize(report):
    """Aggregate repeat-level metrics; do not pretend a short run is a confidence interval."""
    rows = []
    for concurrency in report["config"]["concurrency"]:
        engines = {}
        for engine in ("ajvllm", "vllm"):
            runs = [
                r
                for r in report["runs"]
                if r["engine"] == engine and r["concurrency"] == concurrency and not r["failed"]
            ]
            engines[engine] = {
                "repeats": len(runs),
                "output_tokens_per_s": distribution(r["output_tokens_per_s"] for r in runs),
                **{
                    key: distribution(r["latency"][key]["mean"] for r in runs)
                    for key in ("ttft_s", "tpot_s", "latency_s")
                },
            }
        a, v = engines["ajvllm"]["output_tokens_per_s"], engines["vllm"]["output_tokens_per_s"]
        rows.append(
            {
                "concurrency": concurrency,
                **engines,
                "vllm_over_ajvllm_throughput": v["mean"] / a["mean"] if a and v else None,
            }
        )
    return rows


def plot_report(report, target, engines=("ajvllm", "vllm")):
    import matplotlib

    matplotlib.use("Agg")
    from statistics import mean, stdev

    import matplotlib.pyplot as plt

    panels = [
        ("Output throughput (tokens/s) ↑", lambda r: r["output_tokens_per_s"]),
        ("Mean TTFT (ms) ↓", lambda r: r["latency"]["ttft_s"]["mean"] * 1000),
        ("Mean TPOT (ms) ↓", lambda r: r["latency"]["tpot_s"]["mean"] * 1000),
        ("P95 request latency (s) ↓", lambda r: r["latency"]["latency_s"]["p95"]),
        (
            "Peak sampled GPU memory (GiB) ↓",
            lambda r: (r["gpu"]["memory_used_mib"] or {}).get("max", float("nan")) / 1024,
        ),
        (
            "Mean sampled GPU utilization (%)",
            lambda r: (r["gpu"]["utilization_percent"] or {}).get("mean", float("nan")),
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (title, metric) in zip(axes.flat, panels, strict=True):
        for engine, color in zip(engines, ("#167d9a", "#db7433", "#698537")[: len(engines)], strict=True):
            groups = [
                [
                    metric(r)
                    for r in report["runs"]
                    if r["engine"] == engine and r["concurrency"] == c and not r["failed"]
                ]
                for c in report["config"]["concurrency"]
            ]
            values = [mean(g) if g else float("nan") for g in groups]
            ax.errorbar(
                report["config"]["concurrency"],
                values,
                yerr=[stdev(g) if len(g) > 1 else 0 for g in groups],
                color=color,
                marker="o",
                capsize=4,
                label=engine,
            )
        ax.set(title=title, xlabel="Concurrency", xticks=report["config"]["concurrency"])
        ax.set_ylim(bottom=0)
        if "utilization" in title:
            ax.set_ylim(0, 100)
        ax.grid(alpha=0.2)
        ax.legend()
    title = report.get("title", f"Qwen2.5 single-GPU HTTP comparison | BF16 | graphs={report['config']['graphs']}")
    fig.suptitle(
        title + "\nIdentical token inputs; prefix cache off; error bars: across-run standard deviation",
        fontsize=14,
    )
    fig.savefig(target, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("model/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--dataset", type=Path, default=Path("benchmarks/datasets/long.jsonl"))
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--token-budget", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/comparison"))
    args = parser.parse_args()
    if (
        min(*args.concurrency, *args.prompt_lengths, args.requests, args.token_budget, args.repeats) < 1
        or args.max_tokens < 2
    ):
        parser.error("counts must be positive and max-tokens >= 2")
    if args.requests < max(args.concurrency):
        parser.error("requests must be >= maximum concurrency")
    args.model = args.model.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    # Use the checkpoint tokenizer directly; no model or CUDA context in the coordinator.
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    raw = args.dataset.read_bytes()
    source = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not source:
        parser.error("dataset is empty")
    dataset = []
    for i, length in enumerate(args.prompt_lengths):
        row = source[i % len(source)]
        ids = row.get("token_ids")
        if ids is None:
            if "prompt" not in row:
                parser.error("comparison dataset requires prompt or token_ids (no implicit chat templates)")
            ids = tokenizer.encode(row["prompt"], add_special_tokens=False).ids
        if len(ids) < length:
            parser.error(f"dataset row {i} has only {len(ids)} tokens; requested {length}")
        dataset.append(ids[:length])
    workloads = json.dumps(dataset).encode()
    (args.output / "inputs.json").write_bytes(workloads)
    cmds, capacity = commands(args, args.output)
    report = dict(
        created_at=datetime.now(UTC).isoformat(),
        complete=False,
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        capacity=capacity,
        versions={k: importlib.metadata.version(k) for k in ("torch", "triton", "vllm", "transformers")},
        dataset_sha256=hashlib.sha256(raw).hexdigest(),
        token_inputs_sha256=hashlib.sha256(workloads).hexdigest(),
        commands=cmds,
        server_environment={
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "MAX_JOBS": "2",
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "HF_HUB_OFFLINE": "1",
        },
        runs=[],
    )
    report["hardware"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], text=True
    )
    try:
        for repeat in range(args.repeats):
            for engine in ["ajvllm", "vllm"] if repeat % 2 == 0 else ["vllm", "ajvllm"]:
                print(f"Starting {engine}, repeat {repeat + 1}/{args.repeats}", flush=True)
                with server(cmds[engine], args.output / f"{engine}-{repeat}.log", args) as url:
                    if engine == "ajvllm":
                        with urlopen(url + "/health") as response:
                            report.setdefault("ajvllm_health", []).append(json.load(response))
                    for c in args.concurrency:
                        warm = measure(engine, url, dataset, c, args, warmup=True)
                        if warm["failed"]:
                            raise RuntimeError(f"warmup failed: {warm['requests']}")
                        before = None
                        if engine == "ajvllm":
                            with urlopen(url + "/health", timeout=args.timeout) as response:
                                before = json.load(response)
                            resolved = before["resolved_engine"]
                            if (
                                resolved["max_num_seqs"] != capacity["max_seqs"]
                                or before["memory"]["token_budget"] != args.token_budget
                                or before["kv_cache"]["pool_bytes"] != capacity["kv_pool_bytes"]
                            ):
                                raise RuntimeError("ajvllm adjusted shared capacity; comparison would be unequal")
                        result = measure(engine, url, dataset, c, args)
                        if engine == "ajvllm":
                            with urlopen(url + "/health", timeout=args.timeout) as response:
                                result["health_after"] = json.load(response)
                            result["health_before"] = before
                            if result["health_after"]["memory"]["token_budget"] != args.token_budget:
                                raise RuntimeError("ajvllm changed token budget during measurement")
                        result["repeat"] = repeat
                        report["runs"].append(result)
                        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                        print(
                            f"{engine} c={c}: {result['output_tokens_per_s']:.2f} tokens/s, "
                            f"failures={result['failed']}",
                            flush=True,
                        )
                        if result["failed"]:
                            raise RuntimeError("failed measurement; inspect report requests")
        report["complete"] = True
    finally:
        report["summary"] = summarize(report)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if report["runs"]:
            plot_report(report, args.output / "comparison.png")
    for row in report["summary"]:
        ratio = row["vllm_over_ajvllm_throughput"]
        if ratio is not None:
            print(f"c={row['concurrency']}: vLLM/ajvllm throughput = {ratio:.3f}x")
    print(f"Report and plot: {args.output}")


if __name__ == "__main__":
    main()
