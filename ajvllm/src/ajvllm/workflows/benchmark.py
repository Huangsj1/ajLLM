"""Benchmark an already running server with bounded concurrent streaming requests."""

import argparse
import hashlib
import json
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4


def health(url, timeout):
    with urlopen(url + "/health", timeout=timeout) as response:
        return json.load(response)


def generate(url, item, max_tokens, timeout, sampling=None):
    started = time.perf_counter()
    previous = None
    intervals = []
    first = None
    final = None
    body = {
        "request_id": uuid4().hex,
        "stream": True,
        "sampling": {
            "temperature": 0.8,
            "top_p": 0.9,
            "seed": 0,
            **(sampling or {}),
            "max_tokens": max_tokens,
            "ignore_eos": True,
        },
        **{key: item[key] for key in ("prompt", "messages", "token_ids") if key in item},
    }
    request = Request(url + "/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                event = json.loads(line[6:])
                if event.get("error"):
                    raise RuntimeError(event["error"])
                now = time.perf_counter()
                if event.get("new_token_ids"):
                    if first is None:
                        first = now
                    if previous is not None:
                        intervals.append(now - previous)
                    previous = now
                final = event
        if final is None or final.get("finish_reason") not in ("length", "stop"):
            raise RuntimeError("stream ended without a successful terminal event")
        count = len(final["output_token_ids"])
        latency = time.perf_counter() - started
        return {
            "dataset_id": item.get("id"),
            "output_tokens": count,
            "prompt_tokens": len(final["prompt_token_ids"]),
            "latency_s": latency,
            "ttft_s": None if first is None else first - started,
            "decode_s": None if first is None else previous - first,
            "tpot_s": (previous - first) / (count - 1) if count > 1 else None,
            "itl_s": intervals,
            "server_timing": final.get("timing"),
        }
    except Exception as exc:
        return {"dataset_id": item.get("id"), "error": str(exc)}


def distribution(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    return {
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": values[min(len(values) - 1, int((len(values) - 1) * 0.95 + 0.5))],
        "max": max(values),
    }


def monitor_gpu(stop, samples, errors, gpu, interval):
    while not stop.is_set():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    gpu,
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=3,
            )
            utilization, used, total = map(float, result.stdout.strip().split(","))
            samples.append(
                {
                    "elapsed_clock_s": time.perf_counter(),
                    "utilization_percent": utilization,
                    "memory_used_mib": used,
                    "memory_total_mib": total,
                }
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            errors.append(str(exc))
            return
        stop.wait(interval)


def run_benchmark(
    url,
    dataset,
    *,
    requests=12,
    concurrency=1,
    max_tokens=16,
    warmup=1,
    timeout=60,
    gpu="0",
    interval=0.5,
    sampling=None,
):
    sampling = {"temperature": 0.8, "top_p": 0.9, "seed": 0, **(sampling or {})}
    for index in range(warmup):
        result = generate(url, dataset[index % len(dataset)], max_tokens, timeout, sampling)
        if "error" in result:
            raise RuntimeError("warmup failed: " + result["error"])
    before = health(url, timeout)
    samples, errors = [], []
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_gpu, args=(stop, samples, errors, gpu, interval), daemon=True)
    started = time.perf_counter()
    monitor.start()
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(
                pool.map(
                    lambda index: generate(url, dataset[index % len(dataset)], max_tokens, timeout, sampling),
                    range(requests),
                )
            )
        elapsed = time.perf_counter() - started
    finally:
        stop.set()
        monitor.join(timeout=4)
    after = health(url, timeout)
    successful = [row for row in rows if "error" not in row]
    steps = {}
    for kind, value in after.get("step_timing", {}).items():
        old = before.get("step_timing", {}).get(kind, {"steps": 0, "total_s": 0})
        count, total = value["steps"] - old["steps"], value["total_s"] - old["total_s"]
        steps[kind] = {"steps": count, "total_s": total, "mean_s": total / count if count else None}
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "elapsed_s": elapsed,
        "config": {
            "url": url,
            "requests": requests,
            "concurrency": concurrency,
            "max_tokens": max_tokens,
            "warmup": warmup,
            "sampling": sampling,
            "gpu": gpu,
            "gpu_sample_interval_s": interval,
        },
        "successful_requests": len(successful),
        "failed_requests": requests - len(successful),
        "output_tokens_per_s": sum(row["output_tokens"] for row in successful) / elapsed,
        "requests_per_s": len(successful) / elapsed,
        "latency": {
            key: distribution(row[key] for row in successful) for key in ("ttft_s", "decode_s", "tpot_s", "latency_s")
        },
        "itl_s": distribution(value for row in successful for value in row["itl_s"]),
        "server_steps": steps,
        "stage_seconds": {
            key: value - before.get("stage_seconds", {}).get(key, 0.0)
            for key, value in after.get("stage_seconds", {}).items()
        },
        "transfer_bytes": after.get("transfer_bytes", 0) - before.get("transfer_bytes", 0),
        "gpu": {
            "samples": len(samples),
            "errors": errors,
            "utilization_percent": distribution(row["utilization_percent"] for row in samples),
            "memory_used_mib": distribution(row["memory_used_mib"] for row in samples),
        },
        "health_before": before,
        "health_after": after,
        "requests": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", type=Path, default=Path("benchmarks/datasets/long.jsonl"))
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--gpu", default="0", help="Local nvidia-smi GPU index or UUID")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/latest.json"))
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.max_tokens, args.timeout, args.interval) <= 0 or args.warmup < 0:
        parser.error("counts, timeout and interval must be positive; warmup must be nonnegative")
    raw = args.dataset.read_bytes()
    dataset = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not dataset or any(sum(key in row for key in ("prompt", "messages", "token_ids")) != 1 for row in dataset):
        parser.error("dataset must contain rows with exactly one prompt, messages or token_ids field")
    report = run_benchmark(
        args.url.rstrip("/"),
        dataset,
        requests=args.requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        warmup=args.warmup,
        timeout=args.timeout,
        gpu=args.gpu,
        interval=args.interval,
        sampling={"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k, "seed": args.seed},
    )
    report["dataset"] = {"path": str(args.dataset), "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(dataset)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Completed: {report['successful_requests']}; failed: {report['failed_requests']}")
    for key, value in report["latency"].items():
        if value:
            print(f"{key}: mean {value['mean']:.6f} s, p95 {value['p95']:.6f} s")
    print(f"Output throughput: {report['output_tokens_per_s']:.2f} tokens/s")
    for key, unit in (("utilization_percent", "%"), ("memory_used_mib", "MiB")):
        values = report["gpu"][key]
        if values:
            print(f"GPU {key}: mean {values['mean']:.2f} {unit}, max {values['max']:.2f} {unit}")
    for error in report["gpu"]["errors"]:
        print(f"GPU sampling unavailable: {error}")
    for kind, values in report["server_steps"].items():
        mean = values["mean_s"]
        display = f"{mean:.6f} s" if mean is not None else "no samples"
        print(f"{kind} step mean (synchronized engine wall time): {display}")
    if not report["server_steps"]:
        print("Step timings unavailable; start the server with --profile-steps to enable them.")
    print("Stage totals (s):", json.dumps(report["stage_seconds"]))
    print(f"Sample metadata transfer: {report['transfer_bytes'] / 1024**2:.3f} MiB")
    print(f"Report: {args.output}")
    if report["failed_requests"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
