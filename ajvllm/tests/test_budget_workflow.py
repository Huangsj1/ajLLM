"""Control-plane checks for reproducible budget experiments (no model execution)."""

import copy
import json
import tomllib
from pathlib import Path

from ajvllm.workflows.benchmark_budget_config import result_row, save_report, sweep_config, write_config


def test_sweep_preserves_context_and_independent_settings(tmp_path):
    base = tomllib.loads(Path("configs/engine/benchmark.toml").read_text())
    base["engine"].update(max_prefill_chunk_size=16, max_prefill_tokens_per_step=32)
    base["memory"]["num_blocks"] = 100
    original = copy.deepcopy(base)
    config = sweep_config(base, 2048, 8, [1, 2, 4, 8])
    assert base == original
    assert config["engine"]["max_model_len"] == 16384
    assert config["engine"]["max_num_batched_tokens"] == 2048
    assert config["engine"]["max_num_seqs"] == 8
    assert "max_prefill_chunk_size" not in config["engine"]
    assert "max_prefill_tokens_per_step" not in config["engine"]
    assert "num_blocks" not in config["memory"]
    assert not config["memory"]["enable_prefix_cache"]
    assert config["graphs"]["memory_limit_mb"] == base["graphs"]["memory_limit_mb"]
    path = tmp_path / "candidate.toml"
    write_config(config, path)
    assert tomllib.loads(path.read_text()) == config


def test_empty_monitor_and_failed_requests_are_reportable(tmp_path):
    result = dict(
        failed=4,
        completed=0,
        concurrency=4,
        input_tokens_per_s=0,
        output_tokens_per_s=0,
        latency=dict(ttft_s=None, tpot_s=None, latency_s=None),
        gpu=dict(utilization_percent=None, memory_used_mib=None),
    )
    before = dict(kv_cache=dict(preemptions=2, admission_waits=3, prefix_hits=0), graphs=dict(replays=4))
    after = dict(
        kv_cache=dict(pool_bytes=1024**2, peak_used_bytes=512**2, preemptions=3, admission_waits=5, prefix_hits=0),
        graphs=dict(replays=10),
    )
    row = result_row("decode", 4, 0, result, before, after)
    assert row["status"] == "failed"
    assert row["ttft_s_mean"] is None
    assert row["gpu_peak_mib"] is None
    assert row["preemptions"] == 1
    assert row["admission_waits"] == 2
    assert row["graph_replays"] == 6
    assert row["kv_peak_percent"] == 25
    report = dict(rows=[row, dict(phase="prefill", value=8192, repeat=0, status="failed", error="startup OOM")])
    save_report(report, tmp_path)
    assert json.loads((tmp_path / "report.json").read_text()) == report
    assert "startup OOM" in (tmp_path / "summary.md").read_text()
    assert "failed" in (tmp_path / "results.csv").read_text()
