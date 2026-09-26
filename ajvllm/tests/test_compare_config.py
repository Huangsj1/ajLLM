"""Comparison configuration translation without loading either GPU engine."""

import json
import tomllib

import pytest

from ajvllm.workflows.compare_config import commands, parse_args


def preset(tmp_path, extra=""):
    path = tmp_path / "config.toml"
    path.write_text(
        """[engine]
max_model_len = 16384
max_num_batched_tokens = 2048
max_num_seqs = 6
[memory]
block_size = 16
enable_prefix_cache = false
[graphs]
enabled = true
batch_sizes = [1, 2, 4]
"""
        + extra
    )
    return path


def test_defaults_derive_concurrency_without_shrinking_engine(tmp_path):
    args = parse_args(["--config", str(preset(tmp_path))])
    assert args.concurrency == [1, 2, 4, 6]
    assert args.token_budget == 2048
    cmds, capacity = commands(args, tmp_path)
    assert capacity["context"] == 16384
    assert capacity["max_seqs"] == 6
    assert capacity["kv_pool_bytes"] is None
    assert "--kv-cache-memory-bytes" not in cmds["vllm"]
    assert "--no-enable-prefix-caching" in cmds["vllm"]
    compilation = json.loads(cmds["vllm"][cmds["vllm"].index("--compilation-config") + 1])
    assert compilation["cudagraph_capture_sizes"] == [1, 2, 4]
    saved = tomllib.loads((tmp_path / "ajvllm.toml").read_text())
    assert saved["engine"]["max_model_len"] == 16384
    assert saved["engine"]["max_num_seqs"] == 6


def test_cli_overrides_workload_and_engine_aliases(tmp_path):
    path = preset(tmp_path, "[compare]\nconcurrency = [2, 4]\nrequests = 16\nmax_tokens = 8\n")
    args = parse_args(
        ["--config", str(path), "--concurrency", "1", "--requests", "4", "--token-budget", "512", "--no-graphs"]
    )
    assert args.concurrency == [1]
    assert args.requests == 4
    assert args.max_tokens == 8
    assert args.engine_settings["engine"]["max_num_batched_tokens"] == 512
    assert not args.engine_settings["graphs"]["enabled"]
    cmds, _ = commands(args, tmp_path)
    assert "--enforce-eager" in cmds["vllm"]


def test_explicit_pool_maps_to_equal_bytes(tmp_path):
    path = preset(tmp_path)
    path.write_text(path.read_text().replace("block_size = 16", "block_size = 16\nnum_blocks = 2048"))
    (tmp_path / "config.json").write_text(
        json.dumps(dict(num_hidden_layers=24, num_key_value_heads=2, hidden_size=896, num_attention_heads=14))
    )
    args = parse_args(["--config", str(path), "--model", str(tmp_path)])
    cmds, capacity = commands(args, tmp_path)
    assert capacity["kv_pool_bytes"] == 2048 * 16 * 12288
    assert str(capacity["kv_pool_bytes"]) == cmds["vllm"][cmds["vllm"].index("--kv-cache-memory-bytes") + 1]


@pytest.mark.parametrize(
    "extra", ["[compare]\nmax_tokens=16384\n", "[compare]\nconcurency=[1]\n", '[quantization]\nmode="w8a16"\n']
)
def test_incompatible_settings_fail_before_startup(tmp_path, extra):
    with pytest.raises(SystemExit):
        parse_args(["--config", str(preset(tmp_path, extra))])
