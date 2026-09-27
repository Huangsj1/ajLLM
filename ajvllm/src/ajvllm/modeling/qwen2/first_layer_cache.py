"""Persist and load a vocabulary lookup for the frozen first-layer QKV transform.

Only embedding, first RMSNorm and QKV weights are read by the builder. Artifacts
are content-addressed by those weights, dtype, normalization implementation and
config; RoPE and all context-dependent operations remain in the live model.
"""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

from ajvllm.modeling.qwen2.config import Qwen2Config
from ajvllm.modeling.qwen2.layers import RMSNorm
from ajvllm.modeling.qwen2.weights import _checkpoint_tensors

_KEYS = ["model.embed_tokens.weight", "model.layers.0.input_layernorm.weight"] + [
    f"model.layers.0.self_attn.{projection}_proj.{suffix}"
    for projection in ("q", "k", "v")
    for suffix in ("weight", "bias")
]
_VERSION = 1


def _source(directory, dtype, optimized):
    config = Qwen2Config.from_directory(directory)
    locations = _checkpoint_tensors(Path(directory))
    metadata = dict(version=_VERSION, config=asdict(config), dtype=str(dtype), optimized=optimized)
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    tensors = {}
    for key in _KEYS:
        with safe_open(locations[key], framework="pt", device="cpu") as shard:
            tensor = shard.get_tensor(key).contiguous()
            digest.update(key.encode())
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(memoryview(tensor.view(torch.uint8).numpy()))
            tensors[key] = tensor
    metadata["fingerprint"] = digest.hexdigest()
    return config, tensors, metadata


class FirstLayerFrontEnd(nn.Module):
    """The exact local benchmark boundary: token IDs -> residual embedding and QKV."""

    def __init__(self, config, *, device, dtype, optimized=True):
        super().__init__()
        self.config = config
        self.optimized = optimized
        with torch.device(device):
            self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps).to(dtype=dtype)
            width = config.hidden_size + 2 * config.num_key_value_heads * config.head_dim
            self.qkv = nn.Linear(config.hidden_size, width, dtype=dtype)
        self.register_buffer("table", None, persistent=False)

    @classmethod
    def from_directory(cls, directory, *, device="cuda", dtype=torch.bfloat16, optimized=True):
        if torch.device(device).type != "cuda":
            raise ValueError("first-layer cache computation requires CUDA")
        config, source, metadata = _source(directory, dtype, optimized)
        # Meta construction avoids random initialization of large checkpoint tables.
        module = cls(config, device="meta", dtype=dtype, optimized=optimized)
        module.to_empty(device=device)
        with torch.no_grad():
            module.embedding.weight.copy_(source[_KEYS[0]])
            module.norm.weight.copy_(source[_KEYS[1]])
            offset = 0
            for projection in ("q", "k", "v"):
                weight = source[f"model.layers.0.self_attn.{projection}_proj.weight"]
                width = weight.shape[0]
                module.qkv.weight[offset : offset + width].copy_(weight)
                bias = source[f"model.layers.0.self_attn.{projection}_proj.bias"]
                module.qkv.bias[offset : offset + width].copy_(bias)
                offset += width
        return module.eval().requires_grad_(False), metadata

    def forward(self, token_ids):
        residual = self.embedding(token_ids)
        return residual, self.qkv(self.norm(residual, optimized=self.optimized))

    def lookup(self, token_ids):
        return self.embedding(token_ids), F.embedding(token_ids, self.table)


def artifact_path(directory, metadata):
    return Path(directory) / f"qkv-v{_VERSION}-{metadata['fingerprint']}.safetensors"


@torch.inference_mode()
def build_cache(module, metadata, directory, *, chunk_tokens=4096):
    """Bound GPU workspace by vocabulary chunks; publish a complete file atomically."""
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    path = artifact_path(directory, metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    table = torch.empty(
        module.config.vocab_size, module.qkv.out_features, dtype=module.embedding.weight.dtype, device="cpu"
    )
    for start in range(0, len(table), chunk_tokens):
        end = min(start + chunk_tokens, len(table))
        ids = torch.arange(start, end, device=module.embedding.weight.device)
        table[start:end].copy_(module(ids)[1])
    descriptor, temporary = tempfile.mkstemp(prefix=".qkv-", suffix=".safetensors", dir=path.parent)
    os.close(descriptor)
    try:
        save_file({"qkv": table}, temporary, metadata={"manifest": json.dumps(metadata, sort_keys=True)})
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def load_table(path, metadata, *, device, dtype):
    with safe_open(path, framework="pt", device="cpu") as file:
        if json.loads((file.metadata() or {}).get("manifest", "null")) != metadata:
            raise ValueError("first-layer cache manifest does not match this checkpoint/configuration")
        config = metadata["config"]
        width = config["hidden_size"] + 2 * config["num_key_value_heads"] * (
            config["hidden_size"] // config["num_attention_heads"]
        )
        table = file.get_tensor("qkv")
        if tuple(table.shape) != (config["vocab_size"], width) or table.dtype != dtype:
            raise ValueError("first-layer cache shape or dtype mismatch")
        return table.to(device=device)


def configure_first_layer_cache(model, checkpoint, settings, *, compute_config=None, memory_config=None,
                                quantization_config=None):
    if settings is None or not settings.enabled:
        return
    from ajvllm.attention.backends.triton import resolve_backend
    from ajvllm.config import ComputeConfig, MemoryConfig

    if quantization_config is not None and quantization_config.mode != "none":
        raise ValueError("first-layer QKV cache currently requires unquantized weights")
    optimized = resolve_backend(compute_config or ComputeConfig(), model, memory_config or MemoryConfig()) == "triton"
    _, source, metadata = _source(checkpoint, model.dtype, optimized)
    del source
    path = artifact_path(settings.directory, metadata)
    if not path.is_file():
        backend = "triton" if optimized else "eager"
        raise FileNotFoundError(
            f"missing first-layer QKV cache: {path}; run ajvllm-build-first-layer-cache "
            f"--model {checkpoint} --directory {settings.directory} --dtype {str(model.dtype).split('.')[-1]} "
            f"--backend {backend}"
        )
    model.first_layer_qkv = load_table(path, metadata, device=model.device, dtype=model.dtype)
