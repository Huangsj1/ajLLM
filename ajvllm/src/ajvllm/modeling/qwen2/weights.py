"""Strict local safetensors loading without Transformers model construction."""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from ajvllm.modeling.qwen2.config import Qwen2Config
from ajvllm.modeling.qwen2.model import Qwen2ForCausalLM


def _checkpoint_tensors(directory: Path) -> dict[str, Path]:
    index = directory / "model.safetensors.index.json"
    weight_map = None
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("checkpoint index has no weight_map")
        filenames = set(weight_map.values())
    else:
        filenames = {"model.safetensors"}
    tensors = {}
    for name in sorted(filenames):
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or path.suffix != ".safetensors":
            raise ValueError(f"invalid or missing safetensors shard: {name}")
        with safe_open(path, framework="pt", device="cpu") as shard:
            for key in shard.keys():
                if key in tensors:
                    raise ValueError(f"duplicate checkpoint tensor: {key}")
                if weight_map is not None and weight_map.get(key) != name:
                    raise ValueError(f"checkpoint index disagrees with shard for {key}")
                tensors[key] = path
    if weight_map is not None and set(tensors) != set(weight_map):
        raise ValueError("checkpoint index references missing tensors")
    return tensors


def load_qwen2(
    directory: str | Path, *, device: str | torch.device = "cuda", dtype: torch.dtype | None = None
) -> Qwen2ForCausalLM:
    directory = Path(directory)
    config = Qwen2Config.from_directory(directory)
    dtype = dtype or getattr(torch, config.torch_dtype)
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("supported inference dtypes: float32, float16, bfloat16")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("the model baseline requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the model baseline does not fall back to CPU")
    with torch.cuda.device(device):
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise ValueError("this GPU does not support bfloat16; select float16")
    locations = _checkpoint_tensors(directory)
    with torch.device("meta"):
        model = Qwen2ForCausalLM(config).to(dtype=dtype)
    expected = model.state_dict()
    missing = set(expected) - set(locations)
    tied = {"model.embed_tokens.weight", "lm_head.weight"}
    if config.tie_word_embeddings and tied & set(locations):
        missing -= tied
    unexpected = set(locations) - set(expected)
    if missing or unexpected:
        raise ValueError(f"checkpoint tensor mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    # Validate the entire manifest and tensor shapes before allocating GPU weights.
    for path in set(locations.values()):
        with safe_open(path, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                if tuple(shard.get_slice(name).get_shape()) != tuple(expected[name].shape):
                    raise ValueError(f"checkpoint shape mismatch for {name}")
    # Materialize each unique parameter once. Module.to_empty() recursively
    # allocates tied embedding/head aliases twice before they can be re-tied.
    model.tie_weights()
    allocated = {}
    for module in model.modules():
        for name, parameter in tuple(module.named_parameters(recurse=False, remove_duplicate=False)):
            key = id(parameter)
            if key not in allocated:
                allocated[key] = torch.nn.Parameter(torch.empty_like(parameter, device=device), requires_grad=False)
            setattr(module, name, allocated[key])
    model._init_rope()
    targets = model.state_dict()
    loaded_tied = False
    with torch.no_grad():
        for path in sorted(set(locations.values())):
            with safe_open(path, framework="pt", device="cpu") as shard:
                for name in shard.keys():
                    tensor = shard.get_tensor(name)
                    if not tensor.is_floating_point():
                        raise ValueError(f"expected floating-point checkpoint tensor: {name}")
                    if config.tie_word_embeddings and name in tied:
                        if loaded_tied:
                            if not torch.equal(targets[name], tensor.to(device=device, dtype=dtype)):
                                raise ValueError("checkpoint contains conflicting tied embedding/head weights")
                            continue
                        loaded_tied = True
                    targets[name].copy_(tensor)
    return model.eval().requires_grad_(False)
