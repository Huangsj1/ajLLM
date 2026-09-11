"""Dense ajLLM <-> Qwen3 weight adapters used by the vLLM rollout path.

The training model intentionally has small, descriptive module names while
vLLM natively serves Qwen3.  The dense ajLLM decoder is structurally
compatible with a bias-free Qwen3 decoder with Q/K RMSNorm, so this module
provides one audited mapping used both for initial checkpoint export and for
online NCCL weight refreshes.  Keeping the two paths on this single mapping is
important: a serving copy that starts from SFT but is refreshed with different
names would silently stop being the policy being optimized.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch

from ajllm.modeling import ModelConfig
from ajllm.tokenization import MiniMindTokenizer


def _require_qwen3_compatible(config: ModelConfig) -> None:
    if config.model_type != "dense":
        raise ValueError("vLLM export currently supports only model_type='dense'")
    # Qwen3 always applies this normalization.  The supplied dense SFT model
    # has it enabled, but reject a different model rather than exporting a
    # superficially valid directory with changed inference semantics.
    if not config.qk_norm:
        raise ValueError("vLLM Qwen3 export requires qk_norm=true")
    if config.dropout != 0.0:
        raise ValueError("vLLM Qwen3 export requires dropout=0.0")


def qwen3_config_dict(config: ModelConfig, tokenizer: MiniMindTokenizer, dtype: torch.dtype) -> dict[str, Any]:
    """Return the HF config for a semantically equivalent dense Qwen3 model."""
    _require_qwen3_compatible(config)
    dtype_name = "bfloat16" if dtype == torch.bfloat16 else "float16" if dtype == torch.float16 else "float32"
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": config.vocab_size,
        "hidden_size": config.d_model,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_layers,
        "num_attention_heads": config.num_heads,
        "num_key_value_heads": config.num_kv_heads,
        "head_dim": config.d_model // config.num_heads,
        "hidden_act": "silu",
        "max_position_embeddings": config.max_position_embeddings,
        "rope_theta": config.rope_theta,
        "rms_norm_eps": config.rms_norm_eps,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": config.tie_embeddings,
        "use_cache": True,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "torch_dtype": dtype_name,
        "transformers_version": "4.57.0",
    }


def ajllm_to_qwen3_name(name: str) -> str:
    """Map one unwrapped dense ajLLM parameter name to a Qwen3 HF name."""
    if name == "token_embeddings.weight":
        return "model.embed_tokens.weight"
    if name == "final_norm.weight":
        return "model.norm.weight"
    if name == "lm_head.weight":
        return "lm_head.weight"

    parts = name.split(".")
    if len(parts) < 4 or parts[0] != "layers" or not parts[1].isdigit():
        raise ValueError(f"Unsupported ajLLM parameter for Qwen3 export: {name}")
    layer = parts[1]
    suffix = ".".join(parts[2:])
    mapped_suffixes = {
        "attn_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
        "attention.q_proj.weight": "self_attn.q_proj.weight",
        "attention.k_proj.weight": "self_attn.k_proj.weight",
        "attention.v_proj.weight": "self_attn.v_proj.weight",
        "attention.output_proj.weight": "self_attn.o_proj.weight",
        "attention.q_norm.weight": "self_attn.q_norm.weight",
        "attention.k_norm.weight": "self_attn.k_norm.weight",
        "feed_forward.gate_proj.weight": "mlp.gate_proj.weight",
        "feed_forward.up_proj.weight": "mlp.up_proj.weight",
        "feed_forward.down_proj.weight": "mlp.down_proj.weight",
    }
    try:
        return f"model.layers.{layer}.{mapped_suffixes[suffix]}"
    except KeyError as error:
        raise ValueError(f"Unsupported ajLLM parameter for Qwen3 export: {name}") from error


def _qwen3_rope_layout(tensor: torch.Tensor, name: str, config: ModelConfig) -> torch.Tensor:
    """Convert ajLLM's interleaved RoPE channels to Qwen3's split-half layout.

    ajLLM represents one head as ``[r0, i0, r1, i1, ...]`` whereas Qwen3
    uses ``[r0, r1, ..., i0, i1, ...]``.  Reordering Q/K projection output
    rows and their Q/K RMSNorm gains makes the two RoPE implementations
    algebraically equivalent without changing attention output channels.
    """
    if not (
        name.endswith(".attention.q_proj.weight")
        or name.endswith(".attention.k_proj.weight")
        or name.endswith(".attention.q_norm.weight")
        or name.endswith(".attention.k_norm.weight")
    ):
        return tensor
    head_dim = config.d_model // config.num_heads
    permutation = torch.cat(
        (torch.arange(0, head_dim, 2, device=tensor.device), torch.arange(1, head_dim, 2, device=tensor.device))
    )
    if name.endswith((".attention.q_norm.weight", ".attention.k_norm.weight")):
        return tensor.index_select(0, permutation)
    heads = config.num_heads if name.endswith(".attention.q_proj.weight") else config.num_kv_heads
    return tensor.reshape(heads, head_dim, -1).index_select(1, permutation).reshape_as(tensor)


def iter_qwen3_named_parameters(policy: torch.nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield policy tensors under the names accepted by vLLM's Qwen3 loader."""
    model_config = getattr(policy, "config", None)
    if not isinstance(model_config, ModelConfig):
        raise TypeError("vLLM dense adapter expects an unwrapped ajLLM TransformerLM")
    _require_qwen3_compatible(model_config)
    for name, parameter in policy.named_parameters():
        yield ajllm_to_qwen3_name(name), _qwen3_rope_layout(parameter, name, model_config)


def _mapped_state_dict(state_dict: Mapping[str, torch.Tensor], config: ModelConfig) -> dict[str, torch.Tensor]:
    _require_qwen3_compatible(config)
    converted: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        converted[ajllm_to_qwen3_name(name)] = _qwen3_rope_layout(tensor, name, config).detach().cpu().contiguous()
    return converted


def export_dense_checkpoint_to_vllm(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Export one dense portable checkpoint into a standalone Qwen3 HF directory.

    The destination must be absent or empty.  This prevents accidentally
    mixing weights from one SFT checkpoint with another checkpoint's tokenizer
    or config.
    """
    checkpoint_path = Path(checkpoint_path)
    tokenizer_path = Path(tokenizer_path)
    destination = Path(output_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(tokenizer_path)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty vLLM export directory: {destination}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_config = checkpoint.get("metadata", {}).get("model_config")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(raw_config, dict) or not isinstance(state_dict, dict):
        raise ValueError(f"{checkpoint_path} is not an ajLLM portable model checkpoint")
    model_config = ModelConfig(**raw_config)
    # Dense checkpoints have no legacy MoE conversion to apply.  Keep this
    # explicit so a MoE checkpoint cannot accidentally pass through.
    _require_qwen3_compatible(model_config)
    tokenizer = MiniMindTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError("checkpoint vocabulary and tokenizer vocabulary differ")
    if not state_dict:
        raise ValueError("checkpoint contains an empty model_state_dict")

    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - declared project dependency
        raise ImportError("vLLM export requires safetensors; run `uv sync`") from error

    mapped = _mapped_state_dict(state_dict, model_config)
    first_tensor = next(iter(mapped.values()))
    destination.mkdir(parents=True, exist_ok=True)
    save_file(mapped, destination / "model.safetensors", metadata={"format": "pt"})
    (destination / "config.json").write_text(
        json.dumps(qwen3_config_dict(model_config, tokenizer, first_tensor.dtype), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # The tokenizer is already HF-compatible; retain its exact BPE IDs and
    # chat template rather than regenerating it through a tokenizer wrapper.
    for source in tokenizer_path.iterdir():
        if source.is_file():
            shutil.copy2(source, destination / source.name)
    (destination / "ajllm_export.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "source_checkpoint": str(checkpoint_path),
                "source_model_config": raw_config,
                "architecture": "Qwen3ForCausalLM",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def validate_qwen3_export_directory(path: str | Path) -> Path:
    """Perform cheap, CPU-only checks before vLLM is launched on a server."""
    directory = Path(path)
    required = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "ajllm_export.json")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete ajLLM vLLM export at {directory}: missing {', '.join(missing)}")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config.get("architectures") != ["Qwen3ForCausalLM"] or config.get("model_type") != "qwen3":
        raise ValueError(f"{directory} is not an ajLLM dense Qwen3 export")
    if config.get("attention_bias") is not False:
        raise ValueError("ajLLM dense Qwen3 export has incompatible attention bias settings")
    return directory
