"""Read the supported dense Qwen2.5 architecture from a local checkpoint."""

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

from ajvllm.config import require_int


@dataclass(frozen=True)
class Qwen2Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
        ):
            require_int(name, getattr(self, name), 1)
        if self.hidden_size % self.num_attention_heads or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("hidden_size must divide into query heads; query heads must divide into KV groups")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        if any(not math.isfinite(v) or v <= 0 for v in (self.rms_norm_eps, self.rope_theta)):
            raise ValueError("rms_norm_eps and rope_theta must be finite and positive")
        if type(self.tie_word_embeddings) is not bool:
            raise ValueError("tie_word_embeddings must be a boolean")
        if self.torch_dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("unsupported checkpoint dtype")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_directory(cls, directory: str | Path) -> "Qwen2Config":
        data = json.loads((Path(directory) / "config.json").read_text())
        if data.get("model_type") != "qwen2" or data.get("architectures", ["Qwen2ForCausalLM"]) != ["Qwen2ForCausalLM"]:
            raise ValueError("only dense Qwen2ForCausalLM checkpoints are supported")
        if data.get("hidden_act", "silu") != "silu":
            raise ValueError("only the SiLU gated MLP is supported")
        if data.get("use_sliding_window", False) or any(
            layer != "full_attention" for layer in data.get("layer_types", [])
        ):
            raise ValueError("sliding-window attention is not implemented")
        if data.get("rope_scaling") not in (None, {}):
            raise ValueError("scaled RoPE is not implemented")
        if data.get("quantization_config") is not None or data.get("num_experts", 0):
            raise ValueError("quantized and MoE checkpoints are not supported")
        if data.get("head_dim", data["hidden_size"] // data["num_attention_heads"]) != (
            data["hidden_size"] // data["num_attention_heads"]
        ):
            raise ValueError("custom head_dim is not supported")
        if data.get("attention_bias", True) is not True or data.get("mlp_bias", False) is not False:
            raise ValueError("expected QKV bias and bias-free output/MLP projections")
        values = {field.name: data[field.name] for field in fields(cls) if field.name in data}
        if "dtype" in data:
            values["torch_dtype"] = data["dtype"]
        return cls(**values)
