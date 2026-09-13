"""CPU-only checks for the dense ajLLM -> Qwen3/vLLM adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ajllm.modeling import ModelConfig, build_model
from ajllm.tokenization import MiniMindTokenizer
from ajllm.utils import vllm_util
from ajllm.utils.vllm_export import (
    ajllm_to_qwen3_name,
    export_dense_checkpoint_to_vllm,
    iter_qwen3_named_parameters,
    validate_qwen3_export_directory,
)

TOKENIZER_PATH = Path(__file__).resolve().parents[1] / "assets" / "tokenizers" / "minimind"


def _tiny_config(vocab_size: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        context_length=32,
        max_position_embeddings=64,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=64,
        dropout=0.0,
        qk_norm=True,
        tie_embeddings=True,
        use_flash_attention=False,
        use_cuda_kernels=False,
    )


def test_dense_export_loads_as_qwen3_and_preserves_cpu_logits(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    tokenizer = MiniMindTokenizer.from_pretrained(TOKENIZER_PATH)
    config = _tiny_config(tokenizer.vocab_size)
    torch.manual_seed(7)
    source = build_model(config).eval()
    checkpoint_path = tmp_path / "sft.pt"
    torch.save(
        {"model_state_dict": source.state_dict(), "metadata": {"model_config": config.__dict__}}, checkpoint_path
    )

    exported = export_dense_checkpoint_to_vllm(checkpoint_path, TOKENIZER_PATH, tmp_path / "qwen3")
    assert validate_qwen3_export_directory(exported) == exported
    target = transformers.AutoModelForCausalLM.from_pretrained(exported, dtype=torch.float32).eval()
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(exported)
    assert hf_tokenizer.encode("token-ID parity", add_special_tokens=False) == tokenizer.encode("token-ID parity")
    input_ids = torch.tensor([[1, 3, 5, 7], [2, 4, 6, 8]])
    with torch.no_grad():
        expected = source(input_ids)
        actual = target(input_ids).logits
    # SDPA kernel reduction order differs slightly from Transformers/Qwen3's
    # implementation.  The export must nevertheless be much closer than an
    # fp16/bf16 rollout tolerance (observed max absolute error is ~1e-4).
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=2e-4)
    exported_config = json.loads((exported / "config.json").read_text(encoding="utf-8"))
    assert exported_config["architectures"] == ["Qwen3ForCausalLM"]
    assert exported_config["head_dim"] == 8


def test_dense_export_rejects_moe_and_nonempty_destination(tmp_path: Path) -> None:
    tokenizer = MiniMindTokenizer.from_pretrained(TOKENIZER_PATH)
    config = _tiny_config(tokenizer.vocab_size)
    checkpoint_path = tmp_path / "sft.pt"
    torch.save(
        {
            "model_state_dict": build_model(config).state_dict(),
            "metadata": {"model_config": config.__dict__},
        },
        checkpoint_path,
    )
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty"):
        export_dense_checkpoint_to_vllm(checkpoint_path, TOKENIZER_PATH, destination)

    moe_config = ModelConfig(**{**config.__dict__, "model_type": "moe"})
    moe_path = tmp_path / "moe.pt"
    torch.save(
        {"model_state_dict": build_model(moe_config).state_dict(), "metadata": {"model_config": moe_config.__dict__}},
        moe_path,
    )
    with pytest.raises(ValueError, match="only model_type='dense'"):
        export_dense_checkpoint_to_vllm(moe_path, TOKENIZER_PATH, tmp_path / "moe")


def test_vllm_weight_names_and_token_id_completion_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ajllm_to_qwen3_name("layers.2.attention.q_proj.weight") == "model.layers.2.self_attn.q_proj.weight"
    config = _tiny_config(32)
    names = dict(iter_qwen3_named_parameters(build_model(config)))
    assert "model.embed_tokens.weight" in names
    assert "model.layers.0.self_attn.q_norm.weight" in names
    assert names["model.layers.0.self_attn.q_proj.weight"].shape == (32, 32)

    requests: list[dict] = []

    def fake_request(method: str, url: str, payload: dict | None = None, timeout: int = 60) -> dict:
        assert method == "POST"
        assert url.endswith("/v1/completions")
        assert payload is not None
        requests.append(payload)
        return {
            "choices": [
                {"index": 0, "text": "", "token_ids": [9], "finish_reason": "length"},
                {"index": 1, "text": "", "token_ids": [10], "finish_reason": "length"},
            ]
        }

    monkeypatch.setattr(vllm_util, "_http_json", fake_request)
    completions = vllm_util.generate_completions(
        "http://server", "policy", [[1, 2, 3]], {"temperature": 1.0, "max_tokens": 8, "n": 2, "seed": 4}
    )
    assert [completion.token_ids for completion in completions] == [[9], [10]]
    assert requests[0]["prompt"] == [[1, 2, 3]]
    assert requests[0]["add_special_tokens"] is False


def test_weight_sync_resolves_unindexed_cuda_to_the_current_device(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    monkeypatch.setattr(vllm_util.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(vllm_util.torch.cuda, "set_device", lambda index: calls.append(index))
    monkeypatch.setattr(vllm_util, "_http_json", lambda *args, **kwargs: {"world_size": 1})
    monkeypatch.setattr(vllm_util, "get_ip", lambda: "127.0.0.1", raising=False)

    # Stop before initializing the real NCCL group; the device selection is
    # the regression fixed here and must remain testable without a GPU.
    class StopHere(Exception):
        pass

    class FakeEngine:
        @staticmethod
        def trainer_init(_config: dict) -> None:
            raise StopHere

    import sys
    import types

    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.weight_transfer.nccl_engine",
        types.SimpleNamespace(NCCLWeightTransferEngine=FakeEngine),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils.network_utils",
        types.SimpleNamespace(get_ip=lambda: "127.0.0.1", get_open_port=lambda: 1),
    )
    with pytest.raises(StopHere):
        vllm_util.init_weight_sync("http://server", "cuda")
    assert calls == [0]
