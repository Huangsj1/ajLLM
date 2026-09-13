"""CPU coverage for MiniMind SFT serialization and workflow wiring."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from ajllm.datasets import SFTDataset
from ajllm.modeling import ModelConfig, build_model
from ajllm.tokenization import MiniMindTokenizer
from ajllm.workflows import generate as generate_workflow
from ajllm.workflows.pretrain import run as run_pretrain
from ajllm.workflows.sft import _upgrade_legacy_top1_moe_state_dict, run

TOKENIZER_PATH = Path(__file__).resolve().parents[1] / "assets" / "tokenizers" / "minimind"
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def _write_jsonl(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


def _targets(sample: dict[str, torch.Tensor]) -> list[int]:
    labels = sample["labels"]
    return labels[labels != -100].tolist()


def test_training_configuration_tree_has_dense_and_moe_sft_launches() -> None:
    expected = {
        "model/dense.yaml",
        "model/moe.yaml",
        "pretrain/dense.yaml",
        "pretrain/dense_tp4.yaml",
        "pretrain/moe.yaml",
        "pretrain/moe_ep4.yaml",
        "pretrain/moe_tp2_ep4.yaml",
        "sft/dense.yaml",
        "sft/dense_tp4.yaml",
        "sft/dense_fsdp2.yaml",
        "sft/moe.yaml",
        "sft/moe_ep4.yaml",
        "sft/moe_tp2_ep4.yaml",
    }
    actual = {str(path.relative_to(CONFIG_ROOT)) for path in CONFIG_ROOT.rglob("*.yaml")}
    assert expected <= actual
    for path in CONFIG_ROOT.glob("sft/*.yaml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert (CONFIG_ROOT.parent / config["model_config"]).is_file()
        assert config["pretrained_checkpoint"].startswith("output/pretrain/")
    tp_ep_config = yaml.safe_load((CONFIG_ROOT / "sft" / "moe_tp2_ep4.yaml").read_text(encoding="utf-8"))
    assert tp_ep_config["parallel"] == {"tp_size": 2, "ep_size": 4, "expert_backend": "torch"}


def test_sft_dataset_masks_prompt_and_keeps_reasoning_and_answer(tmp_path) -> None:
    path = tmp_path / "sft.jsonl"
    _write_jsonl(
        path,
        {
            "conversations": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "reasoning_content": "I should add.", "content": "4"},
            ]
        },
    )
    tokenizer = MiniMindTokenizer.from_pretrained(TOKENIZER_PATH)
    sample = SFTDataset(path, tokenizer, sequence_length=128, pad_token_id=tokenizer.pad_token_id)[0]
    expected = tokenizer.encode(f"<think>\nI should add.\n</think>\n\n4{tokenizer.eos_token}\n")

    assert sample["input_ids"].shape == sample["labels"].shape == (128,)
    assert _targets(sample) == expected
    assert tokenizer.bos_token_id not in _targets(sample)


def test_sft_dataset_supports_tool_calls_without_supervising_tool_responses(tmp_path) -> None:
    path = tmp_path / "tool_sft.jsonl"
    _write_jsonl(
        path,
        {
            "conversations": [
                {
                    "role": "system",
                    "content": "",
                    "tools": '[{"function":{"name":"weather","parameters":{"type":"object"}}}]',
                },
                {"role": "user", "content": "What is the weather?"},
                {
                    "role": "assistant",
                    "content": "Checking.",
                    "tool_calls": '[{"function":{"name":"weather","arguments":"{}"}}]',
                },
                {"role": "tool", "content": "{\"temperature\": 20}"},
                {"role": "assistant", "content": "It is 20°C."},
            ]
        },
    )
    tokenizer = MiniMindTokenizer.from_pretrained(TOKENIZER_PATH)
    sample = SFTDataset(path, tokenizer, sequence_length=256, pad_token_id=tokenizer.pad_token_id)[0]
    target_text = tokenizer.decode(_targets(sample))

    assert "Checking." in target_text
    assert '<tool_call>\n{"name": "weather", "arguments": {}}\n</tool_call>' in target_text
    assert "It is 20°C." in target_text
    assert "temperature" not in target_text
    assert "What is the weather?" not in target_text


def test_legacy_top1_moe_checkpoint_is_packed_into_grouped_expert_weights() -> None:
    model_config = ModelConfig(
        vocab_size=64,
        context_length=16,
        max_position_embeddings=32,
        d_model=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        d_ff=64,
        model_type="moe",
        num_experts=2,
        use_flash_attention=False,
        use_cuda_kernels=False,
    )
    model = build_model(model_config)
    grouped_state = model.state_dict()
    legacy_state = {
        key: value.clone()
        for key, value in grouped_state.items()
        if "grouped_experts.gate_up_proj.weight" not in key and "grouped_experts.down_proj.weight" not in key
    }
    prefix = "layers.0.feed_forward"
    gate_up = grouped_state[f"{prefix}.grouped_experts.gate_up_proj.weight"]
    down = grouped_state[f"{prefix}.grouped_experts.down_proj.weight"]
    for expert_index in range(model_config.num_experts):
        gate, up = gate_up[expert_index].chunk(2, dim=0)
        legacy_state[f"{prefix}.experts.{expert_index}.gate_proj.weight"] = gate.clone()
        legacy_state[f"{prefix}.experts.{expert_index}.up_proj.weight"] = up.clone()
        legacy_state[f"{prefix}.experts.{expert_index}.down_proj.weight"] = down[expert_index].clone()

    converted = _upgrade_legacy_top1_moe_state_dict(legacy_state, model)
    model.load_state_dict(converted)

    assert torch.equal(converted[f"{prefix}.grouped_experts.gate_up_proj.weight"], gate_up)
    assert torch.equal(converted[f"{prefix}.grouped_experts.down_proj.weight"], down)


def test_chat_generation_executes_only_registered_tools_and_continues(monkeypatch) -> None:
    tokenizer = MiniMindTokenizer.from_pretrained(TOKENIZER_PATH)
    generated_texts = iter(
        [
            '<tool_call>\n{"name": "double", "arguments": {"value": 21}}\n</tool_call>',
            "The result is 42.",
        ]
    )

    def fake_generate_ids(*args, **kwargs):
        return tokenizer.encode(next(generated_texts))

    monkeypatch.setattr(generate_workflow, "_generate_ids", fake_generate_ids)
    result = generate_workflow.generate_chat(
        model=object(),
        tokenizer=tokenizer,
        messages=[{"role": "user", "content": "Double 21."}],
        max_new_tokens=32,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        device=torch.device("cpu"),
        tools=[{"function": {"name": "double", "parameters": {"type": "object"}}}],
        tool_registry={"double": lambda value: {"result": value * 2}},
    )

    assert result.message == {"role": "assistant", "content": "The result is 42."}
    assert result.tool_rounds == 1
    assert [message["role"] for message in result.messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert result.messages[3]["content"] == '{"result": 42}'


def test_sft_workflow_initializes_from_pretraining_checkpoint_on_cpu(tmp_path) -> None:
    model_values = {
        "model_type": "dense",
        "context_length": 16,
        "max_position_embeddings": 32,
        "d_model": 64,
        "num_layers": 1,
        "num_heads": 4,
        "num_kv_heads": 2,
        "d_ff": 128,
        "use_flash_attention": False,
        "use_cuda_kernels": False,
    }
    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump(model_values), encoding="utf-8")
    tokenizer = MiniMindTokenizer.from_pretrained(TOKENIZER_PATH)
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size, **model_values)
    checkpoint_path = tmp_path / "pretrain.pt"
    torch.save(
        {
            "model_state_dict": build_model(model_config).state_dict(),
            "metadata": {"model_config": model_config.__dict__},
        },
        checkpoint_path,
    )
    data_path = tmp_path / "sft.jsonl"
    _write_jsonl(
        data_path,
        {"conversations": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]},
    )
    output_dir = tmp_path / "output"
    config_path = tmp_path / "sft.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model_config": str(model_path),
                "pretrained_checkpoint": str(checkpoint_path),
                "data_path": str(data_path),
                "validation_data_path": str(data_path),
                "tokenizer": {"path": str(TOKENIZER_PATH)},
                "max_seq_len": 16,
                "batch_size": 1,
                "epochs": 1,
                "max_steps": 1,
                "learning_rate": 1e-3,
                "min_lr": 1e-4,
                "output_dir": str(output_dir),
                "device": "cpu",
                "mixed_precision": None,
                "log_interval": 1,
                "eval_interval": 1,
                "eval_batches": 1,
                "save_interval": 1,
                "use_fsdp": False,
            }
        ),
        encoding="utf-8",
    )

    summary = run(config_path)

    assert (summary["epochs"], summary["steps"]) == (1, 1)
    assert (output_dir / "step_00000001.pt").is_file()
    assert '"event": "evaluation"' in (output_dir / "metrics.jsonl").read_text(encoding="utf-8")


def test_pretrain_entry_uses_shared_causal_lm_workflow_on_cpu(tmp_path) -> None:
    model_values = {
        "model_type": "dense",
        "context_length": 16,
        "max_position_embeddings": 32,
        "d_model": 64,
        "num_layers": 1,
        "num_heads": 4,
        "num_kv_heads": 2,
        "d_ff": 128,
        "use_flash_attention": False,
        "use_cuda_kernels": False,
    }
    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump(model_values), encoding="utf-8")
    data_path = tmp_path / "pretrain.jsonl"
    data_path.write_text('{"text": "A tiny training document."}\n', encoding="utf-8")
    output_dir = tmp_path / "output"
    config_path = tmp_path / "pretrain.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model_config": str(model_path),
                "data_path": str(data_path),
                "tokenizer": {"path": str(TOKENIZER_PATH)},
                "max_seq_len": 16,
                "batch_size": 1,
                "epochs": 1,
                "max_steps": 1,
                "learning_rate": 1e-3,
                "min_lr": 1e-4,
                "output_dir": str(output_dir),
                "device": "cpu",
                "mixed_precision": None,
                "log_interval": 1,
                "save_interval": 1,
                "use_fsdp": False,
            }
        ),
        encoding="utf-8",
    )

    summary = run_pretrain(config_path)

    assert (summary["epochs"], summary["steps"]) == (1, 1)
    assert (output_dir / "step_00000001.pt").is_file()
