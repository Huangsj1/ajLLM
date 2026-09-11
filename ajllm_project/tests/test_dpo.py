"""CPU coverage for DPO data, objective, and workflow wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as functional
import yaml

from ajllm.datasets import DPODataset
from ajllm.modeling import ModelConfig, build_model
from ajllm.tokenization import MiniMindTokenizer
from ajllm.training.dpo_trainer import dpo_loss
from ajllm.workflows import causal_lm
from ajllm.workflows import dpo as dpo_workflow
from ajllm.workflows.dpo import run

TOKENIZER_PATH = Path(__file__).resolve().parents[1] / "assets" / "tokenizers" / "minimind"
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def _write_jsonl(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


def _pair() -> dict:
    return {
        "chosen": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "It is 4."},
        ],
        "rejected": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "It is 5."},
        ],
    }


def test_dpo_configuration_tree_has_dense_and_moe_launches() -> None:
    for name, model in (("dense", "dense"), ("moe", "moe")):
        config = yaml.safe_load((CONFIG_ROOT / "dpo" / f"{name}.yaml").read_text(encoding="utf-8"))
        assert (CONFIG_ROOT.parent / config["model_config"]).is_file()
        assert config["sft_checkpoint"].startswith(f"output/sft/{model}/")
        assert config["data_path"] == "data/dpo.jsonl"
        assert config["beta"] > 0


def test_dpo_entry_reuses_the_shared_causal_lm_workflow() -> None:
    assert dpo_workflow.run is causal_lm.run_dpo


def test_dpo_dataset_masks_prompts_and_requires_a_shared_prefix(tmp_path) -> None:
    tokenizer = MiniMindTokenizer.from_pretrained(TOKENIZER_PATH)
    path = tmp_path / "dpo.jsonl"
    _write_jsonl(path, _pair())
    sample = DPODataset(path, tokenizer, sequence_length=128, pad_token_id=tokenizer.pad_token_id)[0]

    assert set(sample) == {"chosen_input_ids", "chosen_labels", "rejected_input_ids", "rejected_labels"}
    assert sample["chosen_input_ids"].shape == sample["chosen_labels"].shape == (128,)
    assert "It is 4." in tokenizer.decode(sample["chosen_labels"][sample["chosen_labels"] != -100].tolist())
    assert "What is 2 + 2?" not in tokenizer.decode(sample["chosen_labels"][sample["chosen_labels"] != -100].tolist())

    invalid = _pair()
    invalid["rejected"][1]["content"] = "What is 3 + 3?"
    _write_jsonl(path, invalid)
    with pytest.raises(ValueError, match="identical prompt prefix"):
        DPODataset(path, tokenizer, sequence_length=128)[0]


def test_dpo_loss_matches_the_logsigmoid_objective() -> None:
    policy_chosen = torch.tensor([2.0, 1.0])
    policy_rejected = torch.tensor([0.0, 0.0])
    reference_chosen = torch.zeros(2)
    reference_rejected = torch.zeros(2)
    loss, diagnostics = dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta=0.5)

    expected = -functional.logsigmoid(torch.tensor([1.0, 0.5])).mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(diagnostics["reward_margin"], torch.tensor(0.75))
    assert diagnostics["preference_accuracy"].item() == 1.0


def test_dpo_workflow_trains_from_an_sft_checkpoint_on_cpu(tmp_path) -> None:
    model_values = {
        "model_type": "dense",
        "context_length": 32,
        "max_position_embeddings": 64,
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
    sft_checkpoint = tmp_path / "sft.pt"
    torch.save(
        {
            "model_state_dict": build_model(model_config).state_dict(),
            "metadata": {"model_config": model_config.__dict__},
        },
        sft_checkpoint,
    )
    data_path = tmp_path / "dpo.jsonl"
    _write_jsonl(data_path, _pair())
    output_dir = tmp_path / "output"
    config_path = tmp_path / "dpo.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model_config": str(model_path),
                "sft_checkpoint": str(sft_checkpoint),
                "data_path": str(data_path),
                "validation_data_path": str(data_path),
                "tokenizer": {"path": str(TOKENIZER_PATH)},
                "max_seq_len": 32,
                "batch_size": 1,
                "epochs": 1,
                "max_steps": 1,
                "learning_rate": 1e-3,
                "min_lr": 1e-4,
                "beta": 0.1,
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

    checkpoint_path = output_dir / "step_00000001.pt"
    assert (summary["epochs"], summary["steps"]) == (1, 1)
    assert checkpoint_path.is_file()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["metadata"]["stage"] == "dpo"
    metrics = (output_dir / "metrics.jsonl").read_text(encoding="utf-8")
    assert '"event": "evaluation"' in metrics
    assert '"preference_accuracy"' in metrics
