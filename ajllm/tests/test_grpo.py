"""CPU coverage for GRPO prompts, objectives, and shared workflow wiring."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from ajllm.datasets import RLAIFDataset, collate_rlaif
from ajllm.modeling import ModelConfig, build_model
from ajllm.tokenization import MiniMindTokenizer
from ajllm.training import grpo_trainer
from ajllm.training.grpo_trainer import group_relative_advantages, grpo_token_loss
from ajllm.training.rewards import SkyworkRewardModel
from ajllm.workflows import causal_lm
from ajllm.workflows import grpo as grpo_workflow
from ajllm.workflows.grpo import run

TOKENIZER_PATH = Path(__file__).resolve().parents[1] / "assets" / "tokenizers" / "minimind"
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def _write_jsonl(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


def _record() -> dict:
    return {
        "conversations": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "It is 4."},
            {"role": "user", "content": "Answer again, briefly."},
            {"role": "assistant", "content": ""},
        ]
    }


def test_grpo_configuration_tree_has_dense_and_moe_launches() -> None:
    for name, model in (("dense", "dense"), ("moe", "moe")):
        config = yaml.safe_load((CONFIG_ROOT / "grpo" / f"{name}.yaml").read_text(encoding="utf-8"))
        assert (CONFIG_ROOT.parent / config["model_config"]).is_file()
        assert config["sft_checkpoint"].startswith(f"output/sft/{model}/")
        assert config["reward_model"]["path"].endswith("Skywork-Reward-V2-Qwen3-0.6B")
        assert config["max_prompt_len"] + config["max_new_tokens"] <= 512


def test_grpo_entry_reuses_the_shared_causal_lm_workflow() -> None:
    assert grpo_workflow.run is causal_lm.run_grpo


def test_rlaif_dataset_creates_a_generation_prompt_and_keeps_it_unpadded(tmp_path) -> None:
    tokenizer = MiniMindTokenizer.from_pretrained(TOKENIZER_PATH)
    data_path = tmp_path / "rlaif.jsonl"
    _write_jsonl(data_path, _record())
    dataset = RLAIFDataset(data_path, tokenizer, max_prompt_length=64, open_thinking=True)
    sample = dataset[0]

    assert sample["completion_prefix"] == "<think>\n"
    assert 0 < len(sample["prompt_ids"]) <= 64
    assert "Answer again, briefly." in tokenizer.decode(sample["prompt_ids"])
    collated = collate_rlaif([sample, sample])
    assert len(collated["prompt_ids"]) == 2
    assert isinstance(collated["prompt_ids"][0], list)


def test_grpo_group_advantages_and_token_objective_match_equations() -> None:
    rewards = torch.tensor([1.0, 3.0, 10.0, 10.0])
    advantages = group_relative_advantages(rewards, num_generations=2, epsilon=1e-4)
    torch.testing.assert_close(advantages[:2], torch.tensor([-1.0, 1.0]), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(advantages[2:], torch.zeros(2))

    current = torch.tensor([0.2, -0.1])
    old = torch.zeros(2)
    reference = torch.tensor([-0.2, -0.2])
    advantage = torch.tensor(1.0)
    loss, surrogate, kl = grpo_token_loss(current, old, reference, advantage, None, clip_epsilon=0.1, kl_coef=0.04)
    ratio = torch.exp(current)
    expected_surrogate = torch.minimum(ratio, torch.clamp(ratio, 0.9, 1.1)).mean()
    log_ratio = reference - current
    expected_kl = (torch.exp(log_ratio) - log_ratio - 1).mean()
    torch.testing.assert_close(surrogate, expected_surrogate)
    torch.testing.assert_close(kl, expected_kl)
    torch.testing.assert_close(loss, -(expected_surrogate - 0.04 * expected_kl))


def test_skywork_reward_adapter_omits_system_messages() -> None:
    messages = [
        {"role": "system", "content": "hidden"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    assert SkyworkRewardModel._without_system_messages(messages) == messages[1:]


def test_grpo_workflow_runs_one_tiny_cpu_step_with_a_mock_reward_model(tmp_path, monkeypatch) -> None:
    class MockRewardModel:
        def __init__(self, _path, device: torch.device, **kwargs) -> None:
            self.device = device

        def score(self, conversations) -> torch.Tensor:
            return torch.arange(len(conversations), device=self.device, dtype=torch.float32)

    monkeypatch.setattr(causal_lm, "SkyworkRewardModel", MockRewardModel)
    cached_logps: list[tuple[torch.Tensor, torch.Tensor]] = []
    original_loss = grpo_trainer.grpo_token_loss

    def record_behavior_policy(current, old, *args, **kwargs):
        cached_logps.append((current.detach().clone(), old.detach().clone()))
        return original_loss(current, old, *args, **kwargs)

    monkeypatch.setattr(grpo_trainer, "grpo_token_loss", record_behavior_policy)
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
    data_path = tmp_path / "rlaif.jsonl"
    _write_jsonl(data_path, _record())
    output_dir = tmp_path / "output"
    config_path = tmp_path / "grpo.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model_config": str(model_path),
                "sft_checkpoint": str(sft_checkpoint),
                "data_path": str(data_path),
                "tokenizer": {"path": str(TOKENIZER_PATH)},
                "reward_model": {"path": str(tmp_path / "reward")},
                "batch_size": 1,
                "epochs": 1,
                "max_steps": 2,
                "max_prompt_len": 20,
                "max_new_tokens": 4,
                "num_generations": 2,
                "updates_per_rollout": 2,
                "open_thinking": False,
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

    summary = run(config_path)

    checkpoint_path = output_dir / "step_00000002.pt"
    assert (summary["epochs"], summary["steps"]) == (1, 2)
    assert checkpoint_path.is_file()
    assert len(cached_logps) == 2
    torch.testing.assert_close(cached_logps[0][0], cached_logps[0][1])
    torch.testing.assert_close(cached_logps[1][1], cached_logps[0][1])
    assert not torch.equal(cached_logps[1][0], cached_logps[1][1])
    metrics = (output_dir / "metrics.jsonl").read_text(encoding="utf-8")
    assert '"group_reward_std"' in metrics
    assert '"kl"' in metrics
