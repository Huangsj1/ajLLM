from pathlib import Path

from ajllm.config.loader import LoadedConfig
from ajllm.workflows import lm_generate, lm_train, tokenizer_encode, tokenizer_train


def _loaded(project_root: Path, data: dict) -> LoadedConfig:
    config_root = project_root / "configs"
    config_root.mkdir(exist_ok=True)
    source = config_root / "test.yaml"
    return LoadedConfig(data, source, project_root, config_root)


def test_tokenizer_training_lm_training_and_generation(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    train_text = "a little fox and a robot became good friends <|endoftext|>\n" * 30
    validation_text = "the fox looked at the moon <|endoftext|>\n" * 10
    (raw / "train.txt").write_text(train_text, encoding="utf-8")
    (raw / "validation.txt").write_text(validation_text, encoding="utf-8")
    dataset = {
        "name": "tiny",
        "splits": {"train": "data/raw/train.txt", "validation": "data/raw/validation.txt"},
    }
    tokenizer = {"name": "tiny_bpe", "vocab_size": 280, "special_tokens": ["<|endoftext|>"]}

    tokenizer_train.run(_loaded(tmp_path, {"dataset": dataset, "tokenizer": tokenizer, "split": "train"}))
    tokenizer_encode.run(
        _loaded(
            tmp_path,
            {"dataset": dataset, "tokenizer": tokenizer, "splits": ["train", "validation"]},
        )
    )
    model = {
        "name": "tiny_transformer",
        "context_length": 8,
        "d_model": 16,
        "num_layers": 1,
        "num_heads": 4,
        "d_ff": 32,
        "position_encoding": {"type": "rope"},
        "normalization": {"type": "rmsnorm", "placement": "pre"},
        "feed_forward": {"type": "swiglu"},
    }
    training_config = {
        "experiment": "integration",
        "dataset": dataset,
        "tokenizer": tokenizer,
        "model": model,
        "training": {"batch_size": 2, "max_steps": 2},
        "optimizer": {"max_lr": 1e-3, "min_lr_ratio": 0.1},
        "scheduler": {"warmup_steps": 1},
        "logging": {"log_interval": 1, "eval_interval": 1, "eval_batches": 1, "checkpoint_interval": 1},
        "runtime": {"device": "cpu", "seed": 7},
    }
    training_result = lm_train.run(_loaded(tmp_path, training_config))

    generation_result = lm_generate.run(
        _loaded(
            tmp_path,
            {
                "name": "integration_generation",
                "run_directory": training_result["run_directory"],
                "checkpoint": "best",
                "prompts": ["a little fox"],
                "max_new_tokens": 2,
                "temperature": 0.0,
                "device": "cpu",
            },
        )
    )

    assert Path(training_result["checkpoint"]).is_file()
    assert training_result["steps"] == 2
    assert generation_result["results"][0]["completions"][0]["text"].startswith("a little fox")
