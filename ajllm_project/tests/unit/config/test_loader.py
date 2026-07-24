from pathlib import Path

from ajllm.config.loader import load_run_config


def test_loader_resolves_components_and_base(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\nversion='0.0.0'\n", encoding="utf-8")
    for directory in ("datasets", "tokenizers", "models", "runs"):
        (tmp_path / "configs" / directory).mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs/datasets/demo.yaml").write_text("name: demo\nsplits: {train: train.txt}\n", encoding="utf-8")
    (tmp_path / "configs/tokenizers/demo.yaml").write_text("name: demo\nvocab_size: 300\n", encoding="utf-8")
    (tmp_path / "configs/models/demo.yaml").write_text(
        "name: demo\ncontext_length: 8\nd_model: 16\nnum_layers: 1\nnum_heads: 4\n",
        encoding="utf-8",
    )
    (tmp_path / "configs/runs/base.yaml").write_text(
        "dataset: demo\ntokenizer: demo\nmodel: demo\ntraining: {batch_size: 2}\n",
        encoding="utf-8",
    )
    child = tmp_path / "configs/runs/child.yaml"
    child.write_text("base: base.yaml\ntraining: {batch_size: 4}\n", encoding="utf-8")

    loaded = load_run_config(child)

    assert loaded.data["dataset"]["name"] == "demo"
    assert loaded.data["model"]["d_model"] == 16
    assert loaded.data["training"]["batch_size"] == 4
