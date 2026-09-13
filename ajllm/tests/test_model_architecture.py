"""CUDA tests for the canonical model and one complete optimizer step."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from ajllm.modeling import ModelConfig, TransformerLM, load_model_config
from ajllm.modeling.flash_attention import flash_attention_pytorch
from ajllm.tokenization import MiniMindTokenizer
from ajllm.training.checkpoint import load_checkpoint, save_checkpoint
from ajllm.training.logger import RunLogger
from ajllm.training.losses import cross_entropy
from ajllm.training.optimizers import AdamW
from ajllm.training.parallel.fsdp import FullyShardedDataParallel
from ajllm.utils.view_dataset import inspect_jsonl
from ajllm.workflows.compare import plot_metric
from ajllm.workflows.evaluate import evaluate_checkpoint
from ajllm.workflows.generate import generate
from ajllm.workflows.pretrain import run

TOKENIZER_PATH = Path(__file__).resolve().parents[1] / "assets" / "tokenizers" / "minimind"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the decoder requires CUDA + Triton")


def _config(model_type: str = "dense") -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        context_length=16,
        max_position_embeddings=64,
        d_model=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        d_ff=128,
        model_type=model_type,
        num_experts=2,
    )


def test_dense_forward_and_tied_weights() -> None:
    model = TransformerLM(_config()).cuda()
    logits = model(torch.randint(0, 64, (3, 16), device="cuda"))
    assert logits.shape == (3, 16, 64)
    assert model.parameter_count() == sum(parameter.numel() for parameter in model.parameters())


def test_custom_torch_flash_attention_supports_non_power_of_two_head_width() -> None:
    queries = torch.randn(1, 2, 4, 96, device="cuda", requires_grad=True)
    keys = torch.randn(1, 2, 4, 96, device="cuda", requires_grad=True)
    values = torch.randn(1, 2, 4, 96, device="cuda", requires_grad=True)
    output = flash_attention_pytorch(queries, keys, values, True)
    output.sum().backward()
    assert output.shape == queries.shape
    assert queries.grad is not None


def test_run_logger_replaces_old_metrics_unless_resuming(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text('{"step": 99}\n', encoding="utf-8")
    logger = RunLogger(tmp_path)
    logger.log(1, "train", {"loss": 1.0})
    assert '"step": 99' not in metrics_path.read_text(encoding="utf-8")
    resumed_logger = RunLogger(tmp_path, append_existing=True)
    resumed_logger.log(2, "train", {"loss": 0.5})
    assert len(metrics_path.read_text(encoding="utf-8").splitlines()) == 2


def test_jsonl_inspector_reports_counts_and_examples(tmp_path, capsys) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text('{"text": "first", "source": "test"}\n\nnot-json\n{"text": "second"}\n', encoding="utf-8")
    inspect_jsonl(dataset_path, num_examples=1, pretty=False, detailed=True)
    output = capsys.readouterr().out
    assert "Physical JSONL records: 4" in output
    assert "Valid JSON objects: 2" in output
    assert "Empty lines: 1" in output
    assert "Invalid or non-object JSON lines: 1" in output
    assert "First 1 valid record(s)" in output
    assert "first" in output
    inspect_jsonl(dataset_path, num_examples=0, pretty=False)
    assert "First 0 valid record(s)" in capsys.readouterr().out


def test_default_dense_config_is_triton_compatible() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "model" / "dense.yaml"
    config = load_model_config(config_path, vocab_size=6400)
    assert (config.d_model, config.num_heads, config.num_kv_heads) == (768, 12, 4)
    assert config.d_model // config.num_heads == 64
    assert (config.d_model // config.num_heads) & ((config.d_model // config.num_heads) - 1) == 0


def test_minimind_tokenizer_has_the_expected_pretraining_contract() -> None:
    tokenizer = MiniMindTokenizer.from_pretrained(TOKENIZER_PATH)
    assert tokenizer.vocab_size == 6400
    assert (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id) == (0, 1, 2)
    text = "你好, MiniMind!"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_moe_routes_and_produces_auxiliary_loss() -> None:
    model = TransformerLM(_config("moe")).cuda().train()
    logits = model(torch.randint(0, 64, (2, 16), device="cuda"))
    auxiliary_loss = model.auxiliary_loss()
    assert logits.shape == (2, 16, 64)
    assert auxiliary_loss.ndim == 0
    assert torch.isfinite(auxiliary_loss)


def test_grouped_moe_top1_matches_per_expert_swiglu_equations() -> None:
    """Variable-M experts must preserve each independent SwiGLU equation."""
    torch.manual_seed(1)
    model = TransformerLM(_config("moe")).cuda().train()
    experts = model.layers[0].feed_forward.grouped_experts
    assert experts is not None
    inputs = torch.randn(7, 64, device="cuda", requires_grad=True)
    offsets = torch.tensor([0, 2, 7], device="cuda")
    from ajllm.modeling.cuda_kernels import build_variable_m_schedule

    tile_experts, tile_rows = build_variable_m_schedule(offsets)
    outputs = experts(inputs, offsets, tile_experts, tile_rows)
    reference_inputs = inputs.detach().clone().requires_grad_()
    reference_rows = []
    for expert_index in range(2):
        start, end = offsets[expert_index], offsets[expert_index + 1]
        gate_up = reference_inputs[start:end] @ experts.gate_up_proj.weight[expert_index].transpose(0, 1)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = gate * torch.sigmoid(gate) * up
        reference_rows.append(hidden @ experts.down_proj.weight[expert_index].transpose(0, 1))
    reference = torch.cat(reference_rows)
    gradient = torch.randn_like(outputs)
    outputs.backward(gradient)
    input_gradient = inputs.grad.clone()
    reference.backward(gradient)
    torch.testing.assert_close(outputs, reference, rtol=5e-3, atol=4e-5)
    torch.testing.assert_close(input_gradient, reference_inputs.grad, rtol=5e-3, atol=2e-5)


def test_fsdp_wrapper_keeps_grouped_moe_experts_usable() -> None:
    """VariableGroupedLinear retains the custom FSDP gather/reduce-scatter contract."""
    model = FullyShardedDataParallel(TransformerLM(_config("moe")).cuda(), activation_checkpointing=True)
    logits = model(torch.randint(0, 64, (2, 16), device="cuda"))
    loss = cross_entropy(logits, torch.randint(0, 64, (2, 16), device="cuda")) + model.module.auxiliary_loss()
    loss.backward()
    model.finish_gradient_synchronization()
    assert model.module.layers[0].feed_forward.grouped_experts.gate_up_proj.weight_shard.grad is not None


def test_single_training_step_updates_parameters() -> None:
    torch.manual_seed(0)
    model = TransformerLM(_config()).cuda().train()
    optimizer = AdamW(model.parameters(), learning_rate=1e-3)
    input_ids = torch.randint(0, 64, (2, 16), device="cuda")
    labels = torch.randint(0, 64, (2, 16), device="cuda")
    before = model.token_embeddings.weight.detach().clone()
    loss = cross_entropy(model(input_ids), labels)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, model.token_embeddings.weight)


def test_fsdp_wrapper_keeps_tied_embedding_usable() -> None:
    model = FullyShardedDataParallel(TransformerLM(_config()).cuda(), activation_checkpointing=True)
    logits = model(torch.randint(0, 64, (2, 16), device="cuda"))
    loss = cross_entropy(logits, torch.randint(0, 64, (2, 16), device="cuda"))
    loss.backward()
    model.finish_gradient_synchronization()
    full_state = model.full_state_dict()
    assert logits.shape == (2, 16, 64)
    assert "token_embeddings.weight" in full_state
    assert "layers.0.attention.q_proj.weight" in full_state


def test_fsdp_checkpoint_round_trip(tmp_path) -> None:
    model = FullyShardedDataParallel(TransformerLM(_config()).cuda(), activation_checkpointing=False)
    optimizer = AdamW(model.parameters(), learning_rate=1e-3)
    input_ids = torch.randint(0, 64, (2, 16), device="cuda")
    loss = cross_entropy(model(input_ids), torch.randint(0, 64, (2, 16), device="cuda"))
    loss.backward()
    model.finish_gradient_synchronization()
    optimizer.step()
    path = save_checkpoint(tmp_path / "checkpoint.pt", model, optimizer, 3, {"test": True})
    expected_logits = model(input_ids).detach()
    restored = FullyShardedDataParallel(TransformerLM(_config()).cuda(), activation_checkpointing=False)
    restored_optimizer = AdamW(restored.parameters(), learning_rate=1e-3)
    assert load_checkpoint(path, restored, restored_optimizer)["step"] == 3
    assert torch.allclose(expected_logits, restored(input_ids))


def test_cuda_pretraining_workflow_smoke_test(tmp_path) -> None:
    data_path = tmp_path / "data.jsonl"
    data_path.write_text('{"text": "hello world"}\n', encoding="utf-8")
    model_path = tmp_path / "model.yaml"
    model_path.write_text(
        yaml.safe_dump(
            {
                "model_type": "dense",
                "context_length": 16,
                "max_position_embeddings": 32,
                "d_model": 64,
                "num_layers": 1,
                "num_heads": 4,
                "num_kv_heads": 2,
                "d_ff": 128,
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model_config": str(model_path),
                "data_path": str(data_path),
                "validation_data_path": str(data_path),
                "tokenizer": {"path": str(TOKENIZER_PATH)},
                "max_seq_len": 16,
                "batch_size": 1,
                "epochs": 3,
                "max_steps": 2,
                "learning_rate": 1e-3,
                "min_lr": 1e-4,
                "output_dir": str(output_dir),
                "device": "cuda",
                "mixed_precision": None,
                "log_interval": 1,
                "eval_interval": 1,
                "eval_batches": 1,
                "save_interval": 1,
            }
        ),
        encoding="utf-8",
    )
    summary = run(config_path)
    assert (summary["epochs"], summary["steps"], summary["stopped_early"]) == (3, 2, True)
    checkpoint_path = output_dir / "step_00000002.pt"
    assert checkpoint_path.is_file()
    assert '"event": "evaluation"' in (output_dir / "metrics.jsonl").read_text(encoding="utf-8")
    evaluation = evaluate_checkpoint(checkpoint_path, data_path, TOKENIZER_PATH, batch_size=1, device="cuda")
    assert evaluation["evaluated_tokens"] > 0
    figure_path = plot_metric([output_dir], tmp_path / "loss.png")
    assert figure_path.is_file()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    generated_model = TransformerLM(ModelConfig(**checkpoint["metadata"]["model_config"])).cuda()
    generated_model.load_state_dict(checkpoint["model_state_dict"])
    tokenizer = MiniMindTokenizer.from_pretrained(TOKENIZER_PATH)
    assert isinstance(generate(generated_model, tokenizer, "hello", 1, 0.0, 0, 1.0, torch.device("cuda")), str)
