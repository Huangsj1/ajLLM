"""CPU/Gloo numerical comparison for the Top-1 expert-parallel wrapper."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ajllm.modeling import ModelConfig, TransformerLM
from ajllm.training.checkpoint import load_checkpoint, save_checkpoint
from ajllm.training.optimizers import AdamW
from ajllm.training.parallel.expert_parallel import _AllToAll, expert_parallelize


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        context_length=8,
        max_position_embeddings=16,
        d_model=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        d_ff=64,
        model_type="moe",
        num_experts=2,
        num_experts_per_token=1,
        use_cuda_kernels=False,
        use_flash_attention=False,
    )


def _broadcast_module(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for tensor in list(module.parameters()) + list(module.buffers()):
            dist.broadcast(tensor, src=0)


def _average_reference_gradients(reference: torch.nn.Module) -> None:
    for parameter in reference.parameters():
        assert parameter.grad is not None
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(2)


def _worker(rank: int, init_file: str, result_file: str, checkpoint_file: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        import ajllm.modeling.flash_attention as flash_module

        flash_module._compiled_backward = flash_module.flash_backward_pytorch
        torch.manual_seed(200 + rank)  # Verify rank-zero synchronization before expert slicing.
        parallel_model = expert_parallelize(TransformerLM(_config()), expert_backend="torch")
        reference = TransformerLM(_config())
        full_initial_state = parallel_model.full_state_dict()
        if rank == 0:
            reference.load_state_dict(full_initial_state)
        _broadcast_module(reference)

        input_ids = (
            torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
            if rank == 0
            else torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6], [2, 7, 1, 8, 2, 8, 1, 8]])
        )
        targets = torch.roll(input_ids, shifts=-1, dims=1)
        reference_logits = reference(input_ids)
        reference_loss = torch.nn.functional.cross_entropy(reference_logits.reshape(-1, 32), targets.reshape(-1))
        reference_loss = reference_loss + reference.auxiliary_loss()
        reference_loss.backward()

        parallel_logits = parallel_model(input_ids)
        parallel_loss = torch.nn.functional.cross_entropy(parallel_logits.reshape(-1, 32), targets.reshape(-1))
        parallel_loss = parallel_loss + parallel_model.auxiliary_loss()
        parallel_loss.backward()
        parallel_model.finish_gradient_synchronization()
        torch.testing.assert_close(parallel_logits, reference_logits, rtol=2e-4, atol=2e-5)

        _average_reference_gradients(reference)
        reference_parameters = dict(reference.named_parameters())
        parallel_parameters = dict(parallel_model.module.named_parameters())
        for name, parameter in parallel_parameters.items():
            assert parameter.grad is not None, name
            expected = reference_parameters[name].grad
            layout = parallel_model._layouts.get(name)
            if layout is not None:
                shard_size = layout.full_shape[0] // 2
                expected = expected.narrow(0, rank * shard_size, shard_size)
            torch.testing.assert_close(parameter.grad, expected, rtol=4e-4, atol=4e-5)

        reference_optimizer = AdamW(reference.parameters(), learning_rate=1e-3, use_cuda_kernels=False)
        parallel_optimizer = AdamW(parallel_model.parameters(), learning_rate=1e-3, use_cuda_kernels=False)
        reference_optimizer.step()
        parallel_optimizer.step()
        for name, value in parallel_model.full_state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[name], rtol=4e-4, atol=4e-5)

        save_checkpoint(checkpoint_file, parallel_model, parallel_optimizer, 11, {"test": "expert_parallel"})
        restored = expert_parallelize(TransformerLM(_config()), expert_backend="torch")
        restored_optimizer = AdamW(restored.parameters(), learning_rate=1e-3, use_cuda_kernels=False)
        checkpoint = load_checkpoint(checkpoint_file, restored, restored_optimizer)
        assert checkpoint["step"] == 11
        torch.testing.assert_close(restored(input_ids), parallel_model(input_ids), rtol=4e-4, atol=4e-5)

        # A rank may own no routed rows, and a source/destination split may be zero.
        empty_case_inputs = (
            torch.tensor([[1.0], [2.0]], requires_grad=True)
            if rank == 0
            else torch.empty((0, 1), requires_grad=True)
        )
        input_splits = (1, 1) if rank == 0 else (0, 0)
        output_splits = (1, 0)
        empty_case_outputs = _AllToAll.apply(empty_case_inputs, input_splits, output_splits, None)
        torch.testing.assert_close(empty_case_outputs, torch.tensor([[float(rank + 1)]]))
        empty_case_outputs.sum().backward()
        torch.testing.assert_close(empty_case_inputs.grad, torch.ones_like(empty_case_inputs))
        if rank == 0:
            torch.save({"loss": float(parallel_loss), "parameter_count": parallel_model.parameter_count()}, result_file)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required for CPU expert-parallel tests")
def test_expert_parallel_matches_two_source_reference_on_two_cpu_ranks(tmp_path: Path) -> None:
    """Exercise variable routes, two all-to-alls, backward, gradient scale, and AdamW."""
    init_file = tmp_path / "ep_init"
    result_file = tmp_path / "result.pt"
    checkpoint_file = tmp_path / "expert_parallel.pt"
    mp.spawn(_worker, args=(str(init_file), str(result_file), str(checkpoint_file)), nprocs=2, join=True)
    result = torch.load(result_file, weights_only=True)
    assert result["loss"] > 0
    assert result["parameter_count"] == TransformerLM(_config()).parameter_count()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the Triton expert adapter check")
def test_single_rank_expert_parallel_triton_matches_grouped_moe() -> None:
    """Keep the existing local Variable-M Triton expert path usable through the EP adapter."""
    torch.manual_seed(17)
    config = ModelConfig(
        vocab_size=64,
        context_length=16,
        max_position_embeddings=32,
        d_model=64,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        d_ff=128,
        model_type="moe",
        num_experts=2,
        num_experts_per_token=1,
        use_cuda_kernels=True,
        use_flash_attention=True,
    )
    reference = TransformerLM(config).cuda().train()
    wrapped_source = TransformerLM(config).cuda().train()
    wrapped_source.load_state_dict(reference.state_dict())
    parallel_model = expert_parallelize(wrapped_source, expert_backend="triton")
    input_ids = torch.randint(0, config.vocab_size, (2, 16), device="cuda")
    targets = torch.randint(0, config.vocab_size, (2, 16), device="cuda")
    reference_loss = torch.nn.functional.cross_entropy(
        reference(input_ids).reshape(-1, config.vocab_size), targets.reshape(-1)
    )
    reference_loss = reference_loss + reference.auxiliary_loss()
    reference_loss.backward()
    parallel_loss = torch.nn.functional.cross_entropy(
        parallel_model(input_ids).reshape(-1, config.vocab_size), targets.reshape(-1)
    )
    parallel_loss = parallel_loss + parallel_model.auxiliary_loss()
    parallel_loss.backward()
    parallel_model.finish_gradient_synchronization()
    torch.testing.assert_close(parallel_loss, reference_loss, rtol=8e-3, atol=8e-4)
    for name, parameter in parallel_model.module.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad, rtol=1e-2, atol=1e-3)
