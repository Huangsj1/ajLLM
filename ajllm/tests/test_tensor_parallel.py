"""CPU/Gloo correctness tests for the dense tensor-parallel wrapper."""

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
from ajllm.training.parallel.tensor_parallel import TensorParallel, tensor_parallelize


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
        use_cuda_kernels=False,
        use_flash_attention=False,
    )


def _broadcast_module(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for tensor in list(module.parameters()) + list(module.buffers()):
            dist.broadcast(tensor, src=0)


def _worker(rank: int, init_file: str, result_file: str, checkpoint_file: str) -> None:
    """Compare each local TP result to a full, rank-zero reference model."""
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        # Avoid compiling the pedagogical flash backward in each spawned CPU worker.
        import ajllm.modeling.flash_attention as flash_module

        flash_module._compiled_backward = flash_module.flash_backward_pytorch
        torch.manual_seed(100 + rank)  # The wrapper must correct differing initial states.
        parallel_model = tensor_parallelize(TransformerLM(_config()))
        reference = TransformerLM(_config())
        full_initial_state = parallel_model.full_state_dict()
        if rank == 0:
            reference.load_state_dict(full_initial_state)
        _broadcast_module(reference)

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
        targets = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 1], [7, 6, 5, 4, 3, 2, 1, 8]])
        reference_logits = reference(input_ids)
        reference_loss = torch.nn.functional.cross_entropy(reference_logits.reshape(-1, 32), targets.reshape(-1))
        reference_loss.backward()

        parallel_logits = parallel_model(input_ids)
        parallel_loss = torch.nn.functional.cross_entropy(parallel_logits.reshape(-1, 32), targets.reshape(-1))
        parallel_loss.backward()
        torch.testing.assert_close(parallel_logits, reference_logits, rtol=1e-4, atol=1e-5)

        reference_parameters = dict(reference.named_parameters())
        parallel_parameters = dict(parallel_model.module.named_parameters())
        for name, parameter in parallel_parameters.items():
            assert parameter.grad is not None, name
            layout = parallel_model._layouts.get(name)
            expected = reference_parameters[name].grad
            if layout is not None:
                shard_size = layout.full_shape[layout.axis] // 2
                expected = expected.narrow(layout.axis, rank * shard_size, shard_size)
            torch.testing.assert_close(parameter.grad, expected, rtol=3e-4, atol=3e-5)

        reference_optimizer = AdamW(reference.parameters(), learning_rate=1e-3, use_cuda_kernels=False)
        parallel_optimizer = AdamW(parallel_model.parameters(), learning_rate=1e-3, use_cuda_kernels=False)
        reference_optimizer.step()
        parallel_optimizer.step()
        for name, value in parallel_model.full_state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[name], rtol=3e-4, atol=3e-5)

        save_checkpoint(checkpoint_file, parallel_model, parallel_optimizer, 7, {"test": "tensor_parallel"})
        restored = tensor_parallelize(TransformerLM(_config()))
        restored_optimizer = AdamW(restored.parameters(), learning_rate=1e-3, use_cuda_kernels=False)
        checkpoint = load_checkpoint(checkpoint_file, restored, restored_optimizer)
        assert checkpoint["step"] == 7
        torch.testing.assert_close(restored(input_ids), parallel_model(input_ids), rtol=3e-4, atol=3e-5)
        if rank == 0:
            torch.save({"loss": float(parallel_loss), "parameter_count": parallel_model.parameter_count()}, result_file)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required for CPU tensor-parallel tests")
def test_tensor_parallel_matches_dense_reference_on_two_cpu_ranks(tmp_path: Path) -> None:
    """Cover column/row linear, GQA local heads, Q/K norms, and one AdamW update."""
    init_file = tmp_path / "tp_init"
    result_file = tmp_path / "result.pt"
    checkpoint_file = tmp_path / "tensor_parallel.pt"
    mp.spawn(_worker, args=(str(init_file), str(result_file), str(checkpoint_file)), nprocs=2, join=True)
    result = torch.load(result_file, weights_only=True)
    assert result["loss"] > 0
    assert result["parameter_count"] == TransformerLM(_config()).parameter_count()


def test_tensor_parallel_rejects_moe_until_ep_adapter_exists() -> None:
    config = _config().__class__(**(_config().__dict__ | {"model_type": "moe", "num_experts": 2}))
    with pytest.raises(ValueError, match="dense models only"):
        TensorParallel(TransformerLM(config))
