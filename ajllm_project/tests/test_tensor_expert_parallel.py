"""Four-rank CPU/Gloo reference test for the unified TP×EP MoE wrapper."""

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
from ajllm.training.parallel.common import ParallelContext
from ajllm.training.parallel.tensor_expert_parallel import tensor_expert_parallelize


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


def _average_reference_over_ep(reference: torch.nn.Module, context: ParallelContext) -> None:
    for parameter in reference.parameters():
        assert parameter.grad is not None
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=context.ep_group)
        parameter.grad.div_(context.ep_size)


def _expert_gradient_slice(
    full_gradient: torch.Tensor,
    adapter,
    kind: str,
) -> torch.Tensor:
    expert_slice = slice(adapter.expert_start, adapter.expert_start + adapter.local_expert_count)
    if kind == "gate_up":
        local_gate_up = full_gradient[expert_slice]
        return torch.cat(
            (
                local_gate_up[:, adapter.ff_start : adapter.ff_start + adapter.local_d_ff],
                local_gate_up[
                    :,
                    adapter.global_d_ff
                    + adapter.ff_start : adapter.global_d_ff
                    + adapter.ff_start
                    + adapter.local_d_ff,
                ],
            ),
            dim=1,
        )
    return full_gradient[expert_slice, :, adapter.ff_start : adapter.ff_start + adapter.local_d_ff]


def _worker(rank: int, init_file: str, result_file: str, checkpoint_file: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=4)
    try:
        import ajllm.modeling.flash_attention as flash_module

        flash_module._compiled_backward = flash_module.flash_backward_pytorch
        context = ParallelContext.from_distributed(tp_size=2, ep_size=2)
        torch.manual_seed(500 + rank)
        parallel_model = tensor_expert_parallelize(TransformerLM(_config()), context, expert_backend="torch")
        reference = TransformerLM(_config())
        initial_state = parallel_model.full_state_dict()
        if rank == 0:
            reference.load_state_dict(initial_state)
        _broadcast_module(reference)

        input_ids = (
            torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
            if context.ep_rank == 0
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
        torch.testing.assert_close(parallel_logits, reference_logits, rtol=4e-4, atol=4e-5)

        _average_reference_over_ep(reference, context)
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in parallel_model.module.named_parameters():
            assert parameter.grad is not None, name
            expected = reference_parameters[name].grad
            if name in parallel_model._tp_layouts:
                axis, full_shape = parallel_model._tp_layouts[name]
                shard_size = full_shape[axis] // context.tp_size
                expected = expected.narrow(axis, context.tp_rank * shard_size, shard_size)
            elif name in parallel_model._expert_layouts:
                adapter, kind, _ = parallel_model._expert_layouts[name]
                expected = _expert_gradient_slice(expected, adapter, kind)
            torch.testing.assert_close(parameter.grad, expected, rtol=7e-4, atol=7e-5)

        reference_optimizer = AdamW(reference.parameters(), learning_rate=1e-3, use_cuda_kernels=False)
        parallel_optimizer = AdamW(parallel_model.parameters(), learning_rate=1e-3, use_cuda_kernels=False)
        reference_optimizer.step()
        parallel_optimizer.step()
        for name, value in parallel_model.full_state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[name], rtol=7e-4, atol=7e-5)
        save_checkpoint(checkpoint_file, parallel_model, parallel_optimizer, 3, {"test": "tensor_expert_parallel"})
        restored = tensor_expert_parallelize(TransformerLM(_config()), context, expert_backend="torch")
        restored_optimizer = AdamW(restored.parameters(), learning_rate=1e-3, use_cuda_kernels=False)
        checkpoint = load_checkpoint(checkpoint_file, restored, restored_optimizer)
        assert checkpoint["step"] == 3
        torch.testing.assert_close(restored(input_ids), parallel_model(input_ids), rtol=7e-4, atol=7e-5)
        if rank == 0:
            torch.save({"loss": float(parallel_loss)}, result_file)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required for CPU TP×EP tests")
def test_tensor_expert_parallel_matches_four_rank_reference(tmp_path: Path) -> None:
    init_file = tmp_path / "tp_ep_init"
    result_file = tmp_path / "result.pt"
    checkpoint_file = tmp_path / "tensor_expert_parallel.pt"
    mp.spawn(_worker, args=(str(init_file), str(result_file), str(checkpoint_file)), nprocs=4, join=True)
    assert torch.load(result_file, weights_only=True)["loss"] > 0
