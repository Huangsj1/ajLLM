# Parallel training

The canonical `TransformerLM` is unchanged. Parallel execution wraps an already-built model, so the training contract remains `model(input_ids)`, loss, backward, and AdamW. Create the optimizer after wrapping; the pretraining workflow does this automatically.

## Layout and public modules

All implementations live in `src/ajllm/training/parallel/`:

| File | Responsibility |
| --- | --- |
| [common.py](../src/ajllm/training/parallel/common.py) | Process-group helpers, differentiable TP collectives, model broadcast, full-tensor gathering, and `ParallelContext` |
| [fsdp.py](../src/ajllm/training/parallel/fsdp.py) | Existing educational FULL_SHARD wrapper |
| [tensor_parallel.py](../src/ajllm/training/parallel/tensor_parallel.py) | Dense-model TP adapters |
| [expert_parallel.py](../src/ajllm/training/parallel/expert_parallel.py) | Top-1 MoE EP adapter and dispatch |
| [tensor_expert_parallel.py](../src/ajllm/training/parallel/tensor_expert_parallel.py) | Unified MoE TP×EP wrapper |

The previous `training.distributed`, `training.tensor_parallel`, and `training.expert_parallel` modules were removed. Import all parallel wrappers from `ajllm.training.parallel` or from a file in that package.

| Strategy | Wrapper | Supported model | Partition |
| --- | --- | --- | --- |
| FSDP | `FullyShardedDataParallel` | Dense, MoE | Linear parameter/gradient/optimizer storage |
| TP | `TensorParallel` | Dense | Attention heads and SwiGLU intermediate channels |
| EP | `ExpertParallel` | Top-1 grouped MoE | Whole experts |
| TP×EP | `TensorExpertParallel` | Top-1 grouped MoE | Attention heads, expert ownership, and each local expert's intermediate channels |

## Configuration and launch

`WORLD_SIZE` must equal `parallel.tp_size * parallel.ep_size`. There is no separate DP dimension in this implementation. TP ranks in one TP group load the same batch; EP groups load different batches.

```yaml
# Four-GPU dense TP
parallel:
  tp_size: 4
  ep_size: 1

# Four-GPU Top-1 MoE EP
parallel:
  tp_size: 1
  ep_size: 4
  expert_backend: torch

# Four-GPU Top-1 MoE TP×EP
parallel:
  tp_size: 2
  ep_size: 2
  expert_backend: torch
```

```bash
# if not support IB, disable it to avoid SIGSEGV
NCCL_IB_DISABLE=1 \
uv run torchrun --standalone --nproc_per_node=4 \
  -m ajllm.workflows.pretrain --config configs/pretrain_dense_tp4.yaml

NCCL_IB_DISABLE=1 \
uv run torchrun --standalone --nproc_per_node=4 \
  -m ajllm.workflows.pretrain --config configs/pretrain_moe_ep4.yaml

NCCL_IB_DISABLE=1 \
uv run torchrun --standalone --nproc_per_node=4 \
  -m ajllm.workflows.pretrain --config configs/pretrain_moe_tp2_ep2.yaml
```

For FSDP, set `use_fsdp: true` and leave both parallel sizes at one. FSDP×TP and FSDP×EP are rejected. For TP, both `num_heads` and `num_kv_heads` must divide by `tp_size`; for MoE TP×EP, `d_ff` must also divide by `tp_size` and `num_experts` by `ep_size`.

## Tensor parallelism

`TensorParallel` converts a dense transformer's custom `Linear` layers into column- and row-parallel adapters:

```text
attention: Q/K/V column shard by head -> local attention -> output row shard -> SUM
SwiGLU:    gate/up column shard        -> local gate      -> down row shard   -> SUM
```

Column-parallel inputs are copied in forward and their input gradients are summed over the TP group in backward. Row-parallel outputs are summed in forward and their output gradients are copied in backward. GQA preserves the Q:KV group ratio on each rank. Q/K RMSNorm gains remain replicated and use a TP gradient sum; embeddings, final RMSNorm, and vocabulary logits remain replicated.

TP does not increase the effective batch size:

```text
batch_size × gradient_accumulation_steps
```

## Expert parallelism

`ExpertParallel` wraps the existing `MoELayer`. For `N` experts and `P = ep_size`, EP rank `r` owns contiguous IDs `[r * N/P, (r + 1) * N/P)`. Router, attention, norms, embedding, and logits are replicated.

Only the grouped `num_experts_per_token: 1` path is supported. The unchanged router selects a Top-1 expert, packs token rows by owner, exchanges variable row counts with differentiable `all_to_all_single`, evaluates the owner-local experts, then sends rows back to their source rank. The all-to-all backward uses the inverse exchange.

EP ranks process distinct batch shards, so effective batch size is:

```text
batch_size × gradient_accumulation_steps × ep_size
```

Replicated gradients are averaged over the EP group. An expert's local gradient already includes routed tokens from every EP source rank, so it is divided by `ep_size` without an all-reduce across different experts.

`expert_backend: torch` is the reference backend and is covered by CPU/Gloo tests. `expert_backend: triton` keeps the existing Variable-M grouped-expert Triton computation after dispatch. It needs CUDA/NCCL validation before a production run; no original Triton MoE implementation is removed.

## Unified TP×EP for MoE

Use `tp_size > 1` and `ep_size > 1` with an MoE model. The workflow constructs one `ParallelContext` and one `TensorExpertParallel` wrapper. Its rank mapping is `global_rank = ep_rank * tp_size + tp_rank`; a TP group fixes `ep_rank`, and an EP group fixes `tp_rank`.

For code outside the pretraining workflow, construct the same single wrapper:

```python
from ajllm.training.parallel import ParallelContext, tensor_expert_parallelize

context = ParallelContext.from_distributed(tp_size=2, ep_size=2)
model = tensor_expert_parallelize(model, context, expert_backend="torch")
```

Attention becomes `TensorParallelAttention` in every EP replica. Each `feed_forward` becomes `ExpertParallelMoE` exactly once: EP selects its expert range and TP slices each selected expert's gate/up and down intermediate channels. The TP partial expert output is summed before it is returned through EP. This preserves the original MoE architecture and lets the Torch or Triton local expert backend run without changing its public model interface.

Do not manually nest `TensorParallel(ExpertParallel(model))` or the reverse. Both independent wrappers own a `feed_forward` replacement and their process groups and checkpoint layouts would conflict. `TensorExpertParallel` is the single wrapper for that composition; standalone TP and standalone EP remain independent for their supported model types.

In TP×EP, only one rank per TP group contributes metrics, while every EP group contributes its distinct data shard. Effective batch size is therefore also:

```text
batch_size × gradient_accumulation_steps × ep_size
```

## FSDP

`FullyShardedDataParallel` is the existing educational FULL_SHARD wrapper. It stores flat local parameter shards, all-gathers a full layer weight for calculation, and reduce-scatters gradients. Its source moved to [fsdp.py](../src/ajllm/training/parallel/fsdp.py); `fsdp_config.use_activation_checkpointing` controls layer recomputation.

FSDP processes different data shards, so its effective batch size is:

```text
batch_size × gradient_accumulation_steps × WORLD_SIZE
```

## Checkpoints and validation

Every wrapper provides `full_state_dict()` and `load_full_state_dict()`. The main checkpoint uses unwrapped parameter names and full tensor shapes; every participating rank must enter the collective full-state export before rank zero writes the file. Parallel wrappers save rank-local optimizer files and require the same world size, wrapper strategy, and TP/EP topology when resuming.

### Resume after an interruption

Use the latest completed `step_<N>.pt` in `output_dir`; do not use a temporary
`.pt.tmp` file. A parallel checkpoint consists of the main checkpoint and one
optimizer file per rank, so keep the whole set together:

```text
step_00005000.pt             model state, step, data position, and topology
step_00005000.pt.rank0.optim optimizer state for rank 0
step_00005000.pt.rank1.optim optimizer state for rank 1
...
```

Single-process training instead writes `step_00005000.pt.optim`. The checkpoint
step is an already completed optimizer update. Resuming restores the model,
optimizer, epoch, and batch position, then continues at the next update.
The training sampler applies that batch position directly to its index stream,
so it does not read or tokenize the already completed batches during recovery.

Edit the original YAML and set `resume_from` to the main `.pt` file. Keep the
model, tokenizer, data path, sequence length, batch size, accumulation steps,
seed, precision, and parallel settings unchanged. For example, to resume a
four-GPU TP×EP run from step 5,000:

```yaml
# configs/pretrain_moe_tp2_ep2.yaml
max_steps: 10000
resume_from: output/pretrain/moe_tp2_ep2/step_00005000.pt
```

Launch with the same command and process count as the original run:

```bash
uv run torchrun --standalone --nproc_per_node=4 \
  -m ajllm.workflows.pretrain --config configs/pretrain_moe_tp2_ep2.yaml
```

For FSDP, TP, EP, and TP×EP, every rank must start the resume command and each
rank's `.rank<RANK>.optim` file must be present. FSDP requires the same world
size. TP, EP, and TP×EP additionally check the saved wrapper type and TP/EP
topology, so a checkpoint from `tp_size: 2, ep_size: 2` cannot be resumed as
TP=4 or EP=4.

`max_steps` and `epochs` define the total target, not additional work. They
must leave a target larger than the saved step; otherwise the trainer reports
that the checkpoint is already complete. Retaining their original values keeps
the intended learning-rate schedule. Increasing them extends training, but the
remaining warmup/cosine schedule is recalculated against the new total.

The current checkpoint format does not save RNG state. The epoch-seeded training
sampler restores the same data order for checkpoints created with this version
when its settings remain unchanged, but runs with stochastic layers or external
randomness are not bitwise identical to the interrupted process. Checkpoints
created by the older batch-discard recovery path do not contain its single-GPU
shuffle order; they can restore model and optimizer state, but cannot reproduce
that earlier data order exactly.

CPU/Gloo reference tests validate TP, EP, and TP×EP against an unwrapped model:

```bash
uv run pytest tests/test_tensor_parallel.py tests/test_expert_parallel.py \
  tests/test_tensor_expert_parallel.py
```

The TP×EP test uses four ranks with `tp_size=2, ep_size=2`, distinct EP source batches, and checks forward logits, gradients, and one AdamW update. These tests validate communication algebra on this one-GPU development machine; they do not validate NCCL topology, multi-GPU memory use, or Triton communication performance. On a 4- or 8-GPU server, first run the reference suite with `expert_backend: torch`, then a short FP32/BF16 training job, and finally test `expert_backend: triton`.

Current limits: vocabulary-parallel loss, EP Top-K/capacity policy, FSDP×TP/EP, sequence parallelism, pipeline parallelism, and communication/computation overlap are not implemented. [parallel_training_tp_ep.md](parallel_training_tp_ep.md) keeps the earlier design exploration; this document describes the runnable interfaces.
