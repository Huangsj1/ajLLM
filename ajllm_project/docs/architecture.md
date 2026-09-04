# ajLLM pre-training architecture

## Scope

The project has one compact decoder-only model for pre-training. It includes the
project's own Torch/Triton FlashAttention and a simple generation workflow, but
deliberately excludes a generation KV cache, tensor/pipeline parallelism, and
post-training. `modeling/transformer.py` is the only assembly point:
`ModelConfig` is the contract and `TransformerLM` is the model.

| Item | Value |
| --- | ---: |
| vocabulary | 6,400, read from the copied MiniMind tokenizer JSON |
| layers / hidden size | 8 / 768 |
| query / KV heads | 12 / 4 (GQA ratio 3:1) |
| head size | 64 |
| SwiGLU width | 2,432 = `ceil(768 × pi / 64) × 64` |
| training context / RoPE table | 512 / 32,768 |
| RoPE theta / dropout | 1,000,000 / 0 |

The dense model has 62.34M parameters with tied input/output embeddings. The supplied four-expert Top-1 MoE has 196.84M total parameters, with one expert active per token. It raises capacity; it does not save memory.


## Forward path

```text
input IDs [B, T]
  -> token embedding [vocab, d_model]
  -> N × (RMSNorm -> GQA -> residual; RMSNorm -> FFN -> residual)
  -> final RMSNorm
  -> tied embedding transpose (or independent LM head)
  -> logits [B, T, vocab]
```

The pre-training dataset forms `input_ids = tokens[:-1]` and `labels = tokens[1:]`. Label `-100` is ignored by cross entropy. Each JSONL document has BOS/EOS and is padded after EOS. The dataset indexes byte offsets rather than retaining the complete corpus in host memory.

Pre-training has an outer epoch loop. Each epoch iterates every batch from its
current-rank dataloader once; it does not restart the iterator to satisfy a
configured step count. The total number of AdamW updates is
`epochs × ceil(batches_per_epoch / gradient_accumulation_steps)`, which is also
the length used by the warmup/cosine schedule. A final partial accumulation group
at the end of an epoch is rescaled by its actual micro-batch count before update.
An optional `max_steps` reduces that planned total for smoke tests; reaching it
uses the same final metric, evaluation, checkpoint, and summary path as a normal
epoch-complete finish.

## Transformer block

Every block uses pre-norm residuals:

```text
h = h + Attention(RMSNorm(h))
h = h + FFN(RMSNorm(h))
```

RMSNorm calculates in fp32, casts back to the input dtype, uses epsilon `1e-6`, and has only a learned scale.

### GQA, Q/K norm, and RoPE

All projections are bias-free:

```text
Q: 768 -> 12 × 64      K/V: 768 -> 4 × 64       O: 768 -> 768
```

Q and K are RMS-normalized per head, then receive RoPE with theta `1e6`. K/V are repeated from four to twelve heads for causal attention.

`attention.py` first calls this project's `flash_attention.py`: its tiled Torch
implementation on CPU/CUDA, or its Triton kernel on CUDA when supported. The
Triton block-pointer kernel requires a power-of-two head width; the default
64-wide heads meet that requirement, so CUDA training uses the project's Triton
kernel. If a custom backend cannot run, it is disabled for that attention layer and falls back to PyTorch
`scaled_dot_product_attention`, which can itself choose PyTorch's
FlashAttention-2 backend. The project implementation is therefore the default.

Twelve query heads retain the 768-wide hidden representation while providing
finer attention subspaces than the former eight-head layout. Four KV heads keep
the KV cache and K/V projection width at one third of the query-head count.
The resulting 64-wide head is both a common compact-model width and compatible
with the project's current Triton block-pointer kernel. This is a deliberate
small-model trade-off: it adds no new mechanism and preserves the same depth,
SwiGLU width, and 512-token training context.

### SwiGLU and MoE

Dense FFNs use the canonical `activations.py` implementation:

```text
FFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
```

MoE reuses this exact SwiGLU implementation. A bias-free router selects Top-K experts, batches selected tokens per expert, then `index_add_` restores weighted results. Training adds load balance loss:

```text
num_experts × coefficient × sum(mean_router_probability × expert_load)
```

Dispatch loops over experts rather than individual tokens. It is readable, not a fused MoE kernel; more experts can lower throughput.

## Parameter accounting

```text
default dense attention/block     1,572,992
default dense SwiGLU/block        5,603,328
block norms/block                     1,536
8 blocks                          57,422,848
tied embedding                    4,915,200
final RMSNorm                           768
total                            62,338,816
```

MoE adds three extra 5,603,328-parameter experts per block plus routers, for
196,843,264 stored parameters (about 62.36M active parameters per token,
including the router). Top-1 maintains roughly dense FFN compute, but all
expert weights reside in memory.

## Code map and extension boundaries

| Location | Responsibility |
| --- | --- |
| `modeling/layers.py` | bias-free Linear/Embedding and RMSNorm |
| `modeling/attention.py` | RoPE, GQA and SDPA |
| `modeling/activations.py` | canonical SwiGLU |
| `modeling/moe.py` | router and expert dispatch |
| `modeling/transformer.py` | config, block and causal LM |
| `modeling/factory.py` | YAML-to-model construction |

Generation/KV cache should be a future inference module. RoPE scaling needs an explicit config and training evidence. Fused kernels, expert parallelism and tensor/pipeline parallelism should be optional layers over this baseline.
