# Native Qwen2.5 execution

## Implemented model

The CUDA baseline reads local Qwen2.5 dense config and safetensors, implements
RMSNorm, RoPE, GQA, SwiGLU and the causal LM head, and supports contiguous reference KV tensors and a paged CUDA pool. It supports full prefill, repeated prefill chunks, one-token decode,
and mixed request batches. The local checkpoint is Qwen2.5-0.5B-Instruct: 24 layers,
896 hidden features, 14 query heads, two KV heads, and a 151936-entry vocabulary.
All dimensions come from configuration. Both generation-config EOS IDs are used.

The loader supports single and indexed sharded safetensors, validates tensor
names/shapes once at load time, and preserves tied embedding/head storage.
Parameters are constructed on meta and populated on CUDA. MoE, pre-quantized
checkpoint formats, scaled RoPE and sliding-window variants are rejected explicitly.
Optional runtime W8A16 conversion quantizes decoder projections after loading. Transformers
is used for tokenization and as an independent model oracle, never for native
execution. See the [official Qwen config](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json)
and [reference implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen2/modeling_qwen2.py).

## One forward per scheduler batch

`Qwen2Runner.execute` builds one `ModelBatch` from all scheduled requests and
invokes the model once. Prefill chunks and decode tokens share packed projections,
attention, and MLP operations. An empty schedule does not launch the model.

See [architecture](architecture/architecture.md) for scheduling and compute kernels.
The eager oracle pads mixed query/context lengths. The Triton backend reads paged
KV directly using ragged query tiles and split decode, without dense masks or
context gathers. The table below describes the retained eager reference.

| Data | Shape / meaning |
| --- | --- |
| Token IDs and positions | `[T]`, concatenated scheduled slices and absolute positions |
| Sequence IDs / query offsets | `[T]`, map packed tokens to batch rows and local query positions |
| Hidden states and MLP input | `[T, hidden_size]`, no padding in linear projections |
| Queries for attention | `[B, query_heads, Qmax, head_dim]` |
| Packed keys and values | `[B, kv_heads, Kmax, head_dim]` |
| Causal/padding mask | `[B, 1, Qmax, Kmax]`, constructed once per batch |
| Stored KV | Paged pool `[layers, 2, blocks, block_size, kv_heads, head_dim]`; contiguous reference also available |
| LM head output | `[sampling_ready_requests, vocab_size]` |

For each layer, Q/K/V are projected for all T tokens together. The paged backend scatters new K/V into physical slots and gathers padded
contexts from block tables; the contiguous reference packs cached tensors and
constructs replacement caches. Query heads sharing one KV head are folded into the query dimension
for batched QK/PV multiplications; K/V are no longer replicated with
`repeat_interleave`. Scores are reshaped back to query-head layout for causal
masking and softmax. The result is gathered back to T packed rows before the output
projection and MLP. Python loops only manage cache ownership and assemble metadata; they
do not evaluate a model or attention function per request.

For request r with cached length C and new query offset i, key j is visible when
`j <= C+i` and `j < context_length[r]`. Request rows are independent; padding never
becomes a stored token. Decode queries can be padded to the longest prefill chunk.
Query padding is discarded after attention. Tests compare mixed ragged batches
with independent full-context Transformers outputs and assert exactly one model
and projection call per scheduler step.

Partial prefill has no sampling row. Only requests that finish their available
input contribute a final hidden row to the LM head. The runner returns CUDA
tensors keyed by request ID. The sampler applies all policies on GPU; only selected
token IDs, log probabilities and validity flags return to the host. There is no
CPU sampling fallback or special greedy dispatch in the engine. See the
[sampling contract](architecture/architecture.md#sampling) for filtering and RNG rules.

## RoPE tables

The model creates cosine/sine tables for `max_position_embeddings` during
initialization. They are non-persistent buffers, excluded from checkpoint loading.
Each forward only indexes them by the packed absolute positions.

Device/dtype conversion rebuilds the tables once. Weight loading materializes each unique meta parameter once, preserving tied
embedding/head aliases, then initializes the rotary tables on CUDA. It avoids
the temporary duplicate large allocation caused by recursively using `to_empty`.
Rebuilding starts from FP32 angles rather than converting previously rounded BF16
factors back to FP32. The forward path never recomputes frequencies or trigonometry.
Tests check table pointers remain stable across prefill/decode and that conversion
rebuilds correct values.

## KV ownership and validation boundaries

Serving uses `KVCacheManager` for reference-counted blocks and full-prefix reuse.
The model receives device block tables and slot mappings through `ModelBatch.paged`;
attention writes only new KV and returns no replacement caches. Forward failures
never publish incomplete blocks. Terminal events release references; cached pages
may remain available for reuse. `cache_bytes` counts physical pages with active
references, while `pool_bytes` measures the fixed allocation. Neither is the
allocator's reserved-byte metric. The contiguous backend retains independent
per-request replacement tensors as a numerical reference.

The engine validates user token IDs, context limits, request IDs and sampling
settings at admission. The loader validates external checkpoint structure. The
runner checks only cache continuity at its execution boundary. Inner layers trust
the batch metadata and no longer repeat token, vocabulary, shape, or per-layer
cache checks. The engine retains the result-key/width contract and the sampler
retains non-finite/all-masked checks because these detect execution failures.

`Qwen2ForCausalLM.forward` accepts only `ModelBatch` and returns `BatchOutput`.
Numerical tests explicitly build packed inputs and select sampling indices for
all-token comparisons; there is no single-tensor path in the production model.

## Scope and numerical limits

Paged storage removes persistent full-history replacement copies and supports prefix
sharing. Stage 3 adds paged FlashAttention, split decode and elementwise fusions;
eager remains an explicit reference backend. Stage 4 adds optional decode CUDA
Graphs and per-channel W8A16 projections. Batch token budget counts real
input tokens, whereas memory policy also accounts for padded attention workspace
and the general sampler's scores, sorting workspace, cumulative weights and histories.
The service uses conservative capacity estimates and real CUDA warmup/peak
measurements; see [serving](serving.md).

FP32 checks use `atol=2e-4, rtol=2e-5` for the real checkpoint and tighter tolerances
for tiny models. BF16/FP16 comparisons use the same execution shape as the oracle.
Changing chunk/batch shapes can change reduced-precision results; bitwise batch
invariance is not promised. The earlier BF16 seven-token chunk comparison had
exact native/reference parity for the same shape but up to `3.84375` difference
versus a full prompt, also present in the oracle. This is why cross-schedule
correctness diagnosis uses FP32.

## Run

```bash
uv run ajvllm-generate --prompt "What is the capital of France?"
uv run ajvllm-generate --dtype float32 --prompt "Explain GQA." --max-tokens 48
uv run pytest tests/test_qwen2_cuda.py tests/test_batching_cuda.py tests/test_mixed_scheduling_cuda.py -s
uv run python tests/check_local_batch.py
```

CUDA is mandatory. `AJVLLM_TEST_MODEL` overrides the local test checkpoint path;
checkpoint-specific EOS expectations still target Qwen2.5-0.5B-Instruct. The
original CPU synthetic runner, tests, demo and configuration have been deleted.
