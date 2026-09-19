# ajvLLM

An educational single-GPU inference engine for Qwen2.5 dense/GQA models.
It implements native model execution, packed continuous batching, chunked prefill,
per-request contiguous KV, and a persistent HTTP/SSE service. Model execution does
not delegate to vLLM or Transformers. Transformers provides tokenization and the
independent CUDA test oracle.

## Start the service

```bash
uv sync --locked --extra dev
uv run ajvllm-serve --gpu-memory-utilization 0.7 --initial-token-budget 32
```

The default model is `model/Qwen2.5-0.5B-Instruct`, device `cuda:0`, dtype BF16,
and engine settings `configs/engine/qwen2.toml`. The server listens on
`127.0.0.1:8000`, receives requests continuously, and sleeps on its admission queue
when idle. One process owns one GPU engine. CUDA is required; no CPU fallback is used.

```bash
curl http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"sampling":{"max_tokens":32,"temperature":0}}'

curl -N http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"example","messages":[{"role":"user","content":"Explain GQA."}],"sampling":{"max_tokens":64,"temperature":0},"stream":true}'
```

`GET /health` reports engine and memory statistics. `DELETE /requests/example`
cancels an active or waiting request. Streaming client disconnects cancel their
requests automatically. Read the [serving guide](docs/serving.md) for input semantics,
backpressure, memory budgeting, and shutdown behavior.

## Offline generation and GPU tests

```bash
uv run ajvllm-generate --prompt "What is the capital of France?"
uv run ajvllm-generate --prompt "Name two planets." --prompt "Count to five."
uv run pytest -s
uv run ruff check .
```

The offline CLI uses the same packed model path and is useful for fixed workloads.
Its explicit token budget is configured in TOML; adaptive budgeting runs in the
service. Default tests use tiny CUDA models for batching, lifecycle, serving and numerical
checks. Real-checkpoint tests are opt-in (`-m model`) and are excluded by default
to avoid retaining large models in a full test run. A bounded single-model check
is available as `uv run python tests/check_local_batch.py`. The synthetic CPU runner/tests, demo,
and default demo configuration have been removed.

Python 3.12/3.13 and an NVIDIA GPU are required. The dependency baseline follows
`../ajllm`, with CUDA 13.0 PyTorch wheels; HTTP service dependencies are also declared
directly. `uv.lock` pins the environment. vLLM is installed for reference work and
is not imported by the engine. No model/tokenizer files are downloaded at runtime.
If the uv cache is read-only, set `UV_CACHE_DIR=/tmp/ajvllm-uv-cache`; sandboxed GPU
commands also need device access. Normal host execution uses the commands above.

## Implementation

Each scheduler iteration reserves decode tokens first, then fairly allocates its
remaining prefill budget. It executes **one packed mixed `ModelBatch` forward per step**.
Prefill chunks and decode tokens share the same model call. Embeddings, QKV/output projections,
MLPs, and selected output logits operate on packed tensors. Eager attention uses
batched matrix multiplications with padding masks and per-request causal offsets.
Loops over requests only pack/unpack their contiguous KV tensors; they never invoke
separate model or attention forwards. RoPE tables are precomputed and indexed.

This is a readable GPU baseline, not a paged or FlashAttention implementation.
Padded attention workspace and KV copying still have costs. Sampling remains a
host reference implementation after one bulk logits transfer; the removed CPU
artifacts were the synthetic backend/tests, not the host control plane.

Public imports remain `from ajvllm import Engine, EngineConfig, SamplingParams`.
Configuration lives in `config/`, lifecycle types in `requests/`, tokenizer/chat
handling in `tokenization/`, and runtime memory policy in `runtime/`. Future memory,
kernel, quantization, and distributed directories remain reserved.

See [architecture](docs/architecture/architecture.md), [model execution](docs/model_baseline.md),
[packed batching](docs/batching.md), and [serving](docs/serving.md)
for algorithms, limitations, and validation records.
