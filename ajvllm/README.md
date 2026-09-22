# ajvLLM

An educational single-GPU inference engine for Qwen2.5 dense/GQA models.
It implements native model execution, packed continuous batching, chunked prefill,
paged KV storage with prefix reuse, and a persistent HTTP/SSE service. Model execution does
not delegate to vLLM or Transformers. Transformers provides tokenization and the
independent CUDA test oracle.

## Start the service

```bash
uv sync --locked --extra dev
uv run ajvllm-serve --gpu-memory-utilization 0.7
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
Its token budget is configured in TOML; startup calibration and per-step adaptation
share the same runtime as the service. Default tests use tiny CUDA models for batching, lifecycle, serving and numerical
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
Grouped QK/PV matmuls share K/V without replicating them for every query head.
The memory manager owns reference-counted pages and prefix hashes; attention uses
batched scatter/gather without separate model or attention forwards. RoPE tables are precomputed and indexed.

Stage 2b implements block allocation, prefix caching, copy-on-write and preemption
with recomputation. Native PagedAttention and FlashAttention kernels are still
planned; padded attention workspace and eager context gathers still have costs. All sampling policies
use one batched CUDA tensor pipeline, including penalties, masks, temperature,
top-k/top-p and random selection. Only selected results return to the host.

Public imports remain `from ajvllm import Engine, EngineConfig, SamplingParams`.
Configuration lives in `config/`, lifecycle types in `requests/`, tokenizer/chat
handling in `tokenization/`, and runtime memory policy in `runtime/`. Memory algorithms live in `memory/`; kernel, quantization and distributed
directories remain reserved.

See [architecture](docs/architecture/architecture.md), [model execution](docs/model_baseline.md),
[serving](docs/serving.md), and [benchmarking](docs/benchmarking.md)
for algorithms, limitations, and validation records.

## Service performance measurements

Start with `uv run ajvllm-serve --config configs/engine/benchmark.toml --profile-steps`.
In another terminal run `uv run ajvllm-benchmark --concurrency 4 --requests 24`.
The default dataset has 12 long prompts (about 1024/2048/3072 tokens), with 64
output tokens per request. Reports include TTFT, decode duration, TPOT, throughput
and GPU utilization, saved under `benchmarks/results/`.
See [benchmarking](docs/benchmarking.md) for memory expectations, comparison settings,
and sampling controls.
