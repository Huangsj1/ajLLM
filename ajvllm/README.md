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
The eager oracle uses grouped QK/PV matmuls without replicating KV. The default
BF16 paged path uses native Triton attention with direct block-table reads.
The memory manager owns reference-counted pages and prefix hashes; attention uses
batched scatter/gather without separate model or attention forwards. RoPE tables are precomputed and indexed.

Stage 2b implements block allocation, prefix caching, copy-on-write and preemption
with recomputation. Stage 3 implements paged FlashAttention prefill, partitioned
Flash Decode, and fused RMSNorm/residual, RoPE/cache writes and SwiGLU.
Set `[compute] backend = "eager"` to retain the original numerical oracle. All sampling policies
use one CUDA pipeline: fused Triton transforms/scans, stable CUDA sorting and
incremental penalty histories. Penalties, masks, temperature and top-k/top-p stay
on the GPU; only selected results return to the host.

Public imports remain `from ajvllm import Engine, EngineConfig, SamplingParams`.
Configuration lives in `config/`, lifecycle types in `requests/`, tokenizer/chat
handling in `tokenization/`, and runtime memory policy in `runtime/`. Memory algorithms live in `memory/`, native Triton kernels in `kernels/`, and
backend metadata/selection in `attention/backends/`. Optional W8A16 conversion lives
in `quantization/`, and bounded decode CUDA Graphs in `runtime/graphs.py`.

See [architecture](docs/architecture/architecture.md), [model execution](docs/model_baseline.md),
[serving](docs/serving.md), and [benchmarking](docs/benchmarking.md)
for algorithms, limitations, and validation records.

## Service performance measurements

Start with `uv run ajvllm-serve --config configs/engine/benchmark.toml --profile-steps`.
In another terminal run `uv run ajvllm-benchmark --concurrency 4 --requests 24`.
The default dataset has 12 long prompts (about 1024/2048/3072 tokens), with 64
output tokens per request. Reports include TTFT, decode duration, TPOT, throughput
and GPU utilization, saved under `benchmarks/results/`.
To sweep token budgets and decode sequence limits with automatic service startup, run
`uv run ajvllm-benchmark-budget-config --output benchmarks/results/budget`.
It writes tables, plots, and candidate TOMLs; see
[budget selection](docs/benchmarking.md#choosing-scheduling-budgets).

See [benchmarking](docs/benchmarking.md) for memory expectations, comparison settings,
and sampling controls.

Stage 4 options are independent: set `[graphs] enabled = true` for decode graph
replay and `[quantization] mode = "w8a16"` for INT8 decoder projection weights with
FP16/BF16 activations. Both default to off. See the serving guide for memory/capture
limits and the benchmark guide for measured speed and quantization-quality tradeoffs.

Compare against the installed vLLM on the same GPU with
`uv run ajvllm-compare --graphs --output benchmarks/results/compare-graphs`.
Both servers run serially with identical token inputs, BF16 and fixed KV capacity.
The workflow writes a multi-metric plot and detailed JSON; see
[comparison methodology](docs/benchmarking.md#reproducible-comparison-with-vllm).
