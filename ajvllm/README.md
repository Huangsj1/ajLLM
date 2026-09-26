# ajvLLM

An educational single-GPU inference engine for Qwen2.5 dense/GQA models, built
from model execution through scheduling, memory management, and CUDA kernels.
It supports continuous batching, chunked prefill, paged KV caching, prefix reuse,
CUDA sampling, decode CUDA Graphs, and optional W8A16 quantization.

Model execution does not delegate to vLLM or Transformers. Transformers supplies
tokenization and the independent model test oracle; vLLM is installed for reference
and comparison workflows. The local validation checkpoint is Qwen2.5-0.5B-Instruct
on an RTX 3080 Ti with 12 GiB VRAM.

## 1. Prepare the environment

Create the project's `.venv` and install the locked CUDA dependencies:

```bash
uv sync --locked --python 3.12 --extra dev
```

`uv sync` creates the virtual environment automatically. Activation is optional
when using `uv run`; for an interactive environment, use `source .venv/bin/activate`.
The `dev` extra adds pytest and Ruff. The dependency baseline follows `../ajllm`,
but that project does not need to be installed. Keep `uv.lock` for reproducibility;
do not replace the configured PyTorch index with CPU-only wheels.


## 2. Download the model

Download the full [Qwen2.5-0.5B-Instruct checkpoint](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
to the directory used by the default workflows:

```bash
uv run hf download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir model/Qwen2.5-0.5B-Instruct
```

The `hf` CLI is included in the project dependencies. This public checkpoint does
not normally require authentication. If authentication is needed for your Hub
access, run `uv run hf auth login` before downloading. For a reproducible checkpoint,
add `--revision <commit-hash>` to the download command. See the
[Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/guides/download#download-files-to-local-folder).

Keep both the weights and tokenizer/configuration files. The local directory
should include `config.json`, `model.safetensors`, `tokenizer.json`,
`tokenizer_config.json`, and the other files supplied by the repository. Larger
checkpoints may use multiple Safetensors shards and an index instead of one weight
file. Downloading only the weights is not sufficient for chat input processing.

The `model/` directory is ignored by Git. If the checkpoint already exists there,
skip the download. To use another local checkpoint path, pass `--model /path/to/model`
to the workflow. Other model families and MoE models are outside the current scope.

## 3. Generate text locally

Run a short CUDA generation before starting a persistent service:

```bash
uv run ajvllm-generate \
  --model model/Qwen2.5-0.5B-Instruct \
  --config configs/engine/qwen2.toml \
  --prompt "Explain grouped-query attention in two sentences." \
  --max-tokens 64
```

Repeat `--prompt` to submit several requests to the same engine:

```bash
uv run ajvllm-generate --prompt "Name two planets." --prompt "Count to five."
```

The CLI applies the checkpoint's chat template by default; `--raw` selects raw
completion. It uses the same runtime and packed model execution as the service.
First use can take longer because Triton kernels compile and decode graphs are
captured lazily.

## 4. Start the service

```bash
uv run ajvllm-serve \
  --model model/Qwen2.5-0.5B-Instruct \
  --config configs/engine/qwen2.toml \
  --device cuda:0 --dtype bfloat16 \
  --gpu-memory-utilization 0.7 \
  --host 127.0.0.1 --port 8000
```

The service receives requests continuously and sleeps on its admission queue when
idle. One process owns one GPU engine. In another terminal:

```bash
# Readiness, effective configuration, memory accounting, and runtime counters.
curl http://127.0.0.1:8000/health

# Complete response.
curl http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"sampling":{"max_tokens":32,"temperature":0.8,"top_p":0.9}}'

# Stream token events over SSE.
curl -N http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"example","messages":[{"role":"user","content":"Explain GQA."}],"sampling":{"max_tokens":64,"temperature":0.8,"top_p":0.9},"stream":true}'

# Cancel an active or waiting request from another terminal.
curl -X DELETE http://127.0.0.1:8000/requests/example
```

Streaming disconnects cancel their requests automatically. Stop the service with
Ctrl+C. This is the project's `/generate` API, not an OpenAI-compatible chat API.
See the [serving guide](docs/serving.md) for raw prompts/token IDs, request limits,
backpressure, and shutdown behavior.

## Configuration and memory

Choose a TOML under `configs/engine/` with `--config`:

| Setting | Purpose |
|---|---|
| `engine.max_model_len` | Maximum total context, including generated tokens; set from service requirements and model support. |
| `engine.max_num_batched_tokens` | Total token work allowed in one scheduler step, shared by prefill and decode. |
| `engine.max_num_seqs` | Maximum active sequences. |
| `engine.enable_chunked_prefill` | Allow long inputs to span multiple scheduler steps. |
| `memory.enable_prefix_cache` | Reuse computed full prefix blocks across requests. |
| `compute.backend` | `auto` selects supported Triton kernels; `eager` provides the reference path. |
| `graphs.enabled` | Enable bounded decode CUDA Graph capture/replay. |
| `quantization.mode` | `none` or `w8a16` decoder projection weights; activations and KV remain FP16/BF16. |

The current service preset enables CUDA Graphs and leaves quantization disabled.
Optional prefill caps can be omitted: the scheduler divides the residual token
budget among actual prefill demands, redistributing unused shares. Graph batch
buckets and capture limits remain separately configurable; increasing the sequence
limit alone does not extend graph coverage.

Scheduling budgets stay fixed during serving. At startup, the runtime profiles
workspace needs, reserves graph/safety headroom, and sizes the paged KV pool from
the remaining memory under `--gpu-memory-utilization`. Omit `memory.num_blocks`
for automatic pool sizing. Startup prints model, workspace, and pool accounting;
`/health` reports those values and allocator/KV statistics with readable units.

If startup rejects a configuration, reduce the token/sequence budgets or review
available GPU memory. Profiling reduces OOM risk but does not cover every runtime
shape or later memory use by another process. Prompt plus output tokens must fit
`max_model_len`. See [architecture](docs/architecture/architecture.md) for the
allocation and admission-control algorithms.

## Benchmarking

### Measure a running service

Start the service in one terminal, then run the client in another:

```bash
# Terminal 1.
uv run ajvllm-serve --config configs/engine/benchmark.toml --gpu-memory-utilization 0.7

# Terminal 2.
uv run ajvllm-benchmark --concurrency 4 --requests 24 --max-tokens 64 \
  --temperature 0.8 --top-p 0.9 --seed 0 \
  --output benchmarks/results/c4.json
```

Reports include TTFT, TPOT, latency, throughput, GPU utilization/memory, and KV
statistics. Add `--profile-steps` to the **server** for synchronized engine-stage
timing; use a separate run because synchronization changes performance.

The checked-in dataset contains 16 prompts: four each at 1K, 2K, 4K, and 8K tokens.
It fits the benchmark preset's 16384-token context with the output length above.
Rebuild it using the downloaded tokenizer:

```bash
uv run python benchmarks/build_dataset.py
```

### Choose token and sequence budgets

Stop the manually started service before GPU comparisons. This workflow starts
and stops its own services serially:

```bash
uv run ajvllm-benchmark-budget-config \
  --token-budgets 512 2048 8192 \
  --sequence-limits 1 4 8 16 32 \
  --requests 64 --repeats 3 \
  --output benchmarks/results/budget
```

It benchmarks long-input prefill and short-input decode scaling separately, with
prefix caching disabled and automatic KV sizing. Outputs include tables, plots,
raw measurements, startup logs, and candidate TOMLs. Use a new output directory
for each experiment. For staged selection, use `--phase prefill` first, then
`--phase decode --decode-token-budget <chosen-budget>`. See
[budget selection](docs/benchmarking.md#choosing-scheduling-budgets) for controls
and interpretation; validate the chosen pair on representative mixed workloads.

### Compare against vLLM

```bash
uv run ajvllm-compare --config configs/engine/benchmark.toml
```

The workflow runs both engines serially on the same GPU with identical token
inputs and BF16. Engine, memory, and graph settings come from the TOML; its
`[compare]` section controls concurrency, prompt lengths, requests, and repeats.
CLI flags override those values. Without an explicit concurrency list, the workflow
derives test points from `max_num_seqs`. Automatic KV sizing uses the same GPU
utilization target on both sides; set `memory.num_blocks` for equal KV bytes.
It writes a multi-metric plot, resolved configuration, and detailed JSON;
see the [comparison methodology](docs/benchmarking.md#reproducible-comparison-with-vllm).
Generated benchmark results are ignored by Git.

## Development and tests

```bash
uv run --extra dev pytest -s
uv run --extra dev ruff check .

# Optional checkpoint-backed tests (excluded from the default suite).
uv run --extra dev pytest -m model -s

# Bounded single-model CUDA check.
uv run python tests/check_local_batch.py
```

Default inference tests use small CUDA models to check batching, lifecycle,
serving, sampling, memory algorithms, and numerical correctness. Workflow/config
checks do not need model execution. Run GPU tests separately from benchmarks so
competing processes do not distort measurements. Inspect CLI options with
`uv run <workflow-name> --help`.

## Implementation and documentation

Each scheduler iteration reserves decode tokens first, then fairly allocates the
remaining prefill budget. One packed mixed `ModelBatch` forward processes both
phases. Native Triton kernels implement paged FlashAttention prefill, partitioned
Flash Decode, RMSNorm/residual fusion, RoPE/cache writes, and SwiGLU. RoPE tables
are precomputed. The eager backend remains a numerical reference.

The memory manager owns reference-counted pages, prefix hashes, copy-on-write,
and recompute preemption. All sampling policies use one CUDA pipeline; penalties,
masks, temperature, and top-k/top-p remain on the GPU. Only selected results return
to the host. Optional W8A16 and bounded decode CUDA Graphs extend this path.

Public imports: `from ajvllm import Engine, EngineConfig, SamplingParams`.

| Location | Responsibility |
|---|---|
| `src/ajvllm/config/`, `requests/`, `scheduling/` | Configuration, request lifecycle, and scheduling. |
| `src/ajvllm/runtime/`, `execution/` | Startup profiling, engine runtime, model runner, and graph execution. |
| `src/ajvllm/memory/`, `attention/`, `kernels/` | KV algorithms, attention backends, and native CUDA/Triton compute. |
| `src/ajvllm/quantization/`, `tokenization/` | Weight conversion and tokenizer/chat handling. |
| `src/ajvllm/serving/`, `workflows/` | HTTP service and executable workflows. |
| `configs/`, `benchmarks/`, `tests/` | Presets, benchmark datasets/results, and validation. |

Detailed guides: [architecture and roadmap](docs/architecture/architecture.md),
[model baseline](docs/model_baseline.md), [serving](docs/serving.md), and
[benchmarking and measured results](docs/benchmarking.md).
