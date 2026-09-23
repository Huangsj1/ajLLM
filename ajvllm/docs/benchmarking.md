# Service benchmarking

Start the server in one terminal and the benchmark client in another. The client
does not load model weights. Run GPU workloads serially; begin with concurrency 1,
then 2 and 4. The local RTX 3080 Ti has 12 GiB; higher concurrency is not a default
validation target for the padded eager backend.

```bash
# Terminal 1: restart for each controlled comparison.
uv run ajvllm-serve --config configs/engine/benchmark.toml --profile-steps

# Terminal 2: general stochastic sampling, repeated small dataset.
uv run ajvllm-benchmark --concurrency 1 --requests 24 --max-tokens 16 \
  --temperature 0.8 --top-p 0.9 --seed 0 --output benchmarks/results/c1.json
uv run ajvllm-benchmark --concurrency 2 --requests 24 --max-tokens 16 \
  --temperature 0.8 --top-p 0.9 --seed 0 --output benchmarks/results/c2.json
uv run ajvllm-benchmark --concurrency 4 --requests 24 --max-tokens 16 \
  --temperature 0.8 --top-p 0.9 --seed 0 --output benchmarks/results/c4.json
uv run ajvllm-benchmark --concurrency 8 --requests 24 --max-tokens 16 \
  --temperature 0.8 --top-p 0.9 --seed 0 --output benchmarks/results/c8.json
```

The CLI defaults are 24 requests, concurrency 4, 64 output tokens, one excluded
sequential warmup request, temperature 0.8, top-p 0.9, unlimited top-k, seed 0,
and a 300-second socket timeout. `--top-k` enables a candidate limit;
`--temperature 0` requests deterministic selection through the same CUDA sampler.
The report records the sampling settings, including the seed. The workflow fixes
`ignore_eos=true` to request a repeatable output length; actual token counts are
recorded because context capacity can still shorten a response.

## Dataset and memory

`benchmarks/datasets/long.jsonl` contains twelve English operational-report prompts:
four near each of 1024, 2048 and 3072 tokens with the local Qwen tokenizer. Requests
reuse rows cyclically. `benchmarks/build_dataset.py` reproducibly rebuilds the file
using only the local tokenizer. It does not load a model.

`--dataset` accepts JSONL rows containing exactly one of `prompt`, `messages`, or
`token_ids`, plus an optional `id`. A custom shorter dataset and fewer requests
are useful for smoke tests. The longest default input plus 64 output tokens fits
the benchmark configuration's 4096-token context; an older 2048-context server
cannot run all rows. HTTP concurrency is not a guaranteed GPU batch size: actual
admission, chunk size, sequence capacity and token budget come from the scheduler.
Inspect `/health` for `resolved_engine` and the current budget. The server derives
its initial budget from TOML `max_num_batched_tokens`; memory estimation and
warmup can reduce it before serving requests.

For BF16 Qwen2.5-0.5B, persistent KV uses 12 KiB per token. Four resident 2048-token
inputs need about 96 MiB just for KV. The fixed paged pool reserves physical capacity at startup; eager gathers and padded
attention add workspace. The contiguous reference additionally copies replacement caches. Memory utilization is a capacity ceiling, not a target
that the server fills with unused allocations. Long documents exercise real KV
and attention costs; this synthetic repeated dataset is not a production traffic
model. Increase request repetitions for longer measurements rather than inflating
prefill chunks solely to consume memory.

## Measurements

| Metric | Definition |
| --- | --- |
| `ttft_s` | Client send to first streamed token, including transport, tokenization, queueing and prefill |
| `decode_s` | First to last streamed token; zero for one-token output |
| `tpot_s` | First-to-last duration divided by output tokens minus one; null for fewer than two tokens |
| `itl_s` | Inter-token delivery intervals, affected by transport buffering |
| `latency_s` | Client send through successful stream completion |
| Output tokens/s | Successful output tokens divided by measured workload wall time, including prefill |
| Requests/s | Successful completions divided by workload wall time |

Latency summaries contain mean, p50, p95 and maximum. Per-request server timings
start at engine admission and exclude client transport/tokenization. Failed
requests are recorded separately, excluded from latency distributions and cause
CLI exit code 1. They are not silently retried.

With `--profile-steps`, before/after health snapshots provide deltas for pure
prefill, pure decode and mixed step counts/times. They include engine overhead and runtime budget checks;
CUDA synchronization ensures partial-prefill work has completed. Mixed steps
cannot be attributed accurately to separate prefill/decode compute times. TTFT
is not pure prefill time, and pure-prefill step means are not total request prefill
latencies. A category with no samples has a null mean.

`stage_seconds` breaks down preparation, model forward, sampling (the complete
CUDA policy pipeline), and compact result transfer. It includes host launch and
metadata work, not just GPU kernel time. `transfer_bytes` counts only selected
result packets: token IDs, log probabilities and validity flags, 24 bytes per
sample. Full vocabulary logits stay on CUDA. Stage totals exclude some engine
bookkeeping and need not sum to full latency. Without profiling, stage counters
remain zero; normal execution adds no profiling synchronization.

A background nvidia-smi sampler reports device utilization percent and memory in
MiB. It observes the entire selected local GPU, including other processes. For a
remote server, run the client on its GPU host to make these readings relevant.
Unsupported counters produce explicit errors, not fabricated zero utilization.
Use `--gpu` to choose an index/UUID and `--interval` to change the sampling interval.
Very short runs have few GPU samples; do not overinterpret their utilization means.

PyTorch `allocator` fields distinguish live allocations from reserved blocks.
Peak counters follow the runtime budget's per-step reset policy; the memory
controller's `observed_peak_bytes` retains its lifetime maximum. Device-wide
nvidia-smi usage is neither of these. Restart between comparable runs so historical
allocator reservations and peak counters do not contaminate conclusions.

## Comparison discipline

Keep checkpoint, dtype, input dataset, output length, sampling policy, CPU thread
settings, profiling, warmup and background GPU load fixed. Reports save a dataset
SHA-256, client options, per-request data, GPU summaries and server health snapshots.
Adaptive budgets may change during a run; compare their before/after values.
Do not send unrelated traffic while measuring, because server timing deltas include
all concurrent work. Repeat runs when differences are small; compare latency and
memory alongside throughput. The summary does not prove compute saturation.

Reports are written to the git-ignored `benchmarks/results/` directory. Keep only
results useful to ongoing comparisons; one-off experiment runners and historical
reference implementations are not part of the maintained benchmark workflow.
Implementation rationale and brief optimization history live in
[architecture](architecture/architecture.md#implementation-refinements).

## Stage 2b bounded comparison

For storage-only comparisons set `[memory] backend = "contiguous"`, then
`backend = "paged", enable_prefix_cache = false`. Measure prefix reuse separately
with `enable_prefix_cache = true`. Restart for each case. The repeated dataset
and excluded warmup request deliberately warm the prefix cache; this is not a
cold or unique-prompt benchmark. To test cold misses use unique token sequences
or disable prefix caching. A cache salt creates an isolation domain, not a memory
capacity limit.

The benchmark report includes `kv_cache.counters` as before/after deltas.
`kv_write_bytes` counts logical persistent KV writes across layers, excluding
eager context gathers, temporary tensors and hardware memory transactions.
`cow_copy_bytes` records full-block copies separately. `pool_bytes` is fixed
storage; `peak_used_bytes` is a lifetime high-water mark of referenced physical
pages, not a benchmark delta. These complement allocator and device-wide metrics.

Measured on the local RTX 3080 Ti with Qwen2.5-0.5B-Instruct BF16, two CPU
threads, synchronized stage profiling, and a 40% PyTorch allocator cap. Each
cell below used a fresh process with fixed scheduling (no adaptive controller):
context 1152, sequence cap 4, token budget 512, per-request prefill chunk 128,
aggregate prefill budget 480, and block size 16. The paged pool contained 288
blocks (54 MiB). Tokenize the first four `long.jsonl` prompts without special
tokens and truncate them to 256, 512, 768 and 1024 tokens respectively. Cycle
these four rows over 12 measured requests after one excluded warmup request;
use 16 output tokens, temperature 0.8, top-p 0.9, seed 0 and ignore EOS.
All nine cases completed 12/12 requests without failures.

| Backend | Concurrency | Output tokens/s | Mean TTFT (s) | Mean TPOT (s) | Peak allocated MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Contiguous | 1 | 24.50 | 0.1607 | 0.0328 | 1009.8 |
| Paged, prefix off | 1 | 25.11 | 0.1484 | 0.0326 | 1041.3 |
| Paged, prefix on | 1 | 28.42 | 0.0818 | 0.0321 | 1041.3 |
| Contiguous | 2 | 40.93 | 0.2051 | 0.0360 | 1045.5 |
| Paged, prefix off | 2 | 38.64 | 0.2231 | 0.0376 | 1059.2 |
| Paged, prefix on | 2 | 51.85 | 0.1006 | 0.0333 | 1058.8 |
| Contiguous | 4 | 70.21 | 0.2286 | 0.0382 | 1101.4 |
| Paged, prefix off | 4 | 73.57 | 0.2325 | 0.0358 | 1097.3 |
| Paged, prefix on | 4 | 95.23 | 0.1094 | 0.0344 | 1095.5 |

These are single bounded observations, not statistical throughput guarantees or
results for the longer default benchmark settings. Allocated peaks include the
excluded warmup; they exclude allocator reservations and CUDA driver memory.
No conclusion about sustained SM utilization follows from these short runs.

Persistent KV writes fell from 1681.9–1708.7 MiB in the contiguous path to
92.1 MiB with paging alone (about 94.5% fewer bytes). Contiguous writes include
repeated full-context cache replacements; page writes include only 7860 computed
tokens. This excludes the eager history gathers that still occur on every step.
Paging alone changed throughput by +2.5%, -5.6% and +4.8% at concurrency 1/2/4:
metadata preparation and eager gathers can offset savings at this scale. Small
speed differences need repeated measurements before drawing stronger conclusions.

Prefix-enabled cases each reused 5232 input tokens across nine hits, reducing
persistent writes further to 30.8 MiB. Relative to contiguous storage, observed
throughput rose 16.0%, 26.7% and 35.6%; mean TTFT fell about 49–52%. This benefit
comes mainly from skipped prefill, not a faster decode attention kernel. There
were no evictions or preemptions in these measurements; tiny CUDA tests separately
force these conditions and verify output/RNG consistency and ownership cleanup.

The 54 MiB fixed pool slightly increased peak allocation at concurrency 1 and 2.
Paging provides bounded ownership, reuse and a kernel-ready address mapping; it
does not promise lower total VRAM for every workload. The next compute stage
should remove eager gathers and quadratic attention temporaries while preserving
the same packed batch, block-table and slot-mapping contracts. No temporary
comparison scripts or per-run result files are retained in the repository.

## Stage 3 compute comparison

Use `[memory] backend = "paged"` and compare `[compute] backend = "eager"` against
`"triton"` (the default `"auto"` resolves to Triton for the local BF16 checkpoint).
Restart between runs; `/health.compute_backend` records the resolved choice.
Disable prefix caching to isolate compute savings, then measure prefix reuse
separately. Warm the actual input/output shapes before timing: Triton JIT cost is
cold-start work and may otherwise dominate a short trace. The default one-request
client warmup does not necessarily exercise all mixed or split-decode variants.

Bounded local measurements used RTX 3080 Ti, Qwen2.5-0.5B-Instruct BF16, two CPU
threads and synchronized stage profiling. Each backend/concurrency used a fresh
process with fixed scheduling: max context 1152, sequence cap 4, token budget 512,
per-request prefill chunk 128 and aggregate prefill cap 480. Both used the same
54 MiB paged pool (288 blocks of 16 tokens), with prefix caching disabled and no
adaptive budget controller in the comparison. This prevents backend-dependent
capacity resolution from changing the workload. Slice the first four `long.jsonl`
rows to 256/512/768/1024 tokens, cycle them over 12 measured requests, and request
16 outputs with temperature 0.8, top-p 0.9, seed 0 and ignore EOS. An excluded
four-request pass and a small runner probe warm the kernel paths before each run.

| Concurrency | Eager tokens/s | Triton tokens/s | Speedup | Eager / Triton TTFT (s) | Eager / Triton TPOT (s) | Eager / Triton peak allocated MiB |
| --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | 23.72 | 48.64 | 2.05x | 0.1627 / 0.0739 | 0.0341 / 0.0170 | 1051.3 / 1038.3 |
| 2 | 40.51 | 75.30 | 1.86x | 0.2137 / 0.1187 | 0.0359 / 0.0191 | 1069.1 / 1048.2 |
| 4 | 68.28 | 125.83 | 1.84x | 0.2196 / 0.1238 | 0.0385 / 0.0221 | 1106.4 / 1066.9 |

All six reported cases completed 12/12 requests with no failures. Each wrote
7860 KV tokens and had no prefix hits, so the throughput gain does not come from
skipping input computation. Total profiled model time fell from 7.108/4.076/2.387 s
to 3.073/1.909/1.115 s at concurrency 1/2/4. Pure-decode step means fell from
0.0331/0.0342/0.0397 s to 0.0160/0.0175/0.0194 s. Scheduler batch timing can differ
with concurrent HTTP arrivals, so mixed-step counts need not match exactly.
These are bounded observations, not guarantees for the much larger default trace.
Most resident memory here is weights and the fixed pool; smaller attention
workspace therefore gives a modest total allocated-memory reduction at these sizes.

An initial implementation specialized block-table width and SwiGLU token count
at compile time. Its first concurrency-4 trace showed JIT stalls and only 49.61
tokens/s. Making these dimensions runtime values removed the excessive variants;
short-context decode also now avoids a separate merge launch. The table reports
the corrected implementation after warmup, not a mixture of old and new kernels.

### Attention-only measurements

A separate bounded CUDA-event microbenchmark used BF16, 14 query heads, two KV
heads, head dimension 64, page size 16 and shuffled physical page IDs. It compared
the eager gather/grouped-matmul/FP32-softmax reference with native paged attention;
Q/K/V were already computed. Three warmups preceded 20 timed calls; the table
shows median elapsed GPU event time and peak incremental allocated tensor memory.
Metadata, persistent KV and inputs were allocated before measuring. These timings
include launch gaps but exclude projections, RoPE/cache writes, MLPs and sampling.

| Workload: query lengths / context lengths | Eager ms | Triton ms | Eager / Triton temporary MiB |
| --- | ---: | ---: | --- |
| Chunk prefill: `[512]` / `[2048]` | 0.9994 | 0.1431 | 141.875 / 0.875 |
| Mixed: `[128,1,1,1]` / `[2048,4096,3072,1024]` | 1.5985 | 0.1700 | 288.875 / 0.391 |
| Decode: `[1,1,1,1]` / `[4096,3072,2048,1024]` | 0.4229 | 0.0922 | 12.889 / 0.229 |

On the same decode case, forcing a single partition took 0.2120 ms with 0.0068 MiB
output workspace. Splitting uses more small temporary buffers but exposes enough
parallel work to outweigh its merge overhead in this measurement. This does not
establish an optimal threshold on other GPUs. The large mixed/prefill workspace
reduction comes from eliminating query padding, dense KV gather and materialized
scores/probabilities; physical KV pool capacity itself is unchanged.

Tiny CUDA validation covers FP32/FP16/BF16, odd page/head sizes, long cached-prefix
chunks, empty decode partitions, 7:1 GQA, large logits, prefix reuse, preemption,
one-forward/no-gather invariants, and persistent service behavior. A separate
single-checkpoint FP32 comparison measured maximum logit differences of
`3.4333e-5` (prefill), `3.0995e-5` (mixed) and `3.4154e-5` (decode), within
`atol=3e-4, rtol=3e-5`. Reduced-precision scheduling/backend changes are not
promised to generate bitwise-identical logits or sampled text.

No concurrency-8 stress run was used. Tests used a 15% allocator cap; real-model
comparisons used 40%. Temporary experiment scripts were removed; the reusable
benchmark client and datasets remain the supported comparison workflow.
