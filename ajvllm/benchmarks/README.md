# Benchmark assets

`datasets/long.jsonl` is the reusable workload. Rebuild it with
`uv run python benchmarks/build_dataset.py` using the local tokenizer.

Run `uv run ajvllm-benchmark` against an already running server. See
[benchmarking](../docs/benchmarking.md) for configuration, stochastic sampling
options, concurrency and metric definitions. Reports go to the ignored `results/`
directory; temporary experiment scripts are not kept here.

For a serial single-GPU comparison against installed vLLM, run
`uv run ajvllm-compare --output benchmarks/results/compare-eager`.
Add `--graphs` for decode CUDA Graphs in both engines. The workflow produces a
six-panel `comparison.png` and a reproducible `report.json`; see
[comparison methodology](../docs/benchmarking.md#reproducible-comparison-with-vllm).
