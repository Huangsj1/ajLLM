# Benchmark assets

`datasets/long.jsonl` is the reusable workload. Rebuild it with
`uv run python benchmarks/build_dataset.py` using the local tokenizer.

Run `uv run ajvllm-benchmark` against an already running server. See
[benchmarking](../docs/benchmarking.md) for configuration, stochastic sampling
options, concurrency and metric definitions. Reports go to the ignored `results/`
directory; temporary experiment scripts are not kept here.
