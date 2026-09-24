# W8A16 weight-only quantization

`linear.py` converts decoder projections to symmetric per-output-channel INT8
weights with FP32 scales. Native GEMV/GEMM lives in `kernels/quantization.py`.
Embedding/head/norm parameters, activations and KV retain their original precision.

See [architecture](../../../docs/architecture/architecture.md#stage-4-advanced-execution)
for format, conversion lifecycle, numerical limits and graph compatibility.
