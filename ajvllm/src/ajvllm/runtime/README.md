# GPU runtime policy

Implemented: conservative KV/workspace capacity estimates, real CUDA warmup,
measured peak tracking, and gradual token-budget growth/reduction in `budget.py`.
CUDA Graph capture/replay remains planned. See [serving](../../../docs/serving.md).
