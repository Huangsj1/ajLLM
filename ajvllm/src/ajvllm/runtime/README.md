# GPU runtime

Startup profiling uses a disposable KV pool, then sizes final pages from the GPU
memory target minus non-KV demand and reserves. Engine budgets remain fixed.
`inference.py` coordinates both offline and HTTP execution; `graphs.py` captures
only against the final pool. See [serving](../../../docs/serving.md).
