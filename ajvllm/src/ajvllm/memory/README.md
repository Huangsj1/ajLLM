# KV memory subsystem

`blocks.py` owns physical page IDs, references and eviction. `manager.py` maps
requests to pages, hashes complete prefixes and handles copy-on-write.
`storage.py` supplies CUDA storage and eager scatter/gather. `contiguous.py`
retains the numerical comparison backend. The scheduler owns preemption policy.

See [architecture](../../../docs/architecture/architecture.md#stage-2b-memory-algorithms)
for invariants, configuration and the future native paged-kernel boundary.
