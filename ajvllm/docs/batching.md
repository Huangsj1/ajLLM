# Packed mixed batching

Each nonempty `SchedulerOutput` becomes one `ModelBatch` and one model forward.
The runner preserves schedule order, concatenates token slices, and builds
absolute positions, sequence/query offsets, context lengths, and sampling indices.
The schedule retains per-request PREFILL/DECODE state; the model batch has no
single phase because both kinds of work can coexist.

Decode tokens receive budget first. Remaining prefill tokens use capped fair
shares, subject to the shared token budget, aggregate prefill cap, per-request
chunk cap, and sequence limit. These scheduling policies do not split execution.
Partial prefills update KV without sampling; final prefill chunks and decodes
select their last hidden row for the LM head. Logits transfer once per batch.
An empty schedule skips model execution. Execution failure releases affected
requests without committing partial token outputs.

Projections and MLPs process packed real tokens. The current eager attention
backend pads to batch-wide query/context maxima, so decode rows can inherit
prefill padding. Memory admission accounts for this full workspace. Future
variable-length or paged attention kernels should consume packed metadata and
eliminate padding behind this interface. Kernel dispatch may specialize work
without introducing separate whole-model prefill/decode forwards.

## Bounded validation

Run tiny CUDA test files serially with a process timeout and a 15% CUDA allocator
limit. Default pytest excludes checkpoint tests. `tests/check_local_batch.py` is
an optional single-checkpoint check with contexts of at most eight tokens and a
20% allocator limit. Do not run multiple GPU test processes concurrently.
Preserve the loader's shared embedding/head allocation to avoid transient copies.
