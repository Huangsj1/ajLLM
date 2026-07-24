# Configuration

Configuration is split into reusable components and runnable tasks.

```text
configs/
├── datasets/
├── tokenizers/
├── models/
└── runs/
    ├── tokenizer_train/
    ├── tokenizer_encode/
    ├── lm_train/
    ├── lm_sweep/
    ├── lm_evaluate/
    ├── lm_compare/
    └── lm_generate/
```

A run references components by name:

```yaml
dataset: tinystories
tokenizer: tinystories_bpe_10k
model: transformer_baseline
```

The loader resolves these names from their standard directories. A run may inherit another run with a relative `base` path. Nested values are merged, so a sweep or ablation only needs to state changed values.

Paths inside dataset and prompt configurations are resolved relative to the project root. Absolute paths are also accepted.

## Model Variants

```yaml
position_encoding:
  type: rope       # rope, learned, or none
  theta: 10000

normalization:
  type: rmsnorm    # rmsnorm or none
  placement: pre   # pre or post

feed_forward:
  type: swiglu     # swiglu or silu

tie_embeddings: false
```

The vocabulary size is never duplicated in LM configs. It is derived from the selected tokenizer artifact and checked against the encoded dataset manifest.

## Multiprocessing

Tokenizer training and file encoding expose independent process controls:

```yaml
# tokenizer component
pretokenization:
  parallel: true
  num_processes: 8
  chunk_size_bytes: 8388608

# tokenizer encode run
encoding:
  parallel: true
  num_processes: 8
  chunk_size: 16777216
  buffer_tokens: 65536
```

BPE workers are aligned to special-token document boundaries. Encoding workers write temporary binary chunks that the coordinator concatenates in source order. Set `parallel: false` when profiling or debugging on a small corpus.
