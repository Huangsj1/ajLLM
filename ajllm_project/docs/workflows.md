# Workflows

## Tokenizer

1. `tokenizer train` reads a raw dataset split, learns byte-level BPE merges, and writes a tokenizer artifact.
2. `tokenizer encode` loads that artifact and streams selected dataset splits into native-endian `uint32` files.
3. The encoded dataset manifest records the tokenizer fingerprint, token counts, hashes, and source files.

Both pre-tokenization and `uint32` file encoding support optional process pools. The default configurations enable them and expose worker count and chunk size in YAML.

## Language Model

1. `lm train` validates artifact lineage, builds the selected model, and creates a self-contained run directory.
2. `lm evaluate` loads a checkpoint and reports held-out cross-entropy and perplexity.
3. `lm sweep` expands a YAML grid into independent trials and writes a comparison table and learning curves.
4. `lm compare` ranks an explicit list of existing runs.
5. `lm generate` recovers model and tokenizer information from a training run and generates completions.

Checkpoints include the model state, optimizer state, step, resolved configuration, tokenizer fingerprint, and model metadata. `latest.pt`, `best.pt`, and `final.pt` have explicit meanings and can be selected by downstream workflows.

Every training run also writes a model report with exact parameter counts and approximate memory/FLOPs planning values. The estimate separates parameters, saved activations, gradients, AdamW moments, forward FLOPs, per-step training FLOPs, and total training FLOPs.
