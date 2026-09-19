# Native Qwen2.5 baseline

Implemented: configuration validation, eager dense/GQA forward, contiguous layer
KV, and strict local safetensors loading. Model execution runs on CUDA without
Transformers model delegation. See the [baseline guide](../../../../docs/model_baseline.md)
and the [architecture](../../../../docs/architecture/architecture.md).
