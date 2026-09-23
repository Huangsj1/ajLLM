# Attention backend metadata

`triton.py` resolves eager/Triton selection and builds compact ragged launch
metadata once per model batch. The original eager math remains in Qwen2 layers
as an explicit oracle. Both paths execute one mixed model forward.

See [architecture](../../../../docs/architecture/architecture.md#stage-3-compute-algorithms).
