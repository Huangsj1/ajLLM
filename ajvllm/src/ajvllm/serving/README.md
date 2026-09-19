# Persistent serving

Implemented: bounded admission/output queues, an async engine loop with a dedicated
GPU thread, HTTP generation, SSE token events, cancellation, and graceful shutdown.
Distributed routing and prefill/decode disaggregation remain planned.
See [serving](../../../docs/serving.md).
