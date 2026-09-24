"""Bounded decode CUDA graphs with stable metadata buffers and safe padded rows."""

from dataclasses import dataclass

import torch

from ajvllm.execution.batch import ModelBatch
from ajvllm.memory.storage import PagedBatch
from ajvllm.modeling.qwen2.model import BatchOutput


@dataclass
class GraphEntry:
    inputs: ModelBatch
    graph: torch.cuda.CUDAGraph
    output: BatchOutput
    retained_bytes: int


class DecodeGraphs:
    def __init__(self, model, config, context_limit):
        self.model, self.config, self.context_limit = model, config, context_limit
        self.entries = {}
        self.rejected = set()
        self.retained_bytes = 0
        self.captures = self.replays = self.fallbacks = 0
        # make a new cuda stream
        self.stream = torch.cuda.Stream(device=model.device)
        self.last_use = None

    def _key(self, inputs):
        if inputs.attention_metadata is None or inputs.paged is None:
            return None
        if any(length != 1 for length in inputs.query_lengths) or inputs.sample_indices.numel() != inputs.num_requests:
            return None
        size = next((n for n in self.config.batch_sizes if n >= inputs.num_requests), None)
        if size is None:
            return None
        context = min(self.context_limit, max(128, 1 << (inputs.max_context_len - 1).bit_length()))
        return size, context

    def _estimate(self, size):
        cfg = self.model.config
        # Conservative per-graph reserve, including intermediate activations and library workspace.
        activations = (
            size
            * cfg.num_hidden_layers
            * (8 * cfg.hidden_size + 4 * cfg.intermediate_size)
            * self.model.model.embed_tokens.weight.element_size()
        )
        return activations + size * cfg.vocab_size * 8 + 16 * 1024**2

    def _static_inputs(self, inputs, size, context):
        batch = ModelBatch.build([(0,)] * size, [None] * size, self.model.device, range(size), optimized=True)
        width = (context + inputs.paged.storage.block_size - 1) // inputs.paged.storage.block_size
        batch.paged = PagedBatch(
            inputs.paged.storage,
            torch.zeros((size, width), device=self.model.device, dtype=torch.long),
            torch.full((size,), -1, device=self.model.device, dtype=torch.long),
            None,
            None,
        )
        batch.attention_metadata.max_decode_context = context
        self._copy(batch, inputs)
        return batch

    @staticmethod
    def _copy(target, source):
        size = source.num_requests
        target.token_ids.zero_()
        target.positions.zero_()
        target.paged.slot_mapping.fill_(-1)
        target.paged.block_tables.zero_()
        target.attention_metadata.contexts.zero_()
        target.token_ids[:size].copy_(source.token_ids)
        target.positions[:size].copy_(source.positions)
        target.paged.slot_mapping[:size].copy_(source.paged.slot_mapping)
        width = source.paged.block_tables.shape[1]
        target.paged.block_tables[:size, :width].copy_(source.paged.block_tables)
        target.attention_metadata.contexts[:size].copy_(source.attention_metadata.contexts)

    def _capture(self, inputs, key):
        """ Run a model forward, and capture the CUDA graph for later replay. """
        size, context = key
        estimate = self._estimate(size)
        if len(self.entries) >= self.config.max_graphs or self.retained_bytes + estimate > self.config.reserve_bytes:
            self.rejected.add(key)
            return None
        before = torch.cuda.memory_reserved(self.model.device)
        allocated = torch.cuda.memory_allocated(self.model.device)
        static = self._static_inputs(inputs, size, context)
        current = torch.cuda.current_stream(self.model.device)
        # Wait for the current stream to finish before capturing the graph on a private stream.
        self.stream.wait_stream(current)
        # use a clean stream to warmup and capture the graph
        with torch.cuda.stream(self.stream):
            # Compile kernels and initialize BLAS handles before capture.
            for _ in range(2):
                self.model(static)
        current.wait_stream(self.stream)
        self.stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        # use self.stream to capture the graph, and replay it on the same stream
        with torch.cuda.graph(graph, stream=self.stream):
            output = self.model(static)
        current.wait_stream(self.stream)
        retained = max(
            estimate,
            torch.cuda.memory_reserved(self.model.device) - before,
            torch.cuda.memory_allocated(self.model.device) - allocated,
        )
        if self.retained_bytes + retained > self.config.reserve_bytes:
            # Releasing this private graph pool does not change KV ownership.
            self.stream.synchronize()
            self.rejected.add(key)
            return None
        entry = GraphEntry(static, graph, output, retained)
        self.entries[key] = entry
        self.retained_bytes += retained
        self.captures += 1
        return entry

    @torch.inference_mode()
    def execute(self, inputs):
        if self.last_use is not None:
            torch.cuda.current_stream(self.model.device).wait_event(self.last_use)
        key = self._key(inputs)
        entry = self.entries.get(key)
        if key is not None and key not in self.rejected and entry is None:
            # capture
            entry = self._capture(inputs, key)
        if entry is None:
            # fallback to normal execution
            self.fallbacks += 1
            return self.model(inputs)
        # copy current inputs into the static graph input buffers, then replay the graph.
        self._copy(entry.inputs, inputs)
        entry.graph.replay()
        # Callers may retain logits past the next replay; never expose mutable graph output storage.
        output = BatchOutput(entry.output.logits[: inputs.num_requests].clone(), ())
        self.last_use = torch.cuda.Event()
        self.last_use.record(torch.cuda.current_stream(self.model.device))
        self.replays += 1
        return output

    def snapshot(self):
        return {
            "enabled": True,
            "captures": self.captures,
            "replays": self.replays,
            "fallbacks": self.fallbacks,
            "cached_graphs": len(self.entries),
            "retained_budget_bytes": self.retained_bytes,
            "limit_bytes": self.config.reserve_bytes,
        }
