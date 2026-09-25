"""CUDA Qwen2 runner; KV ownership is delegated to the memory subsystem."""

import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from ajvllm.attention.backends.triton import resolve_backend
from ajvllm.config import EngineConfig, require_int
from ajvllm.config.advanced import GraphConfig
from ajvllm.config.compute import ComputeConfig
from ajvllm.config.memory import MemoryConfig
from ajvllm.execution.batch import KVCache, ModelBatch
from ajvllm.memory.manager import KVCacheManager
from ajvllm.modeling.qwen2.model import Qwen2ForCausalLM
from ajvllm.modeling.qwen2.projections import pack_projections
from ajvllm.modeling.qwen2.weights import load_qwen2
from ajvllm.quantization.linear import quantization_stats
from ajvllm.runtime.graphs import DecodeGraphs
from ajvllm.scheduling.batch import Phase, ScheduledRequest, SchedulerOutput


def read_eos_token_ids(directory: str | Path, vocab_size: int) -> tuple[int, ...]:
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    generation_path = directory / "generation_config.json"
    if generation_path.exists():
        config.update(json.loads(generation_path.read_text()))
    ids = config.get("eos_token_id", [])
    ids = [] if ids is None else [ids] if type(ids) is int else ids
    for token in ids:
        require_int("EOS token ID", token)
        if token >= vocab_size:
            raise ValueError("EOS token is outside checkpoint vocabulary")
    # [<|im_end|>, <|endoftext|>]
    return tuple(dict.fromkeys(ids))


class Qwen2Runner:
    def __init__(
        self,
        model: Qwen2ForCausalLM,
        eos_token_ids: tuple[int, ...] = (),
        *,
        memory_config: MemoryConfig | None = None,
        compute_config: ComputeConfig | None = None,
        graph_config: GraphConfig | None = None,
        engine_config: EngineConfig | None = None,
    ):
        if model.device.type != "cuda":
            raise ValueError("Qwen2Runner requires a CUDA model")
        self.model = model.eval().requires_grad_(False)
        self.vocab_size = model.config.vocab_size
        # model's real maximum context length, which may be larger than the engine's max_model_len, but not used
        self.max_model_len = model.config.max_position_embeddings
        self.eos_token_ids = tuple(eos_token_ids)
        self.context_limit = engine_config.max_model_len if engine_config else self.max_model_len
        self._caches: dict[str, KVCache] = {}  # original contiguous KV caches
        self.kv_cache: KVCacheManager | None = None  # KV cache manager for paged memory
        memory_config = memory_config or MemoryConfig(backend="contiguous")
        self.compute_backend = resolve_backend(compute_config or ComputeConfig(backend="eager"), model, memory_config)
        if self.compute_backend == "triton":
            pack_projections(model)
        if memory_config.backend == "paged":
            if engine_config is None:
                raise ValueError("paged runner requires a resolved engine_config")
            cfg = model.config
            self.kv_cache = KVCacheManager(
                memory_config,
                layers=cfg.num_hidden_layers,
                kv_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim,
                device=model.device,
                dtype=model.dtype,
                max_model_len=engine_config.max_model_len,
                max_num_seqs=engine_config.max_num_seqs,
            )
        graph_config = graph_config or GraphConfig()
        if graph_config.enabled and self.compute_backend != "triton":
            raise ValueError("CUDA graphs require the Triton paged backend")
        self.graphs = DecodeGraphs(model, graph_config, self.context_limit) if graph_config.enabled else None
        self.quantization = quantization_stats(model)
        self._cache_written_tokens = 0
        self.profile_enabled = False
        self.stage_seconds = {key: 0.0 for key in ("prepare", "model")}

    def memory_stats(self):
        if self.kv_cache is not None:
            return self.kv_cache.snapshot()
        cfg = self.model.config
        token_bytes = (
            2
            * cfg.num_hidden_layers
            * cfg.num_key_value_heads
            * cfg.head_dim
            * self.model.model.embed_tokens.weight.element_size()
        )
        return {
            "backend": "contiguous",
            "used_bytes": self.cache_bytes,
            "written_tokens": self._cache_written_tokens,
            "kv_write_bytes": self._cache_written_tokens * token_bytes,
        }

    @contextmanager
    def _stage(self, name):
        if not self.profile_enabled:
            yield
            return
        torch.cuda.synchronize(self.model.device)
        started = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize(self.model.device)
            self.stage_seconds[name] += time.perf_counter() - started

    @classmethod
    def from_directory(
        cls, directory: str | Path, *, device: str = "cuda", dtype: torch.dtype | None = None
    ) -> "Qwen2Runner":
        model = load_qwen2(directory, device=device, dtype=dtype)
        return cls(model, read_eos_token_ids(directory, model.config.vocab_size))

    @property
    def num_active_states(self) -> int:
        return len(self.kv_cache.states) if self.kv_cache is not None else len(self._caches)

    @property
    def cache_bytes(self) -> int:
        if self.kv_cache is not None:
            return self.kv_cache.snapshot()["used_bytes"]
        return sum(
            tensor.numel() * tensor.element_size()
            for cache in self._caches.values()
            for layer in cache
            for tensor in layer
        )

    def cached_tokens(self, request_id: str) -> int:
        """Returns the number of tokens already cached for the given request ID."""
        if self.kv_cache is not None:
            state = self.kv_cache.states.get(request_id)
            return len(state.tokens) if state else 0
        cache = self._caches.get(request_id)
        # cache[0][0] means 1st layer's k tensor, shape: (num_heads, seq_len, head_dim)
        return 0 if cache is None else cache[0][0].shape[1]

    @torch.inference_mode()
    def execute(self, batch: SchedulerOutput) -> dict[str, torch.Tensor]:
        # Admission validates tokens and the scheduler constructs valid slices. The
        # runner checks only the cache-continuity contract across this boundary.
        for item in batch.requests:
            if item.start_pos != self.cached_tokens(item.request_id):
                raise ValueError(f"noncontiguous input for {item.request_id}")
        if not batch.requests:
            return {}
        if self.kv_cache is not None:
            self.kv_cache.storage.wait()
            for item in batch.requests:
                if item.request_id not in self.kv_cache.states:
                    self.kv_cache.attach(item.request_id, ())
                if not self.kv_cache.reserve(item.request_id, item.end_pos):
                    raise MemoryError("KV block reservation exhausted")
        # All scheduled slices share one packed model forward.
        sampling_rows = [row for row, item in enumerate(batch.requests) if item.do_sample]
        with self._stage("prepare"):
            inputs = ModelBatch.build(
                [item.token_ids for item in batch.requests],
                [self._caches.get(item.request_id) for item in batch.requests],
                self.model.device,
                sampling_rows,
                starts=[item.start_pos for item in batch.requests],
                optimized=self.compute_backend == "triton",
            )
            if self.kv_cache is not None:
                inputs.paged = self.kv_cache.batch(
                    [item.request_id for item in batch.requests],
                    inputs.positions,
                    inputs.sequence_ids,
                    inputs.context_lengths,
                    gather=self.compute_backend != "triton",
                )
        try:
            with self._stage("model"):
                output = self.graphs.execute(inputs) if self.graphs is not None else self.model(inputs)
        finally:
            if self.kv_cache is not None:
                self.kv_cache.storage.record()
        if self.kv_cache is None:
            self._cache_written_tokens += sum(item.end_pos for item in batch.requests)
            self._caches.update(
                (item.request_id, cache) for item, cache in zip(batch.requests, output.caches, strict=True)
            )
        else:
            for item in batch.requests:
                self.kv_cache.commit(item.request_id, item.token_ids)
        if output.logits is None:
            return {}
        return {batch.requests[index].request_id: row for index, row in zip(sampling_rows, output.logits, strict=True)}

    @torch.inference_mode()
    def probe(self, sequences):
        """Exercise the production execution path without retaining warmup prefixes."""
        if self.num_active_states:
            raise RuntimeError("warmup requires an idle runner")
        manager = self.kv_cache
        enabled = manager.enable_prefix_cache if manager else False
        if manager:
            manager.enable_prefix_cache = False
        ids = [f"warmup-{row}" for row in range(len(sequences))]
        try:
            self.execute(
                SchedulerOutput(
                    tuple(
                        ScheduledRequest(rid, tuple(tokens), 0, Phase.PREFILL, True)
                        for rid, tokens in zip(ids, sequences, strict=True)
                    )
                )
            )
            if max(map(len, sequences)) < self.context_limit:
                self.execute(
                    SchedulerOutput(
                        tuple(
                            ScheduledRequest(rid, (0,), len(tokens), Phase.DECODE, True)
                            for rid, tokens in zip(ids, sequences, strict=True)
                        )
                    )
                )
        finally:
            for rid in ids:
                self.release(rid)
            if manager:
                manager.enable_prefix_cache = enabled

    def release(self, request_id: str) -> None:
        if self.kv_cache is not None:
            self.kv_cache.release(request_id)
        else:
            self._caches.pop(request_id, None)
