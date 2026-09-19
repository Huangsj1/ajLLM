"""CUDA Qwen2 runner with private contiguous KV per request."""

import json
from pathlib import Path

import torch

from ajvllm.config import require_int
from ajvllm.execution.batch import KVCache, ModelBatch
from ajvllm.modeling.qwen2.model import Qwen2ForCausalLM
from ajvllm.modeling.qwen2.weights import load_qwen2
from ajvllm.scheduling.batch import SchedulerOutput


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
    def __init__(self, model: Qwen2ForCausalLM, eos_token_ids: tuple[int, ...] = ()):
        if model.device.type != "cuda":
            raise ValueError("Qwen2Runner requires a CUDA model")
        self.model = model.eval().requires_grad_(False)
        self.vocab_size = model.config.vocab_size
        self.max_model_len = model.config.max_position_embeddings
        self.eos_token_ids = tuple(eos_token_ids)
        self._caches: dict[str, KVCache] = {}

    @classmethod
    def from_directory(
        cls, directory: str | Path, *, device: str = "cuda", dtype: torch.dtype | None = None
    ) -> "Qwen2Runner":
        model = load_qwen2(directory, device=device, dtype=dtype)
        return cls(model, read_eos_token_ids(directory, model.config.vocab_size))

    @property
    def num_active_states(self) -> int:
        return len(self._caches)

    @property
    def cache_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for cache in self._caches.values()
            for layer in cache
            for tensor in layer
        )

    def cached_tokens(self, request_id: str) -> int:
        """Returns the number of tokens already cached for the given request ID."""
        cache = self._caches.get(request_id)
        # cache[0][0] means 1st layer's k tensor, shape: (num_heads, seq_len, head_dim)
        return 0 if cache is None else cache[0][0].shape[1]

    @torch.inference_mode()
    def execute(self, batch: SchedulerOutput) -> dict[str, list[float]]:
        # Admission validates tokens and the scheduler constructs valid slices. The
        # runner checks only the cache-continuity contract across this boundary.
        for item in batch.requests:
            if item.start_pos != self.cached_tokens(item.request_id):
                raise ValueError(f"noncontiguous input for {item.request_id}")
        if not batch.requests:
            return {}
        # All scheduled slices share one packed model forward.
        sampling_rows = [row for row, item in enumerate(batch.requests) if item.do_sample]
        inputs = ModelBatch.build(
            [item.token_ids for item in batch.requests],
            [self._caches.get(item.request_id) for item in batch.requests],
            self.model.device,
            sampling_rows,
        )
        output = self.model(inputs)
        self._caches.update((item.request_id, cache) for item, cache in zip(batch.requests, output.caches, strict=True))
        if output.logits is None:
            return {}
        rows = output.logits.cpu().tolist()
        return {batch.requests[index].request_id: logits for index, logits in zip(sampling_rows, rows, strict=True)}

    def release(self, request_id: str) -> None:
        self._caches.pop(request_id, None)
