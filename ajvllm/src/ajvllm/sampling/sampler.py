"""Fused CUDA sampling with request-owned incremental penalty history."""

import time
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ajvllm.execution.transfer import upload
from ajvllm.kernels.sampling import sample_sorted, update_history
from ajvllm.requests import Request


@dataclass(frozen=True)
class Sample:
    token_id: int
    logprob: float


@dataclass
class History:
    owner: weakref.ReferenceType
    signature: tuple
    buffer: torch.Tensor  # shape (3, vocab), including: seen, counts, stopped
    generated: int = 0


class Sampler:
    def __init__(self, max_histories=128):
        self.max_histories = max_histories
        self.profile_enabled = False
        self.stage_seconds = {"sampling": 0.0, "transfer": 0.0}
        self.transfer_bytes = 0
        self.histories: dict[int, History] = {}

    def release(self, request):
        self.histories.pop(id(request), None)

    def _metadata(self, requests, vocab, device, eos_ids):
        parameters, metadata, updates = [], [], []
        for request in requests:
            p = request.sampling_params
            penalties = p.repetition_penalty != 1 or p.presence_penalty != 0 or p.frequency_penalty != 0
            stops = p.effective_stop_ids(eos_ids) if len(request.output_token_ids) < p.min_tokens else ()
            history = None
            if penalties or stops:
                key = id(request)
                signature = (p, eos_ids, vocab, device)
                history = self.histories.pop(key, None)
                if (
                    history is None
                    or history.signature != signature
                    or history.generated > len(request.output_token_ids)
                ):
                    buffer = torch.zeros((3, vocab), device=device, dtype=torch.int32)
                    owner = weakref.ref(request, lambda _, key=key: self.histories.pop(key, None))
                    history = History(owner, signature, buffer)
                    base = buffer.data_ptr()
                    if penalties:
                        # input tokens
                        updates.extend(base + token * 4 for token in request.prompt_token_ids)
                    updates.extend(base + (2 * vocab + token) * 4 for token in stops if token < vocab)
                self.histories[key] = history
                base = history.buffer.data_ptr()
                if penalties:
                    # output tokens
                    for token in request.output_token_ids[history.generated :]:
                        updates.extend((base + token * 4, base + (vocab + token) * 4))
                history.generated = len(request.output_token_ids)
            else:
                self.release(request)
            parameters.append(
                (
                    p.repetition_penalty,
                    p.presence_penalty,
                    p.frequency_penalty,
                    p.temperature if p.temperature else 1.0,
                    p.top_p,
                )
            )
            metadata.append(
                (
                    history.buffer.data_ptr() if history else 0,
                    int(len(request.output_token_ids) < p.min_tokens),
                    1 if p.temperature == 0 else min(p.top_k or vocab, vocab),
                )
            )
        if updates:
            update_history(upload(updates, device=device, dtype=torch.int64))
        return (
            # sampling parameters, shape (batch, 5)
            upload(parameters, device=device, dtype=torch.float32),
            # metadata, shape (batch, 3), including: history buffer pointer, min_tokens flag, top_k
            upload(metadata, device=device, dtype=torch.int64),
        )

    @torch.inference_mode()
    def sample(
        self, logits: Mapping[str, torch.Tensor], requests: Sequence[Request], eos_token_ids: tuple[int, ...] = ()
    ) -> dict[str, Sample]:
        if not requests:
            return {}
        device = logits[requests[0].request_id].device
        if device.type != "cuda":
            raise ValueError("sampling requires CUDA logits")
        if self.profile_enabled:
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        # shape (batch, vocab)
        scores = torch.stack([logits[request.request_id] for request in requests])
        # get sampling parameters and metadata for each request
        params, metadata = self._metadata(requests, scores.shape[1], device, tuple(eos_token_ids))
        draws = []
        for request in requests:
            if request.rng is None:
                request.rng = torch.Generator(device=device)
                if request.sampling_params.seed is None:
                    request.rng.seed()
                else:
                    request.rng.manual_seed(request.sampling_params.seed % (1 << 64))
            draws.append(torch.rand((1,), device=device, generator=request.rng))
        # sample from the fused kernel
        result = sample_sorted(scores, params, metadata, torch.cat(draws))
        if self.profile_enabled:
            torch.cuda.synchronize(device)
            self.stage_seconds["sampling"] += time.perf_counter() - started
        started = time.perf_counter()
        rows = result.cpu().tolist()
        # Bound cached histories even when sampled requests are later preempted.
        # Evict only after the kernels have finished reading their pointer metadata.
        while len(self.histories) > self.max_histories:
            self.histories.pop(next(iter(self.histories)))
        self.transfer_bytes += result.numel() * result.element_size()
        if self.profile_enabled:
            self.stage_seconds["transfer"] += time.perf_counter() - started
        if any(error for _, _, error in rows):
            raise ValueError("invalid or all-masked sampling distribution")
        return {
            request.request_id: Sample(int(token), logprob)
            for request, (token, logprob, _) in zip(requests, rows, strict=True)
        }
