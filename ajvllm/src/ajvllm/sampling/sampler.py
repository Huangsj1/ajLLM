"""Batched CUDA sampling. Only selected tokens, log probabilities and status leave the device."""

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ajvllm.requests import Request


@dataclass(frozen=True)
class Sample:
    token_id: int
    logprob: float


class Sampler:
    def __init__(self):
        self.profile_enabled = False
        self.stage_seconds = {"sampling": 0.0, "transfer": 0.0}
        self.transfer_bytes = 0

    @torch.inference_mode()
    def sample(
        self, logits: Mapping[str, torch.Tensor], requests: Sequence[Request], eos_token_ids: tuple[int, ...] = ()
    ) -> dict[str, Sample]:
        if not requests:
            return {}
        device = logits[requests[0].request_id].device
        if self.profile_enabled:
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        scores = torch.stack([logits[request.request_id] for request in requests]).float()
        if not scores.is_cuda:
            raise ValueError("sampling requires CUDA logits")
        batch_size, vocab_size = scores.shape
        policies = [request.sampling_params for request in requests]
        invalid = (torch.isnan(scores) | torch.isposinf(scores)).any(dim=1)

        # History is control-plane metadata; counts and all score transforms live on CUDA.
        seen = torch.zeros_like(scores, dtype=torch.bool)
        counts = torch.zeros_like(scores)
        history_indices, generated_indices, stop_indices = [], [], []
        for row, request in enumerate(requests):
            history_indices.extend(row * vocab_size + token for token in request.token_ids)
            generated_indices.extend(row * vocab_size + token for token in request.output_token_ids)
            if len(request.output_token_ids) < request.sampling_params.min_tokens:
                stop_indices.extend(
                    row * vocab_size + token
                    for token in request.sampling_params.effective_stop_ids(eos_token_ids)
                    if token < vocab_size
                )
        history = torch.tensor(history_indices, device=device, dtype=torch.long)
        generated = torch.tensor(generated_indices, device=device, dtype=torch.long)
        seen.view(-1).scatter_(0, history, True)
        counts.view(-1).scatter_add_(0, generated, torch.ones(generated.shape, device=device))
        repetition = torch.tensor([p.repetition_penalty for p in policies], device=device)[:, None]
        presence = torch.tensor([p.presence_penalty for p in policies], device=device)[:, None]
        frequency = torch.tensor([p.frequency_penalty for p in policies], device=device)[:, None]
        # repeation penalty and presence/frequency penalties
        penalized = torch.where(scores > 0, scores / repetition, scores * repetition)
        scores = torch.where(seen, penalized, scores) - presence * (counts > 0) - frequency * counts
        invalid |= (torch.isnan(scores) | torch.isposinf(scores)).any(dim=1)
        stops = torch.tensor(stop_indices, device=device, dtype=torch.long)
        scores.view(-1).index_fill_(0, stops, -torch.inf)

        # All rows use the same distribution path. Temperature zero is top-k=1;
        # stable sorting resolves tied scores in ascending token-ID order.
        values, token_ids = scores.sort(dim=1, descending=True, stable=True)
        invalid |= ~torch.isfinite(values[:, 0])
        values = values.masked_fill(invalid[:, None], 0)
        temperatures = torch.tensor([p.temperature if p.temperature else 1.0 for p in policies], device=device)[:, None]
        temperatures = temperatures.clamp_min(torch.finfo(torch.float32).tiny)
        scaled = (values - values[:, :1]) / temperatures
        scaled = scaled.masked_fill(torch.isneginf(values), -torch.inf)
        limits = torch.tensor([1 if p.temperature == 0 else p.top_k or vocab_size for p in policies], device=device)
        ranks = torch.arange(vocab_size, device=device)[None, :]
        scaled.masked_fill_(ranks >= limits[:, None], -torch.inf)
        probabilities = scaled.softmax(dim=1)
        previous_mass = torch.cat((torch.zeros((batch_size, 1), device=device), probabilities.cumsum(1)[:, :-1]), dim=1)
        top_p = torch.tensor([p.top_p for p in policies], device=device)[:, None]
        excluded = (top_p < 1) & (previous_mass >= top_p) & (ranks > 0)
        probabilities.masked_fill_(excluded, 0)
        probabilities /= probabilities.sum(dim=1, keepdim=True)
        cumulative = probabilities.cumsum(dim=1)
        cumulative /= cumulative[:, -1:].clone()

        draws = []
        for request in requests:
            if request.rng is None:
                request.rng = torch.Generator(device=device)
                if request.sampling_params.seed is None:
                    request.rng.seed()
                else:
                    request.rng.manual_seed(request.sampling_params.seed % (1 << 64))
            draws.append(torch.rand((1,), device=device, generator=request.rng))
        indices = torch.searchsorted(cumulative, torch.stack(draws), right=True).clamp_max(vocab_size - 1)
        tokens = token_ids.gather(1, indices).squeeze(1)
        logprobs = probabilities.gather(1, indices).squeeze(1).log()
        invalid |= ~torch.isfinite(logprobs)
        # Float64 represents integer vocabulary IDs exactly; one compact batched transfer.
        result = torch.stack((tokens.double(), logprobs.double(), invalid.double()), dim=1)
        if self.profile_enabled:
            torch.cuda.synchronize(device)
            self.stage_seconds["sampling"] += time.perf_counter() - started
        started = time.perf_counter()
        rows = result.cpu().tolist()
        self.transfer_bytes += result.numel() * result.element_size()
        if self.profile_enabled:
            self.stage_seconds["transfer"] += time.perf_counter() - started
        if any(error for _, _, error in rows):
            raise ValueError("invalid or all-masked sampling distribution")
        return {
            request.request_id: Sample(int(token), logprob)
            for request, (token, logprob, _) in zip(requests, rows, strict=True)
        }
