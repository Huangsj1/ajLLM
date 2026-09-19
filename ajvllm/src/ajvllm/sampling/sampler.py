"""Readable CPU sampling oracle. GPU sampling is a later backend concern."""

import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ajvllm.sampling.params import SamplingParams


@dataclass(frozen=True)
class Sample:
    token_id: int
    logprob: float


class Sampler:
    def sample(
        self,
        logits: Sequence[float],  # (vocab_size,) logits for the next token
        params: SamplingParams,
        rng: random.Random,
        prompt_token_ids: Sequence[int] = (),
        output_token_ids: Sequence[int] = (),
        eos_token_ids: tuple[int, ...] = (),
    ) -> Sample:
        scores = [float(value) for value in logits]
        if not scores or any(math.isnan(value) or value == math.inf for value in scores):
            raise ValueError("logits must be nonempty and contain no NaN or positive infinity")
        # 1.penalize repetition and presence/frequency of history tokens
        for token in set(prompt_token_ids) | set(output_token_ids):
            value = scores[token]
            scores[token] = value / params.repetition_penalty if value > 0 else value * params.repetition_penalty
        for token, count in Counter(output_token_ids).items():
            scores[token] -= params.presence_penalty + params.frequency_penalty * count
        # 2. mask EOS and stop tokens if min_tokens not yet reached
        if len(output_token_ids) < params.min_tokens:
            for token in params.effective_stop_ids(eos_token_ids):
                if 0 <= token < len(scores):
                    scores[token] = -math.inf
        if any(math.isnan(value) or value == math.inf for value in scores):
            raise ValueError("sampling penalties produced non-finite scores")
        # 3. sort candidates by score, then sample according to temperature, top-k, and top-p
        candidates = sorted((i for i, value in enumerate(scores) if value != -math.inf), key=lambda i: (-scores[i], i))
        if not candidates:
            raise ValueError("all logits are masked")
        if params.temperature == 0:
            return Sample(candidates[0], 0.0)
        if params.top_k:
            candidates = candidates[: params.top_k]
        # 4. use temperature, top-k, and top-p to sample a token from the candidates
        # Subtract before dividing to avoid overflow for very small temperatures.
        maximum = scores[candidates[0]]
        weights = [math.exp((scores[i] - maximum) / params.temperature) for i in candidates]
        total = math.fsum(weights)
        if params.top_p < 1:
            cumulative = 0.0
            for index, weight in enumerate(weights):
                cumulative += weight
                if cumulative >= params.top_p * total:
                    candidates = candidates[: index + 1]
                    weights = weights[: index + 1]
                    break
            total = math.fsum(weights)
        draw = rng.random() * total
        cumulative = 0.0
        for token, weight in zip(candidates, weights, strict=True):
            cumulative += weight
            if draw < cumulative:
                return Sample(token, math.log(weight) - math.log(total))
        # Rounding at the final cumulative sum: choose the last nonzero mass.
        index = max(i for i, weight in enumerate(weights) if weight > 0)
        return Sample(candidates[index], math.log(weights[index]) - math.log(total))
