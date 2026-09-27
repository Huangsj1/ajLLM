"""Decoding strategies return committed progress and one or more output tokens."""

from dataclasses import dataclass

from ajvllm.sampling.sampler import Sample


@dataclass
class StepResult:
    samples: dict[str, tuple[Sample, ...]]
    computed: dict[str, int]


class AutoregressiveDecoder:
    num_draft_tokens = 0

    def __init__(self, runner):
        self.runner = runner

    def execute(self, batch, requests, sampler, eos):
        logits = self.runner.execute(batch)
        ready = [requests[item.request_id] for item in batch.requests if item.do_sample]
        if set(logits) != {request.request_id for request in ready}:
            raise ValueError("runner logits keys do not match sampling-ready request IDs")
        if any(row.shape != (self.runner.vocab_size,) for row in logits.values()):
            raise ValueError("runner logits width does not match vocabulary")
        samples = sampler.sample(logits, ready, eos)
        return StepResult(
            {rid: (sample,) for rid, sample in samples.items()},
            {item.request_id: item.num_tokens for item in batch.requests},
        )

    def snapshot(self):
        return {"enabled": False}
