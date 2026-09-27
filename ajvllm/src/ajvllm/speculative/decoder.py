"""Batched draft rollout, one packed target verification, and transactional KV commit."""

import math
from dataclasses import replace

import torch

from ajvllm.execution.step import StepResult
from ajvllm.execution.transfer import upload
from ajvllm.sampling.sampler import Sample, Sampler
from ajvllm.scheduling.batch import Phase, ScheduledRequest, SchedulerOutput
from ajvllm.speculative.sampling import categorical, uniforms, verify


class SpeculativeDecoder:
    def __init__(self, target, draft, num_draft_tokens):
        self.runner = target
        self.draft = draft
        self.num_draft_tokens = num_draft_tokens
        self.draft_sampler = Sampler()
        self.counters = dict(
            rounds=0,
            proposed_tokens=0,
            accepted_tokens=0,
            emitted_tokens=0,
            target_forwards=0,
            draft_forwards=0,
            verification_tokens=0,
        )

    def snapshot(self):
        c = self.counters
        return dict(
            enabled=True,
            num_draft_tokens=self.num_draft_tokens,
            **c,
            acceptance_rate=c["accepted_tokens"] / max(1, c["proposed_tokens"]),
            tokens_per_round=c["emitted_tokens"] / max(1, c["rounds"]),
            draft_kv_cache=self.draft.memory_stats(),
            draft_stage_seconds=dict(self.draft.stage_seconds) | dict(self.draft_sampler.stage_seconds),
            draft_graphs=self.draft.graphs.snapshot() if self.draft.graphs else {"enabled": False},
        )

    def _draft(self, items):
        if not items:
            return {}
        self.counters["draft_forwards"] += 1
        return self.draft.execute(SchedulerOutput(tuple(items)), commit=False)

    @torch.inference_mode()
    def execute(self, batch, requests, sampler, eos):
        self.draft.profile_enabled = self.runner.profile_enabled
        self.draft_sampler.profile_enabled = sampler.profile_enabled
        items = {item.request_id: item for item in batch.requests}
        speculative = [item for item in batch.requests if item.num_draft_tokens]
        shadows = {
            item.request_id: replace(
                requests[item.request_id], output_token_ids=list(requests[item.request_id].output_token_ids)
            )
            for item in speculative
        }
        proposals = {rid: [] for rid in shadows}
        q = {rid: [] for rid in shadows}
        target_shadows = {}
        try:
            # draft model batch forward(including prefill and decode phases) to get the logits for all draft tokens(including kv cache)
            logits = self._draft(
                [
                    replace(
                        item, do_sample=bool(item.num_draft_tokens) or item.phase == Phase.DECODE, num_draft_tokens=0
                    )
                    for item in batch.requests
                ]
            )
            # draft model batch forward(only for decode phase) to get the logits for all draft tokens(including kv cache)
            for depth in range(max((i.num_draft_tokens for i in speculative), default=0)):
                active = [item for item in speculative if item.num_draft_tokens > depth]
                pending = [shadows[i.request_id] for i in active]
                probs = self.draft_sampler.distributions(logits, pending, eos)
                tokens = categorical(probs, uniforms(pending, 1, probs.device)[:, 0])
                # Only selected draft IDs and validity reach the host, never vocabulary logits.
                metadata = torch.stack((tokens.double(), probs.sum(-1).double()), -1).cpu().tolist()
                next_items = []
                for index, (item, (token, mass)) in enumerate(zip(active, metadata, strict=True)):
                    if not 0.99 < mass < 1.01:
                        raise ValueError("invalid draft distribution")
                    rid, token = item.request_id, int(token)
                    proposals[rid].append(token)
                    q[rid].append(probs[index])
                    shadows[rid].output_token_ids.append(token)
                    if item.num_draft_tokens > depth + 1:
                        next_items.append(ScheduledRequest(rid, (token,), item.end_pos + depth, Phase.DECODE, True))
                logits = self._draft(next_items)

            verification = SchedulerOutput(
                tuple(
                    replace(
                        item, token_ids=item.token_ids + tuple(proposals.get(item.request_id, ())), num_draft_tokens=0
                    )
                    for item in batch.requests
                )
            )
            # target model batch forward(including prefill and decode phases) to get the logits for all draft tokens(including kv cache)
            logits = self.runner.execute(verification, commit=False, sample_all=frozenset(shadows))
            self.counters["target_forwards"] += 1
            ordinary = [requests[i.request_id] for i in batch.requests if i.do_sample and not i.num_draft_tokens]
            samples = {rid: (sample,) for rid, sample in sampler.sample(logits, ordinary, eos).items()}
            computed = {item.request_id: item.num_tokens for item in batch.requests}
            p = {rid: [] for rid in shadows}
            target_shadows = {
                rid: replace(requests[rid], output_token_ids=list(requests[rid].output_token_ids)) for rid in shadows
            }
            for depth in range(max((i.num_draft_tokens + 1 for i in speculative), default=0)):
                active = [i for i in speculative if i.num_draft_tokens >= depth]
                pending = [target_shadows[i.request_id] for i in active]
                probs = sampler.distributions({i.request_id: logits[i.request_id][depth] for i in active}, pending, eos)
                for index, item in enumerate(active):
                    rid = item.request_id
                    p[rid].append(probs[index])
                    if depth < item.num_draft_tokens:
                        target_shadows[rid].output_token_ids.append(proposals[rid][depth])
            flush, results, counts = [], [], []
            ordered = []
            for k in sorted({item.num_draft_tokens for item in speculative}):
                group = [item for item in speculative if item.num_draft_tokens == k]
                ids = [item.request_id for item in group]
                result, accepted = verify(
                    torch.stack([torch.stack(p[rid]) for rid in ids]),
                    torch.stack([torch.stack(q[rid]) for rid in ids]),
                    upload([proposals[rid] for rid in ids], device=self.runner.model.device),
                    uniforms([requests[rid] for rid in ids], k + 1, self.runner.model.device),
                )
                results.append(result.flatten(0, 1))
                counts.append(accepted)
                ordered.extend(group)
            if results:
                lengths = torch.cat(counts).cpu().tolist()
                rows = torch.cat(results).cpu().tolist()
                offset = 0
                for item, accepted in zip(ordered, lengths, strict=True):
                    rid = item.request_id
                    chosen = rows[offset : offset + accepted + 1]
                    offset += item.num_draft_tokens + 1
                    stops = requests[rid].sampling_params.effective_stop_ids(eos)
                    emitted = []
                    for token, logprob in chosen:
                        if not math.isfinite(logprob):
                            raise ValueError("invalid target or residual distribution")
                        emitted.append(Sample(int(token), logprob))
                        if int(token) in stops:
                            break
                    samples[rid] = tuple(emitted)
                    computed[rid] = len(emitted)
                    if len(emitted) == item.num_draft_tokens + 1:
                        flush.append(
                            ScheduledRequest(
                                rid, (proposals[rid][-1],), item.start_pos + item.num_draft_tokens, Phase.DECODE, True
                            )
                        )
                    self.counters["rounds"] += 1
                    self.counters["proposed_tokens"] += item.num_draft_tokens
                    self.counters["accepted_tokens"] += min(accepted, len(emitted))
                    self.counters["emitted_tokens"] += len(emitted)
                    self.counters["verification_tokens"] += item.num_draft_tokens + 1
            # All-accepted transactions need the last draft token's KV before publication.
            self._draft(flush)
            for item in verification.requests:
                rid = item.request_id
                # commit tokens and prefix cache for all accepted draft tokens, including the last one if it was accepted
                self.runner.kv_cache.commit(rid, item.token_ids[: computed[rid]])
                if items[rid].phase == Phase.DECODE:
                    # trim the rejected draft tokens from the KV cache
                    self.runner.kv_cache.trim(rid)
            return StepResult(samples, computed)
        finally:
            for request in shadows.values():
                self.draft_sampler.release(request)
            for request in target_shadows.values():
                sampler.release(request)
