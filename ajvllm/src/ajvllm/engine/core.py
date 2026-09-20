"""Single-threaded engine loop with explicit runner failure and cleanup semantics."""

import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace

from ajvllm.config import EngineConfig, require_int
from ajvllm.engine.metrics import EngineMetrics
from ajvllm.execution.protocol import ModelRunner
from ajvllm.requests import FinishReason, Request, RequestOutput, RequestStatus
from ajvllm.sampling.params import SamplingParams
from ajvllm.sampling.sampler import Sampler
from ajvllm.scheduling.batch import Phase, SchedulerOutput
from ajvllm.scheduling.scheduler import Scheduler


class EngineExecutionError(RuntimeError):
    """Terminal error events for the failed batch are available in outputs."""

    def __init__(self, outputs: tuple[RequestOutput, ...]):
        super().__init__("runner execution or sampling failed; affected requests were released")
        self.outputs = outputs


class Engine:
    def __init__(
        self,
        runner: ModelRunner,
        config: EngineConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.runner = runner
        self.config = config or EngineConfig()
        model_limit = getattr(runner, "max_model_len", None)
        if model_limit is not None and self.config.max_model_len > model_limit:
            raise ValueError("engine max_model_len exceeds the runner's checkpoint context length")
        self.eos_token_ids = tuple(runner.eos_token_ids)
        self._scheduler = Scheduler(self.config)
        self._sampler = Sampler()
        self._clock = clock
        self._metrics = EngineMetrics()
        self._pending_outputs: deque[RequestOutput] = deque()  # unexpected outputs
        self._closed = False
        self.last_batch = SchedulerOutput()

    def _validate_tokens(self, tokens: Iterable[int]) -> None:
        for token in tokens:
            require_int("token ID", token)
            if token >= self.runner.vocab_size:
                raise ValueError(f"token ID {token} is outside runner vocabulary")

    @property
    def metrics(self) -> EngineMetrics:
        return replace(self._metrics)

    @property
    def num_unfinished_requests(self) -> int:
        return len(self._scheduler.requests)

    @property
    def has_unfinished_requests(self) -> bool:
        # Include undelivered zero-token terminal outputs so run() drains them.
        return bool(self._scheduler.requests or self._pending_outputs)

    @property
    def token_budget(self) -> int:
        return self._scheduler.token_budget

    def set_token_budget(self, budget: int) -> None:
        if not 1 <= budget <= self.config.max_num_batched_tokens:
            raise ValueError("token budget must stay within the configured ceiling")
        if not self.config.enable_chunked_prefill and budget < self.config.max_model_len:
            raise ValueError("unchunked prefill requires a full-context budget")
        self._scheduler.token_budget = budget

    def add_request(
        self, request_id: str, prompt_token_ids: Iterable[int], sampling_params: SamplingParams | None = None
    ) -> None:
        if self._closed:
            raise RuntimeError("engine is closed")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")
        if request_id in self._scheduler.requests or any(o.request_id == request_id for o in self._pending_outputs):
            raise ValueError(f"duplicate active request ID: {request_id}")
        params = sampling_params if sampling_params is not None else SamplingParams()
        if not isinstance(params, SamplingParams):
            raise TypeError("sampling_params must be SamplingParams")
        tokens = tuple(prompt_token_ids)
        if not tokens or len(tokens) > self.config.max_model_len:
            raise ValueError("prompt must be nonempty and fit max_model_len")
        self._validate_tokens(tokens)
        self._validate_tokens(params.stop_token_ids)
        request = Request(request_id, tokens, params, self._clock())
        self._scheduler.add(request)
        if params.max_tokens == 0 or len(tokens) == self.config.max_model_len:
            self._pending_outputs.append(self._finish(request, FinishReason.LENGTH))

    def _output(
        self,
        request: Request,
        new_token_ids: tuple[int, ...] = (),
        finish_reason: FinishReason | None = None,
        stop_token_id: int | None = None,
        logprob: float | None = None,
        error: str | None = None,
    ) -> RequestOutput:
        return RequestOutput(
            request.request_id,
            request.prompt_token_ids,
            tuple(request.output_token_ids),
            new_token_ids,
            request.status,
            finish_reason,
            stop_token_id,
            logprob,
            request.arrival_time,
            request.first_token_time,
            request.finish_time,
            error,
        )

    def _finish(
        self,
        request: Request,
        reason: FinishReason,
        new_token_ids: tuple[int, ...] = (),
        stop_token_id: int | None = None,
        logprob: float | None = None,
        error: str | None = None,
    ) -> RequestOutput:
        if reason == FinishReason.CANCELLED:
            request.status = RequestStatus.CANCELLED
            self._metrics.cancelled_requests += 1
        elif reason == FinishReason.ERROR:
            request.status = RequestStatus.FAILED
            self._metrics.failed_requests += 1
        else:
            request.status = RequestStatus.FINISHED
            self._metrics.finished_requests += 1
        request.finish_time = self._clock()
        self._scheduler.remove(request.request_id)
        self.runner.release(request.request_id)
        return self._output(request, new_token_ids, reason, stop_token_id, logprob, error)

    def cancel_request(self, request_id: str) -> RequestOutput | None:
        request = self._scheduler.requests.get(request_id)
        return self._finish(request, FinishReason.CANCELLED) if request is not None else None

    def step(self) -> list[RequestOutput]:
        if self._closed:
            raise RuntimeError("engine is closed")
        # 1. build the current batch of scheduled requests
        self.last_batch = batch = self._scheduler.schedule()
        if not batch.requests:
            outputs = list(self._pending_outputs)
            self._pending_outputs.clear()
            return outputs
        self._metrics.num_steps += 1
        self._metrics.scheduled_prefill_tokens += sum(i.num_tokens for i in batch.requests if i.phase == Phase.PREFILL)
        self._metrics.scheduled_decode_tokens += sum(i.num_tokens for i in batch.requests if i.phase == Phase.DECODE)
        try:
            # 2. execute the model
            logits = self.runner.execute(batch)
            ready = [self._scheduler.requests[item.request_id] for item in batch.requests if item.do_sample]
            if set(logits) != {request.request_id for request in ready}:
                raise ValueError("runner logits keys do not match sampling-ready request IDs")
            if any(row.shape != (self.runner.vocab_size,) for row in logits.values()):
                raise ValueError("runner logits width does not match vocabulary")
            # 3. sample the next token for each request that is ready to decode
            samples = self._sampler.sample(logits, ready, self.eos_token_ids)
        except Exception as exc:
            errors = tuple(
                self._finish(self._scheduler.requests[item.request_id], FinishReason.ERROR, error=str(exc))
                for item in batch.requests
            )
            raise EngineExecutionError(errors) from exc
        outputs = list(self._pending_outputs)
        self._pending_outputs.clear()
        for item in batch.requests:
            request = self._scheduler.requests[item.request_id]
            request.num_computed_tokens += item.num_tokens
            # if in prefill phase and not yet finished the whole input, continue
            if not item.do_sample:
                continue
            sample = samples[item.request_id]
            request.output_token_ids.append(sample.token_id)
            self._metrics.generated_tokens += 1
            if request.first_token_time is None:
                request.first_token_time = self._clock()
            reason = None
            stop_token = None
            if sample.token_id in request.sampling_params.effective_stop_ids(self.eos_token_ids):
                reason, stop_token = FinishReason.STOP, sample.token_id
            elif (
                len(request.output_token_ids) >= request.sampling_params.max_tokens
                or request.num_tokens >= self.config.max_model_len
            ):
                reason = FinishReason.LENGTH
            if reason is not None:
                # finish the request and release its resources
                outputs.append(self._finish(request, reason, (sample.token_id,), stop_token, sample.logprob))
            else:
                outputs.append(self._output(request, (sample.token_id,), logprob=sample.logprob))
        return outputs

    def run(self) -> Iterator[RequestOutput]:
        while self.has_unfinished_requests:
            yield from self.step()

    def close(self) -> list[RequestOutput]:
        outputs = list(self._pending_outputs)
        self._pending_outputs.clear()
        for request in list(self._scheduler.requests.values()):
            outputs.append(self._finish(request, FinishReason.CANCELLED))
        self._closed = True
        return outputs
