"""A persistent async control loop with one dedicated model-execution thread."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import torch

from ajvllm import Engine, EngineExecutionError, SamplingParams
from ajvllm.runtime.budget import MemoryBudget


class ServiceBusy(RuntimeError):
    pass


@dataclass
class Command:
    kind: str
    request_id: str
    payload: object
    reply: asyncio.Future


class RequestStream:
    def __init__(self, service, request_id: str, capacity: int):
        self.service = service
        self.request_id = request_id
        self.queue = asyncio.Queue(maxsize=capacity)

    async def next(self):
        result = await self.queue.get()
        if isinstance(result, Exception):
            raise result
        return result

    async def events(self):
        try:
            while True:
                output = await self.next()
                yield output
                if output.finished:
                    break
        finally:
            await self.service.cancel(self.request_id, stream=self)


class EngineService:
    def __init__(
        self, engine: Engine, budget: MemoryBudget | None = None, *, max_pending_requests=128, stream_capacity=128
    ):
        if max_pending_requests < 1 or stream_capacity < 1:
            raise ValueError("queue limits must be positive")
        self.engine = engine
        self.budget = budget
        self.max_pending_requests = max_pending_requests
        self.stream_capacity = stream_capacity
        self.commands = asyncio.Queue(maxsize=max_pending_requests)
        self.streams: dict[str, RequestStream] = {}
        self.reserved_ids: set[str] = set()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ajvllm-gpu")
        self.task = None
        self.closing = False
        self.failure: str | None = None
        self.stats = {}

    async def start(self):
        if self.task is None:
            self.task = asyncio.create_task(self._run())
        return self

    async def submit(self, request_id: str, token_ids, params: SamplingParams) -> RequestStream:
        if self.closing or self.failure or self.task is None:
            raise RuntimeError("service is not running")
        if request_id in self.reserved_ids:
            raise ValueError("duplicate active request ID")
        if len(self.reserved_ids) >= self.max_pending_requests:
            raise ServiceBusy("request capacity is full")
        self.reserved_ids.add(request_id)
        reply = asyncio.get_running_loop().create_future()
        try:
            self.commands.put_nowait(Command("add", request_id, (tuple(token_ids), params), reply))
            return await asyncio.shield(reply)
        except asyncio.CancelledError:
            await self.cancel(request_id)
            raise
        except Exception:
            self.reserved_ids.discard(request_id)
            raise

    async def cancel(self, request_id: str, *, stream: RequestStream | None = None):
        if stream is not None and self.streams.get(request_id) is not stream:
            return None
        if self.closing or self.failure or request_id not in self.reserved_ids:
            return None
        reply = asyncio.get_running_loop().create_future()
        await self.commands.put(Command("cancel", request_id, None, reply))
        return await asyncio.shield(reply)

    async def close(self):
        if not self.closing:
            self.closing = True
            if self.task is not None and not self.task.done():
                reply = asyncio.get_running_loop().create_future()
                await self.commands.put(Command("close", "", None, reply))
        if self.task is not None:
            await self.task
        self.executor.shutdown(wait=True)

    def _command(self, command: Command) -> bool:
        try:
            if command.kind == "close":
                command.reply.set_result(None)
                return False
            if command.kind == "add":
                tokens, params = command.payload
                self.engine.add_request(command.request_id, tokens, params)
                stream = RequestStream(self, command.request_id, self.stream_capacity)
                self.streams[command.request_id] = stream
                command.reply.set_result(stream)
            else:
                output = self.engine.cancel_request(command.request_id)
                if output is not None:
                    self._publish(output)
                command.reply.set_result(output)
        except Exception as exc:
            self.reserved_ids.discard(command.request_id)
            command.reply.set_exception(exc)
        return True

    def _publish(self, output):
        stream = self.streams.get(output.request_id)
        if stream is None:
            return
        if stream.queue.full():
            # A slow consumer cannot retain an unbounded output history or KV cache.
            self.engine.cancel_request(output.request_id)
            while not stream.queue.empty():
                stream.queue.get_nowait()
            stream.queue.put_nowait(ServiceBusy("output consumer is too slow; request cancelled"))
            self.streams.pop(output.request_id, None)
            self.reserved_ids.discard(output.request_id)
            return
        stream.queue.put_nowait(output)
        if output.finished:
            self.streams.pop(output.request_id, None)
            self.reserved_ids.discard(output.request_id)

    def _step(self):
        if self.budget:
            self.budget.before_step(self.engine)
        try:
            outputs = self.engine.step()
        except EngineExecutionError as exc:
            if self.budget and isinstance(exc.__cause__, torch.cuda.OutOfMemoryError):
                self.budget.on_oom()
            outputs = list(exc.outputs)
        if self.budget:
            self.budget.after_step(self.engine)
        return outputs

    async def _run(self):
        loop = asyncio.get_running_loop()
        try:
            running = True
            while running:
                if not self.engine.has_unfinished_requests and self.commands.empty():
                    running = self._command(await self.commands.get())
                # Bound admission work so a flood cannot starve model execution.
                for _ in range(self.max_pending_requests):
                    if not running or self.commands.empty():
                        break
                    running = self._command(self.commands.get_nowait())
                if not running:
                    break
                if self.engine.has_unfinished_requests:
                    for output in await loop.run_in_executor(self.executor, self._step):
                        self._publish(output)
                    self.stats = {
                        "engine": asdict(self.engine.metrics),
                        "active_requests": self.engine.num_unfinished_requests,
                        "token_budget": self.engine.token_budget,
                        "memory": self.budget.snapshot() if self.budget else None,
                    }
                await asyncio.sleep(0)
        except Exception as exc:
            self.failure = str(exc)
            for stream in self.streams.values():
                while not stream.queue.empty():
                    stream.queue.get_nowait()
                stream.queue.put_nowait(RuntimeError(f"engine service failed: {exc}"))
            self.streams.clear()
        finally:
            for output in self.engine.close():
                self._publish(output)
            self.reserved_ids.clear()
            while not self.commands.empty():
                command = self.commands.get_nowait()
                if not command.reply.done():
                    command.reply.set_exception(RuntimeError("service stopped"))
