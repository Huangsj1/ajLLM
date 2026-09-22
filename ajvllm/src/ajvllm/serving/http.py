"""Local HTTP generation and SSE endpoints; one process owns one engine."""

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import asdict
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from ajvllm import SamplingParams
from ajvllm.serving.presentation import memory_display, request_timing
from ajvllm.serving.service import EngineService, ServiceBusy
from ajvllm.tokenization.qwen2 import Qwen2Tokenizer


class GenerationInput(BaseModel):
    request_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    prompt: str | None = None
    messages: list[dict[str, str]] | None = None
    token_ids: list[int] | None = None
    sampling: dict = Field(default_factory=dict)
    stream: bool = False
    cache_salt: str = ""

    @model_validator(mode="after")
    def one_input(self):
        if sum(value is not None for value in (self.prompt, self.messages, self.token_ids)) != 1:
            raise ValueError("provide exactly one of prompt, messages, or token_ids")
        return self


def create_app(service: EngineService, tokenizer: Qwen2Tokenizer) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        await service.start()
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(title="ajvLLM", lifespan=lifespan)

    @app.get("/health")
    async def health():
        if service.failure or service.closing:
            raise HTTPException(503, service.failure or "service is stopping")
        return {"status": "ready", **memory_display(service.stats)}

    @app.delete("/requests/{request_id}")
    async def cancel(request_id: str):
        output = await service.cancel(request_id)
        return {"request_id": request_id, "cancelled": output is not None}

    def encode(body):
        if body.token_ids is not None:
            return body.token_ids
        if body.messages is not None:
            return tokenizer.encode_chat(body.messages)
        return tokenizer.encode(body.prompt)

    def serialize(output):
        data = asdict(output)
        for key in ("arrival_time", "first_token_time", "finish_time"):
            data.pop(key)
        return data | {"text": tokenizer.decode(output.output_token_ids), "timing": request_timing(output)}

    @app.post("/generate")
    async def generate(body: GenerationInput, request: Request):
        try:
            params = SamplingParams(**body.sampling)
            ids = await asyncio.to_thread(encode, body)
            handle = await service.submit(body.request_id, ids, params, cache_salt=body.cache_salt)
        except ServiceBusy as exc:
            raise HTTPException(429, str(exc)) from exc
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        if body.stream:

            async def events():
                try:
                    async for output in handle.events():
                        yield f"data: {json.dumps(serialize(output), ensure_ascii=False)}\n\n"
                except Exception as exc:
                    yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")
        try:
            while True:
                try:
                    output = await asyncio.wait_for(handle.next(), timeout=0.2)
                except TimeoutError:
                    if await request.is_disconnected():
                        raise HTTPException(499, "client disconnected")
                    continue
                if output.finished:
                    if output.error:
                        raise HTTPException(500, output.error)
                    return serialize(output)
        finally:
            await service.cancel(body.request_id, stream=handle)

    return app
