"""Real CUDA service tests: arrivals in flight, backpressure, HTTP streaming and shutdown."""

import asyncio
import json
import socket
import threading

import httpx
import pytest
import torch
import uvicorn
from test_qwen2_cuda import pair, tiny_config

from ajvllm import Engine, EngineConfig, SamplingParams
from ajvllm.config.memory import MemoryConfig
from ajvllm.execution.qwen2 import Qwen2Runner
from ajvllm.runtime.inference import InferenceRuntime
from ajvllm.serving.http import create_app
from ajvllm.serving.service import EngineService, ServiceBusy
from ajvllm.tokenization.qwen2 import Qwen2Tokenizer

pytestmark = pytest.mark.cuda


@pytest.fixture(params=["contiguous", "paged"])
def runner(request):
    assert torch.cuda.is_available()
    model, _ = pair(tiny_config())
    runner = Qwen2Runner(
        model,
        memory_config=MemoryConfig(backend=request.param, block_size=4),
        engine_config=EngineConfig(max_model_len=128, max_num_seqs=3),
    )
    return runner


def make_service(runner, **kwargs):
    engine = Engine(
        runner, EngineConfig(max_model_len=128, max_num_seqs=3, max_num_batched_tokens=4, max_prefill_chunk_size=2)
    )
    runtime = InferenceRuntime(engine)
    service = EngineService(runtime, **kwargs)
    assert service.runtime is runtime
    return service


def test_live_arrival_during_forward_and_idle_restart(runner):
    async def scenario():
        service = await make_service(runner).start()
        entered, resume = threading.Event(), threading.Event()
        batches = []

        def hook(_, args):
            batches.append(args[0].query_lengths)
            if len(batches) == 1:
                entered.set()
                assert resume.wait(5)

        handle = runner.model.register_forward_pre_hook(hook)
        try:
            first = await service.submit("long", [1] * 15, SamplingParams(max_tokens=6, temperature=0))
            assert await asyncio.to_thread(entered.wait, 5)
            arrival = asyncio.create_task(service.submit("late", [2], SamplingParams(max_tokens=6, temperature=0)))
            await asyncio.sleep(0)
            assert not service.commands.empty()  # HTTP/control loop is responsive during GPU execution.
            resume.set()
            late = await arrival

            async def collect(stream):
                return [output async for output in stream.events()]

            a, b = await asyncio.gather(collect(first), collect(late))
            assert a[-1].finished and b[-1].finished
            assert any(len(lengths) == 2 and 1 in lengths and 2 in lengths for lengths in batches)
            assert service.engine.num_unfinished_requests == 0
            again = await service.submit("after-idle", [3], SamplingParams(max_tokens=1, temperature=0))
            assert (await again.next()).finished
        finally:
            resume.set()
            handle.remove()
            await service.close()
        assert runner.cache_bytes == 0

    asyncio.run(scenario())


def test_backpressure_cancel_validation_and_shutdown(runner):
    async def scenario():
        service = await make_service(runner, max_pending_requests=1, stream_capacity=1).start()
        try:
            with pytest.raises(ValueError):
                await service.submit("bad", [], SamplingParams())
            stream = await service.submit("slow", [1] * 20, SamplingParams(max_tokens=30, temperature=0))
            with pytest.raises(ServiceBusy):
                await service.submit("overflow", [1], SamplingParams())
            for _ in range(300):
                if "slow" not in service.reserved_ids:
                    break
                await asyncio.sleep(0.01)
            with pytest.raises(ServiceBusy, match="slow"):
                await stream.next()
            assert runner.cache_bytes == 0
            stream = await service.submit("cancel", [1] * 100, SamplingParams(max_tokens=20))
            output = await service.cancel("cancel")
            assert output.finished and output.finish_reason == "cancelled"
            stream = await service.submit("shutdown", [1] * 100, SamplingParams(max_tokens=20))
            await service.close()
            events = [item async for item in stream.events()]
            assert events[-1].finished
        finally:
            await service.close()
        assert runner.cache_bytes == 0

    asyncio.run(scenario())


def test_old_stream_cannot_cancel_reused_request_id(runner):
    async def scenario():
        service = await make_service(runner).start()
        try:
            old = await service.submit("id", [1], SamplingParams(max_tokens=1, temperature=0))
            assert (await old.next()).finished
            new = await service.submit("id", [2] * 60, SamplingParams(max_tokens=4))
            assert await service.cancel("id", stream=old) is None
            assert "id" in service.reserved_ids
            await service.cancel("id", stream=new)
        finally:
            await service.close()

    asyncio.run(scenario())


def test_http_sse_disconnect_cancellation_and_concurrent_requests(runner):
    async def scenario():
        service = make_service(runner)
        # Only token IDs are submitted to the tiny GPU model. The real local tokenizer decodes them.
        app = create_app(service, Qwen2Tokenizer("model/Qwen2.5-0.5B-Instruct"))
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            for _ in range(200):
                if server.started:
                    break
                await asyncio.sleep(0.01)
            assert server.started
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10) as client:
                assert (await client.get("/health")).status_code == 200
                invalid = await client.post("/generate", json={"token_ids": [], "sampling": {"max_tokens": 1}})
                assert invalid.status_code == 422
                responses = await asyncio.gather(
                    *(
                        client.post(
                            "/generate",
                            json={
                                "request_id": str(i),
                                "token_ids": [i + 1] * (10 + i),
                                "sampling": {"max_tokens": 3, "temperature": 0},
                            },
                        )
                        for i in range(3)
                    )
                )
                assert all(
                    response.status_code == 200 and len(response.json()["output_token_ids"]) == 3
                    for response in responses
                )
                async with client.stream(
                    "POST",
                    "/generate",
                    json={
                        "request_id": "disconnect",
                        "token_ids": [1] * 12,
                        "stream": True,
                        "sampling": {"max_tokens": 100, "temperature": 0},
                    },
                ) as response:
                    assert response.status_code == 200
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            assert json.loads(line[6:])["new_token_ids"]
                            break  # Close the TCP stream while the model is still generating.
                for _ in range(200):
                    if not service.reserved_ids:
                        break
                    await asyncio.sleep(0.01)
                assert not service.reserved_ids
                assert service.engine.metrics.cancelled_requests >= 1 and runner.cache_bytes == 0
        finally:
            server.should_exit = True
            await task
            sock.close()
        assert runner.cache_bytes == 0

    asyncio.run(scenario())


def test_execution_failure_does_not_stop_service_and_waiting_cancel(runner):
    async def scenario():
        engine = Engine(runner, EngineConfig(max_model_len=128, max_num_seqs=1, max_num_batched_tokens=2))
        service = await EngineService(InferenceRuntime(engine)).start()
        calls = 0

        def fail_once(*_):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("injected batch failure")

        hook = runner.model.register_forward_hook(fail_once)
        try:
            bad = await service.submit("bad", [1], SamplingParams(max_tokens=1))
            error = await bad.next()
            assert error.finish_reason == "error"
            await service.submit("live", [1] * 100, SamplingParams(max_tokens=20))
            waiting = await service.submit("waiting", [2] * 100, SamplingParams(max_tokens=20))
            assert runner.cached_tokens("waiting") == 0
            await service.cancel("waiting")
            assert (await waiting.next()).finish_reason == "cancelled"
            await service.cancel("live")
            good = await service.submit("good", [2], SamplingParams(max_tokens=1))
            assert (await good.next()).finish_reason != "error"
            assert service.failure is None
        finally:
            hook.remove()
            await service.close()
        assert runner.cache_bytes == 0

    asyncio.run(scenario())


def test_benchmark_streaming_and_timing_on_cuda(runner):
    from ajvllm.serving.presentation import memory_display
    from ajvllm.workflows.benchmark import run_benchmark

    async def scenario():
        service = make_service(runner, profile_steps=True)
        app = create_app(service, Qwen2Tokenizer("model/Qwen2.5-0.5B-Instruct"))
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            for _ in range(200):
                if server.started:
                    break
                await asyncio.sleep(0.01)
            assert server.started
            report = await asyncio.to_thread(
                run_benchmark,
                f"http://127.0.0.1:{port}",
                [{"id": "short", "token_ids": [1, 2]}, {"id": "chunked", "token_ids": [3] * 9}],
                requests=4,
                concurrency=2,
                max_tokens=3,
                warmup=1,
                interval=0.1,
            )
            assert report["successful_requests"] == 4 and report["failed_requests"] == 0
            assert report["output_tokens_per_s"] > 0
            assert all(row["output_tokens"] == 3 for row in report["requests"])
            for row in report["requests"]:
                assert row["latency_s"] >= row["ttft_s"] > 0
                assert row["tpot_s"] >= 0
                assert row["server_timing"]["display"]["latency_s"].endswith(" s")
            assert sum(value["steps"] for value in report["server_steps"].values()) > 0
            assert report["server_steps"]["prefill"]["total_s"] > 0
            assert report["gpu"]["samples"] > 0 or report["gpu"]["errors"]
            # A one-token response has no decode interval or TPOT denominator.
            one = await asyncio.to_thread(
                run_benchmark,
                f"http://127.0.0.1:{port}",
                [{"token_ids": [1]}],
                requests=1,
                concurrency=1,
                max_tokens=1,
                warmup=0,
            )
            assert one["latency"]["tpot_s"] is None
            assert memory_display({"target_bytes": 1024**3})["target_display"] == "1.00 GiB"
        finally:
            server.should_exit = True
            await task
            sock.close()
        assert runner.cache_bytes == 0

    asyncio.run(scenario())
