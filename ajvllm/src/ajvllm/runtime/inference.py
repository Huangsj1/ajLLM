"""Shared startup and budgeted execution for offline and online inference."""

from dataclasses import dataclass

import torch

from ajvllm import Engine, EngineExecutionError
from ajvllm.attention.backends.triton import resolve_backend
from ajvllm.config import MemoryConfig
from ajvllm.config.compute import ComputeConfig
from ajvllm.execution.capacity import Qwen2MemoryEstimate
from ajvllm.execution.qwen2 import Qwen2Runner, read_eos_token_ids
from ajvllm.modeling.qwen2.weights import load_qwen2
from ajvllm.runtime.budget import MemoryBudget


@dataclass
class InferenceRuntime:
    engine: Engine
    budget: MemoryBudget | None = None

    @classmethod
    def from_model(cls, model, config, *, memory_config=None, compute_config=None, eos_token_ids=(), **budget_options):
        memory_config = memory_config or MemoryConfig()
        compute_config = ComputeConfig(backend=resolve_backend(compute_config or ComputeConfig(), model, memory_config))
        estimate = Qwen2MemoryEstimate(
            model.config, model.model.embed_tokens.weight.element_size(), config, memory_config, compute_config.backend
        )
        # Measure weights/static buffers before allocating KV; plan capacity first.
        budget = MemoryBudget(model.device, config, estimate=estimate, **budget_options)
        runner = Qwen2Runner(
            model,
            eos_token_ids,
            memory_config=memory_config,
            engine_config=budget.config,
            compute_config=compute_config,
        )
        budget.warmup(runner.probe)
        engine = Engine(runner, budget.config)
        engine.set_token_budget(budget.stats.token_budget)
        return cls(engine, budget)

    @classmethod
    def from_directory(cls, directory, config, *, device="cuda", dtype=None, **options):
        model = load_qwen2(directory, device=device, dtype=dtype)
        return cls.from_model(
            model, config, eos_token_ids=read_eos_token_ids(directory, model.config.vocab_size), **options
        )

    def step(self):
        if self.budget:
            self.budget.before_step(self.engine)
        try:
            outputs = self.engine.step()
        except EngineExecutionError as exc:
            if self.budget and isinstance(exc.__cause__, torch.cuda.OutOfMemoryError):
                self.budget.on_oom()
            raise
        else:
            if self.budget:
                self.budget.after_step(self.engine)
            return outputs

    def run(self):
        while self.engine.has_unfinished_requests:
            yield from self.step()
