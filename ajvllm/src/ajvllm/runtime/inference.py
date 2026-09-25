"""Shared startup and budgeted execution for offline and online inference."""

from dataclasses import dataclass, replace

from ajvllm import Engine
from ajvllm.attention.backends.triton import resolve_backend
from ajvllm.config import MemoryConfig
from ajvllm.config.advanced import GraphConfig, QuantizationConfig
from ajvllm.config.compute import ComputeConfig
from ajvllm.execution.qwen2 import Qwen2Runner, read_eos_token_ids
from ajvllm.modeling.qwen2.projections import pack_projections
from ajvllm.modeling.qwen2.weights import load_qwen2
from ajvllm.quantization.linear import quantize_model
from ajvllm.runtime.budget import MemoryBudget
from ajvllm.runtime.profiling import profile_memory


@dataclass
class InferenceRuntime:
    engine: Engine
    budget: MemoryBudget | None = None

    @classmethod
    def from_model(
        cls,
        model,
        config,
        *,
        memory_config=None,
        compute_config=None,
        graph_config=None,
        quantization_config=None,
        eos_token_ids=(),
        **budget_options,
    ):
        memory_config = memory_config or MemoryConfig()
        graph_config = graph_config or GraphConfig()
        quantize_model(model, quantization_config or QuantizationConfig())
        compute_config = ComputeConfig(backend=resolve_backend(compute_config or ComputeConfig(), model, memory_config))
        if compute_config.backend == "triton":
            pack_projections(model)
        budget = MemoryBudget(model, config, graph_reserve_bytes=graph_config.reserve_bytes, **budget_options)
        peak, temporary_pool, workspace, block_bytes = profile_memory(
            model, config, memory_config, compute_config, budget
        )
        if memory_config.backend == "paged":
            blocks = budget.resolve(
                profile_peak=peak,
                temporary_pool_bytes=temporary_pool,
                workspace_reserve=workspace,
                block_bytes=block_bytes,
                minimum_blocks=(config.max_model_len + memory_config.block_size - 1) // memory_config.block_size,
                explicit_blocks=memory_config.num_blocks,
            )
            memory_config = replace(memory_config, num_blocks=blocks)
        else:
            budget.record_profile(profile_peak=peak, temporary_pool_bytes=0, workspace_reserve=workspace)
        runner = Qwen2Runner(
            model,
            eos_token_ids,
            memory_config=memory_config,
            engine_config=config,
            compute_config=compute_config,
            graph_config=graph_config,
        )
        return cls(Engine(runner, config), budget)

    @classmethod
    def from_directory(cls, directory, config, *, device="cuda", dtype=None, **options):
        model = load_qwen2(directory, device=device, dtype=dtype)
        return cls.from_model(
            model, config, eos_token_ids=read_eos_token_ids(directory, model.config.vocab_size), **options
        )

    def step(self):
        return self.engine.step()

    def run(self):
        while self.engine.has_unfinished_requests:
            yield from self.step()
