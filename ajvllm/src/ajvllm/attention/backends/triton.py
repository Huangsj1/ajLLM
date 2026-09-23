"""Compact per-batch launch metadata; no padded queries, masks or gathered KV."""

from dataclasses import dataclass
from itertools import accumulate

import torch

from ajvllm.kernels.attention import PREFILL_TILE


@dataclass
class AttentionMetadata:
    starts: torch.Tensor            # start indices of each request in the packed token buffer
    contexts: torch.Tensor          # context lengths of each request
    prefill_tiles: torch.Tensor     # (row, start) pairs for each prefill tile
    decode_rows: torch.Tensor       # rows of requests that are single-token decode requests
    max_decode_context: int         # maximum context length of all decode requests

    @classmethod
    def build(cls, lengths, contexts, device):
        decode = [row for row, length in enumerate(lengths) if length == 1]
        tiles = [
            (row, start) for row, length in enumerate(lengths) if length > 1 for start in range(0, length, PREFILL_TILE)
        ]
        return cls(
            torch.tensor([0, *accumulate(lengths)], device=device, dtype=torch.int32),
            torch.tensor(contexts, device=device, dtype=torch.int32),
            torch.tensor(tiles, device=device, dtype=torch.int32).reshape(-1, 2),
            torch.tensor(decode, device=device, dtype=torch.int32),
            max((contexts[row] for row in decode), default=0),
        )


def resolve_backend(config, model, memory_config):
    supported = (
        memory_config.backend == "paged"
        and model.device.type == "cuda"
        and torch.cuda.get_device_capability(model.device)[0] >= 8
        and model.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and 0 < model.config.head_dim <= 256
    )
    if config.backend == "triton" and not supported:
        raise ValueError("Triton requires paged KV, SM80+, FP16/BF16/FP32 and head_dim <= 256")
    if config.backend == "auto":
        return "triton" if supported and model.dtype != torch.float32 else "eager"
    return config.backend
