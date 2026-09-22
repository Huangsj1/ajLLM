"""Contiguous reference cache tensor types."""

import torch

LayerKV = tuple[torch.Tensor, torch.Tensor]
KVCache = tuple[LayerKV, ...]
