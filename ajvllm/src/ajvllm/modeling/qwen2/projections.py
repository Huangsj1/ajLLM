"""Pack checkpoint projections once, before inference and graph capture."""

import torch
from torch import nn


@torch.no_grad()
def pack_projections(model):
    # Keep the checkpoint layout in the loader. Runtime modules own only the
    # packed weights, so fusion does not retain a second copy of the model.
    # Quantized modules retain their existing W8A16 execution path.
    for layer in model.model.layers:
        for parent, names, target in (
            (layer.self_attn, ("q_proj", "k_proj", "v_proj"), "qkv_proj"),
            (layer.mlp, ("gate_proj", "up_proj"), "gate_up_proj"),
        ):
            if hasattr(parent, target):
                continue
            modules = [getattr(parent, name) for name in names]
            if not all(isinstance(module, nn.Linear) for module in modules):
                continue
            with torch.device("meta"):
                packed = nn.Linear(
                    modules[0].in_features,
                    sum(module.out_features for module in modules),
                    bias=modules[0].bias is not None,
                )
            packed.weight = nn.Parameter(torch.cat([module.weight for module in modules]), requires_grad=False)
            if packed.bias is not None:
                packed.bias = nn.Parameter(torch.cat([module.bias for module in modules]), requires_grad=False)
            setattr(parent, target, packed)
            for name in names:
                delattr(parent, name)
