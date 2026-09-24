"""Symmetric per-output-channel W8A16; no calibration or activation quantization."""

import torch
from torch import nn
from torch.nn import functional as F

from ajvllm.kernels.quantization import linear_w8a16


class Int8Linear(nn.Module):
    def __init__(self, weight, scales, bias):
        super().__init__()
        self.register_buffer("qweight", weight)
        self.register_buffer("scales", scales)
        self.register_buffer("bias", bias)
        self.out_features, self.in_features = weight.shape

    @classmethod
    @torch.no_grad()
    def from_linear(cls, module):
        if module.weight.dtype not in (torch.float16, torch.bfloat16) or not module.weight.is_cuda:
            raise ValueError("W8A16 requires CUDA FP16/BF16 model weights")
        # Bound conversion scratch to a small slice rather than a whole FP32 matrix.
        quantized = torch.empty_like(module.weight, dtype=torch.int8)
        scales = torch.empty(module.out_features, device=module.weight.device, dtype=torch.float32)
        for start in range(0, module.out_features, 128):
            values = module.weight[start : start + 128].float()
            maximum = values.abs().amax(dim=1)
            scale = torch.where(maximum == 0, 1.0, maximum / 127)
            quantized[start : start + 128] = (values / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
            scales[start : start + 128] = scale
        bias = module.bias.detach().clone() if module.bias is not None else None
        return cls(quantized, scales, bias)

    def forward(self, x):
        return linear_w8a16(x, self.qweight, self.scales, self.bias)

    def reference(self, x):
        """GPU numerical oracle; production never materializes dequantized weights."""
        weight = (self.qweight.float() * self.scales[:, None]).to(x.dtype)
        return F.linear(x, weight, self.bias)


def quantize_model(model, config):
    if config.mode == "none":
        return
    if model.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("W8A16 requires FP16/BF16 activation dtype")
    if torch.cuda.get_device_capability(model.device)[0] < 8:
        raise ValueError("W8A16 kernels require SM80+")
    # Only decoder projections: preserve embeddings, tied LM head, and norms.
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name, child in tuple(parent.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(parent, name, Int8Linear.from_linear(child))


def quantization_stats(model):
    linears = [module for module in model.modules() if isinstance(module, Int8Linear)]
    return {
        "mode": "w8a16" if linears else "none",
        "linear_layers": len(linears),
        "weight_dtype": "int8" if linears else str(model.dtype),
        "scale_dtype": "float32" if linears else None,
        "activation_dtype": str(model.dtype),
        "kv_dtype": str(model.dtype),
        "model_storage_bytes": sum(t.numel() * t.element_size() for t in (*model.parameters(), *model.buffers())),
    }
