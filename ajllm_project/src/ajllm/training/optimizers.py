"""AdamW implemented directly from its update equations."""

from __future__ import annotations

import torch


class AdamW(torch.optim.Optimizer):
    """Adam with decoupled weight decay."""

    def __init__(
        self,
        parameters,
        learning_rate: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        beta1, beta2 = betas
        if learning_rate < 0:
            raise ValueError("learning_rate must be non-negative")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("Adam beta values must be in [0, 1)")
        defaults = {
            "lr": learning_rate,
            "betas": betas,
            "eps": epsilon,
            "weight_decay": weight_decay,
        }
        super().__init__(parameters, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["first_moment"] = torch.zeros_like(parameter)
                    state["second_moment"] = torch.zeros_like(parameter)
                state["step"] += 1
                step = state["step"]
                first_moment = state["first_moment"]
                second_moment = state["second_moment"]
                first_moment.mul_(beta1).add_(gradient, alpha=1 - beta1)
                second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                adjusted_learning_rate = group["lr"] * (1 - beta2**step) ** 0.5 / (1 - beta1**step)
                parameter.addcdiv_(first_moment, second_moment.sqrt().add_(group["eps"]), value=-adjusted_learning_rate)
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
        return loss
