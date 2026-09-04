"""Pre-training mechanics; the workflow module only wires configuration and I/O."""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from ajllm.training.checkpoint import load_checkpoint, save_checkpoint
from ajllm.training.distributed import FullyShardedDataParallel
from ajllm.training.evaluation import evaluate_causal_lm
from ajllm.training.logger import RunLogger
from ajllm.training.losses import cross_entropy
from ajllm.training.optimizers import AdamW
from ajllm.training.schedulers import warmup_cosine_learning_rate


@dataclass(frozen=True)
class PretrainConfig:
    epochs: int
    max_steps: int | None
    learning_rate: float
    min_lr: float
    warmup_steps: int
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    mixed_precision: str | None = None
    log_interval: int = 50
    eval_interval: int = 0
    eval_batches: int | None = None
    save_interval: int = 2_000
    resume_from: str | None = None

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")


def is_main_process() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _mean_across_ranks(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.AVG)
    return value


def clip_gradients(parameters, maximum_norm: float, epsilon: float = 1e-6) -> float:
    """Clip global gradient norm with primitive tensor operations."""
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    if not parameters:
        return 0.0
    squared_norm = sum(torch.sum(parameter.grad.detach().float().square()) for parameter in parameters)
    norm = torch.sqrt(squared_norm)
    norm_value = float(norm.item())
    if norm_value > maximum_norm:
        scale = maximum_norm / (norm_value + epsilon)
        for parameter in parameters:
            parameter.grad.mul_(scale)
    return norm_value


class Pretrainer:
    """One clear optimization loop shared by one-GPU and FSDP launches."""

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        config: PretrainConfig,
        output_dir: str | Path,
        metadata: dict[str, Any],
        validation_dataloader: DataLoader | None = None,
    ) -> None:
        self.model, self.dataloader, self.device = model, dataloader, device
        self.config, self.output_dir, self.metadata = config, Path(output_dir), metadata
        self.validation_dataloader = validation_dataloader
        self.optimizer = AdamW(
            model.parameters(),
            learning_rate=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision == "fp16" and device.type == "cuda")
        self.logger = (
            RunLogger(self.output_dir, append_existing=bool(config.resume_from)) if is_main_process() else None
        )

    def _autocast(self):
        enabled = self.device.type == "cuda" and self.config.mixed_precision in {"bf16", "fp16"}
        dtype = torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype, enabled=enabled)

    def train(self) -> dict[str, float | int | bool]:
        batches_per_epoch = len(self.dataloader)
        if batches_per_epoch == 0:
            raise ValueError("The training dataloader has no batches")
        updates_per_epoch = ceil(batches_per_epoch / self.config.gradient_accumulation_steps)
        epoch_total_steps = self.config.epochs * updates_per_epoch
        total_steps = min(epoch_total_steps, self.config.max_steps or epoch_total_steps)
        start_step = start_epoch = start_batch = 0
        if self.config.resume_from:
            checkpoint = load_checkpoint(self.config.resume_from, self.model, self.optimizer, self.device)
            start_step = int(checkpoint["step"])
            saved_state = checkpoint.get("training_state", {})
            start_epoch = int(saved_state.get("epoch", start_step // updates_per_epoch))
            inferred_batch = (start_step % updates_per_epoch) * self.config.gradient_accumulation_steps
            start_batch = int(
                saved_state.get("batch_in_epoch", inferred_batch)
            )
        if start_step >= total_steps:
            raise ValueError("The checkpoint has already completed the requested max_steps or epochs")
        self.model.train()
        progress = tqdm(total=total_steps, initial=start_step, disable=not is_main_process(), unit="step")
        started = time.perf_counter()
        running_lm_loss = running_auxiliary_loss = running_total_loss = 0.0
        logged_steps = 0
        processed_tokens = 0
        completed = start_step
        for epoch in range(start_epoch, self.config.epochs):
            if isinstance(self.dataloader.sampler, DistributedSampler):
                self.dataloader.sampler.set_epoch(epoch)
            iterator = iter(self.dataloader)
            batches_seen = 0
            batches_to_skip = start_batch if epoch == start_epoch else 0
            while batches_seen < batches_to_skip:
                next(iterator)
                batches_seen += 1
            while batches_seen < batches_per_epoch and completed < total_steps:
                lr = warmup_cosine_learning_rate(
                    completed, self.config.learning_rate, self.config.min_lr, self.config.warmup_steps, total_steps
                )
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                self.optimizer.zero_grad(set_to_none=True)
                lm_loss_for_log = auxiliary_loss_for_log = total_loss_for_log = None
                accumulation_count = 0
                while (
                    accumulation_count < self.config.gradient_accumulation_steps and batches_seen < batches_per_epoch
                ):
                    batch = next(iterator)
                    batches_seen += 1
                    accumulation_count += 1
                    input_ids, labels = batch["input_ids"].to(self.device), batch["labels"].to(self.device)
                    with self._autocast():
                        lm_loss = cross_entropy(self.model(input_ids), labels)
                        auxiliary_loss = self._auxiliary_loss()
                        total_loss = lm_loss + auxiliary_loss
                        loss = total_loss / self.config.gradient_accumulation_steps
                    self.scaler.scale(loss).backward()
                    lm_loss_for_log = self._accumulate(lm_loss_for_log, lm_loss)
                    auxiliary_loss_for_log = self._accumulate(auxiliary_loss_for_log, auxiliary_loss)
                    total_loss_for_log = self._accumulate(total_loss_for_log, total_loss)
                    processed_tokens += input_ids.numel()
                if accumulation_count != self.config.gradient_accumulation_steps:
                    # The last partial accumulation needs the same mean-gradient scale as full groups.
                    correction = self.config.gradient_accumulation_steps / accumulation_count
                    for parameter in self.model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                self.scaler.unscale_(self.optimizer)
                if isinstance(self.model, FullyShardedDataParallel):
                    self.model.finish_gradient_synchronization()
                grad_norm = clip_gradients(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                completed += 1
                assert (
                    lm_loss_for_log is not None
                    and auxiliary_loss_for_log is not None
                    and total_loss_for_log is not None
                )
                mean_lm_loss = _mean_across_ranks(lm_loss_for_log.float() / accumulation_count).item()
                mean_auxiliary_loss = _mean_across_ranks(auxiliary_loss_for_log.float() / accumulation_count).item()
                mean_total_loss = _mean_across_ranks(total_loss_for_log.float() / accumulation_count).item()
                running_lm_loss += mean_lm_loss
                running_auxiliary_loss += mean_auxiliary_loss
                running_total_loss += mean_total_loss
                logged_steps += 1
                progress.update(1)
                progress.set_postfix(
                    epoch=f"{epoch + 1}/{self.config.epochs}", loss=f"{mean_total_loss:.4f}", lr=f"{lr:.2e}"
                )
                if completed % self.config.log_interval == 0 or completed == 1 or completed == total_steps:
                    if self.logger:
                        elapsed = max(time.perf_counter() - started, 1e-9)
                        self.logger.log(
                            completed,
                            "train",
                            {
                                "epoch": epoch + 1,
                                "epoch_step": batches_seen,
                                "loss": running_total_loss / logged_steps,
                                "lm_loss": running_lm_loss / logged_steps,
                                "auxiliary_loss": running_auxiliary_loss / logged_steps,
                                "total_loss": running_total_loss / logged_steps,
                                "learning_rate": lr,
                                "gradient_norm": grad_norm,
                                "tokens_per_second": processed_tokens / elapsed,
                            },
                        )
                    running_lm_loss = running_auxiliary_loss = running_total_loss = 0.0
                    logged_steps = 0
                if (
                    self.validation_dataloader is not None
                    and self.config.eval_interval > 0
                    and (completed % self.config.eval_interval == 0 or completed == total_steps)
                ):
                    evaluation = evaluate_causal_lm(
                        self.model,
                        self.validation_dataloader,
                        self.device,
                        self.config.mixed_precision,
                        self.config.eval_batches,
                    )
                    if self.logger:
                        self.logger.log(completed, "evaluation", evaluation)
                if completed % self.config.save_interval == 0 or completed == total_steps:
                    next_epoch = epoch + 1 if batches_seen == batches_per_epoch else epoch
                    next_batch = 0 if next_epoch > epoch else batches_seen
                    save_checkpoint(
                        self.output_dir / f"step_{completed:08d}.pt",
                        self.model,
                        self.optimizer,
                        completed,
                        self.metadata,
                        {"epoch": next_epoch, "batch_in_epoch": next_batch},
                    )
            start_batch = 0
            if completed == total_steps:
                break
        progress.close()
        summary: dict[str, float | int | bool] = {
            "epochs": self.config.epochs,
            "steps": completed,
            "planned_steps": epoch_total_steps,
            "stopped_early": completed < epoch_total_steps,
            "elapsed_seconds": time.perf_counter() - started,
        }
        if is_main_process():
            import json

            (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _auxiliary_loss(self) -> torch.Tensor:
        base_model = self.model.module if isinstance(self.model, FullyShardedDataParallel) else self.model
        return base_model.auxiliary_loss()

    @staticmethod
    def _accumulate(total: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
        return value.detach() if total is None else total + value.detach()
