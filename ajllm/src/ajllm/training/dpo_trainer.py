"""Direct Preference Optimization training mechanics."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from ajllm.training.checkpoint import load_checkpoint, save_checkpoint
from ajllm.training.logger import RunLogger
from ajllm.training.optimizers import AdamW
from ajllm.training.parallel import ExpertParallel, FullyShardedDataParallel, TensorExpertParallel, TensorParallel
from ajllm.training.pretrainer import PretrainConfig, _mean_across_ranks, clip_gradients, is_main_process
from ajllm.training.schedulers import warmup_cosine_learning_rate

_PARALLEL_WRAPPERS = (FullyShardedDataParallel, TensorParallel, ExpertParallel, TensorExpertParallel)


@dataclass(frozen=True)
class DPOConfig(PretrainConfig):
    """Optimizer settings plus the inverse-KL scale used by DPO."""

    beta: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.beta <= 0:
            raise ValueError("beta must be positive")


def token_log_probabilities(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return each selected causal token log-probability, masking ``-100`` labels."""
    valid = labels != -100
    safe_labels = labels.masked_fill(~valid, 0)
    # get the labels' log-probabilities from the effective logits (log(p_label))
    selected = functional.log_softmax(logits.float(), dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return selected.masked_fill(~valid, 0.0)


def sequence_log_probabilities(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum completion-token log probabilities for every sequence in a batch."""
    # \sum_t log p(x_t | x_{<t}) for each sequence in the batch, ignoring padding
    return token_log_probabilities(logits, labels).sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute ``-log sigmoid(beta * ((pi_w-ref_w) - (pi_l-ref_l)))``.

    The returned rewards are detached diagnostics in the conventional DPO
    units, i.e. ``beta * (log pi - log pi_ref)``.
    """
    chosen_reward = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_reward = beta * (policy_rejected_logps - reference_rejected_logps)
    margin = chosen_reward - rejected_reward
    loss = -functional.logsigmoid(margin).mean()
    return loss, {
        "chosen_reward": chosen_reward.detach().mean(),
        "rejected_reward": rejected_reward.detach().mean(),
        "reward_margin": margin.detach().mean(),
        "preference_accuracy": (margin.detach() > 0).float().mean(),
    }


def _base_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, _PARALLEL_WRAPPERS) else model


class DPOTrainer:
    """Train a policy against a frozen reference model on preference pairs."""

    def __init__(
        self,
        model: torch.nn.Module,
        reference_model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        config: DPOConfig,
        output_dir: str | Path,
        metadata: dict[str, Any],
        validation_dataloader: DataLoader | None = None,
    ) -> None:
        self.model, self.reference_model = model, reference_model
        self.dataloader, self.device, self.config = dataloader, device, config
        self.output_dir, self.metadata = Path(output_dir), metadata
        self.validation_dataloader = validation_dataloader
        self.use_cuda_kernels = _base_model(model).config.use_cuda_kernels
        self.optimizer = AdamW(
            model.parameters(),
            learning_rate=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
            use_cuda_kernels=self.use_cuda_kernels,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision == "fp16" and device.type == "cuda")
        self.logger = (
            RunLogger(self.output_dir, append_existing=bool(config.resume_from)) if is_main_process() else None
        )
        self._freeze_reference()

    def _freeze_reference(self) -> None:
        self.reference_model.eval()
        for parameter in self.reference_model.parameters():
            parameter.requires_grad_(False)

    def _autocast(self):
        enabled = self.device.type == "cuda" and self.config.mixed_precision in {"bf16", "fp16"}
        dtype = torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype, enabled=enabled)

    def _loss_for_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        # concatenate chosen and rejected sequences in one batch for efficiency
        chosen_input_ids = batch["chosen_input_ids"].to(self.device)
        chosen_labels = batch["chosen_labels"].to(self.device)
        rejected_input_ids = batch["rejected_input_ids"].to(self.device)
        rejected_labels = batch["rejected_labels"].to(self.device)
        input_ids = torch.cat((chosen_input_ids, rejected_input_ids), dim=0)
        labels = torch.cat((chosen_labels, rejected_labels), dim=0)
        with self._autocast():
            with torch.no_grad():
                reference_logps = sequence_log_probabilities(self.reference_model(input_ids), labels)
            policy_logps = sequence_log_probabilities(self.model(input_ids), labels)
            batch_size = chosen_input_ids.shape[0]
            preference_loss, diagnostics = dpo_loss(
                policy_logps[:batch_size],
                policy_logps[batch_size:],
                reference_logps[:batch_size],
                reference_logps[batch_size:],
                self.config.beta,
            )
            auxiliary_loss = _base_model(self.model).auxiliary_loss()
            return preference_loss, auxiliary_loss, preference_loss + auxiliary_loss, diagnostics

    def _gradient_norm(self) -> float:
        if isinstance(self.model, (TensorParallel, ExpertParallel, TensorExpertParallel)):
            return self.model.clip_grad_norm_(self.config.grad_clip)
        return clip_gradients(self.model.parameters(), self.config.grad_clip, self.use_cuda_kernels)

    def _evaluate(self) -> dict[str, float]:
        if self.validation_dataloader is None:
            raise AssertionError("_evaluate requires a validation dataloader")
        was_training = self.model.training
        self.model.eval()
        totals = torch.zeros(7, device=self.device, dtype=torch.float64)
        with torch.no_grad():
            for batch_index, batch in enumerate(self.validation_dataloader):
                if self.config.eval_batches is not None and batch_index >= self.config.eval_batches:
                    break
                preference_loss, auxiliary_loss, total_loss, diagnostics = self._loss_for_batch(batch)
                count = batch["chosen_input_ids"].shape[0]
                totals += torch.tensor(
                    [
                        preference_loss.item() * count,
                        auxiliary_loss.item() * count,
                        total_loss.item() * count,
                        diagnostics["chosen_reward"].item() * count,
                        diagnostics["rejected_reward"].item() * count,
                        diagnostics["preference_accuracy"].item() * count,
                        count,
                    ],
                    device=self.device,
                    dtype=torch.float64,
                )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
        # Every TP rank evaluates the same batch; averaged values are unchanged.
        samples = int(totals[6].item())
        if samples <= 0:
            raise ValueError("Validation data has no preference pairs")
        values = (totals[:6] / samples).tolist()
        if was_training:
            self.model.train()
        return {
            "dpo_loss": values[0],
            "auxiliary_loss": values[1],
            "total_loss": values[2],
            "chosen_reward": values[3],
            "rejected_reward": values[4],
            "preference_accuracy": values[5],
            "evaluated_pairs": samples,
        }

    def train(self) -> dict[str, float | int | bool]:
        batches_per_epoch = len(self.dataloader)
        if batches_per_epoch == 0:
            raise ValueError("The training dataloader has no batches")
        updates_per_epoch = ceil(batches_per_epoch / self.config.gradient_accumulation_steps)
        planned_steps = self.config.epochs * updates_per_epoch
        total_steps = min(planned_steps, self.config.max_steps or planned_steps)
        start_step = start_epoch = start_batch = 0
        if self.config.resume_from:
            checkpoint = load_checkpoint(self.config.resume_from, self.model, self.optimizer, self.device)
            start_step = int(checkpoint["step"])
            state = checkpoint.get("training_state", {})
            start_epoch = int(state.get("epoch", start_step // updates_per_epoch))
            start_batch = int(
                state.get("batch_in_epoch", (start_step % updates_per_epoch) * self.config.gradient_accumulation_steps)
            )
        if start_step >= total_steps:
            raise ValueError("The checkpoint has already completed the requested max_steps or epochs")
        self.model.train()
        progress = tqdm(total=total_steps, initial=start_step, disable=not is_main_process(), unit="step")
        started = time.perf_counter()
        running = {
            key: 0.0
            for key in (
                "dpo_loss",
                "auxiliary_loss",
                "total_loss",
                "chosen_reward",
                "rejected_reward",
                "reward_margin",
                "preference_accuracy",
            )
        }
        logged_steps = completed = start_step
        for epoch in range(start_epoch, self.config.epochs):
            sampler = self.dataloader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            batches_to_skip = start_batch if epoch == start_epoch else 0
            if hasattr(sampler, "set_start_index"):
                batch_size = self.dataloader.batch_size
                if not isinstance(batch_size, int):
                    raise TypeError("checkpoint recovery requires a fixed DataLoader batch_size")
                sampler.set_start_index(batches_to_skip * batch_size)
                batches_seen = batches_to_skip
            else:
                batches_seen = 0
            iterator = iter(self.dataloader)
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
                totals: dict[str, torch.Tensor | None] = {key: None for key in running}
                accumulation_count = 0
                while accumulation_count < self.config.gradient_accumulation_steps and batches_seen < batches_per_epoch:
                    batch = next(iterator)
                    batches_seen += 1
                    accumulation_count += 1
                    preference_loss, auxiliary_loss, total_loss, diagnostics = self._loss_for_batch(batch)
                    self.scaler.scale(total_loss / self.config.gradient_accumulation_steps).backward()
                    values = {
                        "dpo_loss": preference_loss,
                        "auxiliary_loss": auxiliary_loss,
                        "total_loss": total_loss,
                        **diagnostics,
                    }
                    for key, value in values.items():
                        totals[key] = value.detach() if totals[key] is None else totals[key] + value.detach()
                if accumulation_count != self.config.gradient_accumulation_steps:
                    correction = self.config.gradient_accumulation_steps / accumulation_count
                    for parameter in self.model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                self.scaler.unscale_(self.optimizer)
                if isinstance(self.model, _PARALLEL_WRAPPERS):
                    self.model.finish_gradient_synchronization()
                gradient_norm = self._gradient_norm()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                completed += 1
                means = {
                    key: _mean_across_ranks(value.float() / accumulation_count).item()
                    for key, value in totals.items()
                    if value is not None
                }
                for key, value in means.items():
                    running[key] += value
                logged_steps += 1
                progress.update(1)
                progress.set_postfix(
                    epoch=f"{epoch + 1}/{self.config.epochs}", loss=f"{means['total_loss']:.4f}", lr=f"{lr:.2e}"
                )
                if completed % self.config.log_interval == 0 or completed == 1 or completed == total_steps:
                    if self.logger:
                        self.logger.log(
                            completed,
                            "train",
                            {
                                "epoch": epoch + 1,
                                "epoch_step": batches_seen,
                                **{key: value / logged_steps for key, value in running.items()},
                                "learning_rate": lr,
                                "gradient_norm": gradient_norm,
                            },
                        )
                    running = {key: 0.0 for key in running}
                    logged_steps = 0
                if (
                    self.validation_dataloader is not None
                    and self.config.eval_interval > 0
                    and (completed % self.config.eval_interval == 0 or completed == total_steps)
                ):
                    evaluation = self._evaluate()
                    if self.logger:
                        self.logger.log(completed, "evaluation", evaluation)
                if completed % self.config.save_interval == 0 or completed == total_steps:
                    next_epoch = epoch + 1 if batches_seen == batches_per_epoch else epoch
                    save_checkpoint(
                        self.output_dir / f"step_{completed:08d}.pt",
                        self.model,
                        self.optimizer,
                        completed,
                        self.metadata,
                        {"epoch": next_epoch, "batch_in_epoch": 0 if next_epoch > epoch else batches_seen},
                    )
            start_batch = 0
            if completed == total_steps:
                break
        progress.close()
        summary: dict[str, float | int | bool] = {
            "epochs": self.config.epochs,
            "steps": completed,
            "planned_steps": planned_steps,
            "stopped_early": completed < planned_steps,
            "elapsed_seconds": time.perf_counter() - started,
        }
        if is_main_process():
            (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
