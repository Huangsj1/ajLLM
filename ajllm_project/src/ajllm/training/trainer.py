"""Reusable Transformer language model training loop."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from ajllm.training.checkpoint import load_checkpoint, save_checkpoint
from ajllm.training.data import TokenDataset
from ajllm.training.logger import RunLogger
from ajllm.training.losses import cross_entropy
from ajllm.training.optimizers import AdamW
from ajllm.training.schedulers import warmup_cosine_learning_rate


def clip_gradients(parameters, maximum_norm: float, epsilon: float = 1e-6) -> float:
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    if not parameters:
        return 0.0
    total_norm = torch.sqrt(sum(torch.sum(parameter.grad.square()) for parameter in parameters))
    norm_value = float(total_norm.item())
    if norm_value > maximum_norm:
        scale = maximum_norm / (norm_value + epsilon)
        for parameter in parameters:
            parameter.grad.mul_(scale)
    return norm_value


class Trainer:
    """Optimize a language model and maintain a self-contained run directory."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: TokenDataset,
        validation_dataset: TokenDataset | None,
        run_directory: str | Path,
        config: dict[str, Any],
        device: torch.device,
        seed: int,
    ) -> None:
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.run_directory = Path(run_directory)
        self.config = config
        self.device = device
        self.generator = np.random.default_rng(seed)
        self.logger = RunLogger(self.run_directory)

        optimizer_config = config.get("optimizer", {})
        self.maximum_learning_rate = float(optimizer_config.get("max_lr", 1e-3))
        self.minimum_learning_rate = float(
            optimizer_config.get(
                "min_lr",
                self.maximum_learning_rate * float(optimizer_config.get("min_lr_ratio", 0.1)),
            )
        )
        self.optimizer = AdamW(
            self.model.parameters(),
            learning_rate=self.maximum_learning_rate,
            betas=(float(optimizer_config.get("beta1", 0.9)), float(optimizer_config.get("beta2", 0.95))),
            epsilon=float(optimizer_config.get("epsilon", 1e-8)),
            weight_decay=float(optimizer_config.get("weight_decay", 0.1)),
        )

    @torch.no_grad()
    def evaluate(self, batches: int, batch_size: int, context_length: int) -> dict[str, float]:
        self.model.eval()
        results: dict[str, float] = {}
        datasets = {"train": self.train_dataset, "validation": self.validation_dataset}
        for name, dataset in datasets.items():
            if dataset is None:
                continue
            losses = []
            for _ in range(batches):
                inputs, targets = dataset.batch(batch_size, context_length, self.device, self.generator)
                losses.append(float(cross_entropy(self.model(inputs), targets).item()))
            results[f"{name}_loss"] = sum(losses) / len(losses)
            results[f"{name}_perplexity"] = math.exp(min(results[f"{name}_loss"], 50.0))
        self.model.train()
        return results

    def train(self, checkpoint_metadata: dict[str, Any]) -> dict[str, Any]:
        training = self.config.get("training", {})
        logging = self.config.get("logging", {})
        batch_size = int(training.get("batch_size", 64))
        context_length = int(self.model.context_length)
        tokens_per_step = batch_size * context_length
        if "max_steps" in training:
            max_steps = int(training["max_steps"])
        elif "total_tokens" in training:
            max_steps = max(1, int(training["total_tokens"]) // tokens_per_step)
        else:
            raise ValueError("training.max_steps or training.total_tokens is required")

        warmup_steps = int(self.config.get("scheduler", {}).get("warmup_steps", 0))
        gradient_clip = float(training.get("gradient_clip", 1.0))
        log_interval = int(logging.get("log_interval", 50))
        eval_interval = int(logging.get("eval_interval", 500))
        eval_batches = int(logging.get("eval_batches", 20))
        checkpoint_interval = int(logging.get("checkpoint_interval", 10000))
        checkpoint_directory = self.run_directory / "checkpoints"
        checkpoint_directory.mkdir(parents=True, exist_ok=True)

        start_step = 0
        resume_from = training.get("resume_from")
        if resume_from:
            checkpoint = load_checkpoint(resume_from, self.model, self.optimizer, self.device)
            start_step = int(checkpoint["step"])

        best_validation_loss = float("inf")
        training_started = time.perf_counter()
        self.model.train()
        progress = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps, desc="Training", unit="step")
        for step in progress:
            learning_rate = warmup_cosine_learning_rate(
                step,
                self.maximum_learning_rate,
                self.minimum_learning_rate,
                warmup_steps,
                max_steps,
            )
            for parameter_group in self.optimizer.param_groups:
                parameter_group["lr"] = learning_rate

            inputs, targets = self.train_dataset.batch(batch_size, context_length, self.device, self.generator)
            logits = self.model(inputs)
            loss = cross_entropy(logits, targets)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = clip_gradients(self.model.parameters(), gradient_clip)
            self.optimizer.step()
            completed_step = step + 1
            progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{learning_rate:.2e}")

            if completed_step % log_interval == 0 or completed_step == 1:
                elapsed = max(time.perf_counter() - training_started, 1e-9)
                self.logger.log(
                    completed_step,
                    "train",
                    {
                        "loss": float(loss.item()),
                        "learning_rate": learning_rate,
                        "gradient_norm": gradient_norm,
                        "tokens_per_second": completed_step * tokens_per_step / elapsed,
                    },
                )

            if eval_interval > 0 and (completed_step % eval_interval == 0 or completed_step == max_steps):
                evaluation = self.evaluate(eval_batches, batch_size, context_length)
                self.logger.log(completed_step, "evaluation", evaluation)
                validation_loss = evaluation.get("validation_loss")
                if validation_loss is not None and validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    save_checkpoint(
                        checkpoint_directory / "best.pt",
                        self.model,
                        self.optimizer,
                        completed_step,
                        checkpoint_metadata,
                    )

            if checkpoint_interval > 0 and completed_step % checkpoint_interval == 0:
                save_checkpoint(
                    checkpoint_directory / f"step_{completed_step:08d}.pt",
                    self.model,
                    self.optimizer,
                    completed_step,
                    checkpoint_metadata,
                )
                save_checkpoint(
                    checkpoint_directory / "latest.pt",
                    self.model,
                    self.optimizer,
                    completed_step,
                    checkpoint_metadata,
                )

        final_path = save_checkpoint(
            checkpoint_directory / "final.pt",
            self.model,
            self.optimizer,
            max_steps,
            checkpoint_metadata,
        )
        save_checkpoint(
            checkpoint_directory / "latest.pt",
            self.model,
            self.optimizer,
            max_steps,
            checkpoint_metadata,
        )
        final_evaluation = self.evaluate(eval_batches, batch_size, context_length)
        if not (checkpoint_directory / "best.pt").is_file():
            save_checkpoint(
                checkpoint_directory / "best.pt",
                self.model,
                self.optimizer,
                max_steps,
                checkpoint_metadata,
            )
        summary = {
            "status": "completed",
            "steps": max_steps,
            "tokens_seen": max_steps * tokens_per_step,
            "elapsed_seconds": time.perf_counter() - training_started,
            "best_validation_loss": None if math.isinf(best_validation_loss) else best_validation_loss,
            "checkpoint": str(final_path),
            **final_evaluation,
        }
        with (self.run_directory / "summary.json").open("w", encoding="utf-8") as output_file:
            json.dump(summary, output_file, indent=2)
        return summary
