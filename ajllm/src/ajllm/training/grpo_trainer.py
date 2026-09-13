"""Batched online Group Relative Policy Optimization (GRPO) mechanics."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
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
from ajllm.training.rewards import SkyworkRewardModel
from ajllm.training.schedulers import warmup_cosine_learning_rate

_PARALLEL_WRAPPERS = (FullyShardedDataParallel, TensorParallel, ExpertParallel, TensorExpertParallel)


@dataclass(frozen=True)
class GRPOConfig(PretrainConfig):
    """Online rollout, repeated-update, and GRPO objective settings."""

    num_generations: int = 4
    updates_per_rollout: int = 4
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    kl_coef: float = 0.04
    clip_epsilon: float = 0.2
    advantage_epsilon: float = 1e-4
    rollout_backend: str = "torch"
    vllm: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.gradient_accumulation_steps != 1:
            raise ValueError("GRPO currently requires gradient_accumulation_steps: 1")
        if self.num_generations < 2 or self.updates_per_rollout < 1 or self.max_new_tokens < 1:
            raise ValueError("num_generations >= 2, updates_per_rollout >= 1, and max_new_tokens >= 1 are required")
        if self.temperature < 0 or self.top_k < 0 or not 0 < self.top_p <= 1:
            raise ValueError("temperature/top_k/top_p are invalid")
        if self.kl_coef < 0 or self.clip_epsilon < 0 or self.advantage_epsilon <= 0:
            raise ValueError("kl_coef, clip_epsilon, and advantage_epsilon are invalid")
        if self.rollout_backend not in {"torch", "vllm"}:
            raise ValueError("rollout_backend must be 'torch' or 'vllm'")
        if self.rollout_backend == "vllm" and not isinstance(self.vllm, dict):
            raise ValueError("rollout_backend='vllm' requires a vllm mapping")


@dataclass
class RolloutBatch:
    """Right-padded fixed trajectories with their static GRPO quantities."""

    input_ids: torch.Tensor  # padding(input_ids + output_ids)
    action_positions: torch.Tensor  # shape (B*G, max_completion), completion-token positions
    completion_ids: torch.Tensor  # padding(output_ids)
    completion_mask: torch.Tensor  # shape (B*G, max_completion), valid completion-token mask
    rewards: torch.Tensor  # shape (B*G,), one scalar reward per rollout
    advantages: torch.Tensor  # shape (B*G,), one scalar advantage per rollout
    reference_logps: torch.Tensor  # shape (B*G, max_completion), zero at padding positions
    old_logps: torch.Tensor | None = None  # shape (B*G, max_completion), zero at padding positions


def group_relative_advantages(rewards: torch.Tensor, num_generations: int, epsilon: float) -> torch.Tensor:
    """Normalize terminal rewards within each prompt's response group."""
    if rewards.ndim != 1 or rewards.numel() % num_generations:
        raise ValueError("rewards must be a flat tensor divisible by num_generations")
    grouped = rewards.view(-1, num_generations)
    return (
        (grouped - grouped.mean(dim=1, keepdim=True)) / (grouped.std(dim=1, keepdim=True, unbiased=False) + epsilon)
    ).reshape(-1)


def grpo_token_loss(
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor | None,
    clip_epsilon: float,
    kl_coef: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean masked GRPO loss, surrogate, and KL over rollout sequences.

    ``advantages`` is one scalar per completion.  Each sequence is token-mean
    reduced before the batch mean, preventing longer completions from receiving
    disproportionate weight.
    """
    if not (current_logps.shape == old_logps.shape == reference_logps.shape):
        raise ValueError("GRPO log-probability tensors must have identical shapes")
    if current_logps.ndim == 1:
        current_logps = current_logps.unsqueeze(0)
        old_logps = old_logps.unsqueeze(0)
        reference_logps = reference_logps.unsqueeze(0)
    if advantages.ndim == 0:
        advantages = advantages.unsqueeze(0)
    if advantages.shape != current_logps.shape[:1]:
        raise ValueError("advantages must have one value per rollout")
    mask = torch.ones_like(current_logps, dtype=torch.bool) if completion_mask is None else completion_mask.bool()
    ratio = torch.exp(current_logps - old_logps)
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    surrogate = torch.minimum(ratio * advantages.unsqueeze(1), clipped_ratio * advantages.unsqueeze(1))
    log_ratio_to_reference = reference_logps - current_logps
    kl = torch.exp(log_ratio_to_reference) - log_ratio_to_reference - 1
    counts = mask.sum(dim=1).clamp_min(1)
    sequence_surrogate = (surrogate * mask).sum(dim=1) / counts
    sequence_kl = (kl * mask).sum(dim=1) / counts
    return -(sequence_surrogate - kl_coef * sequence_kl).mean(), sequence_surrogate.mean(), sequence_kl.mean()


def _base_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, _PARALLEL_WRAPPERS) else model


def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    filtered = logits.float() / temperature
    if top_k:
        filtered = filtered.masked_fill(
            filtered < torch.topk(filtered, min(top_k, filtered.numel())).values[-1], float("-inf")
        )
    if top_p < 1:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        remove = torch.cumsum(functional.softmax(sorted_logits, dim=-1), dim=-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        filtered = filtered.scatter(0, sorted_indices, sorted_logits.masked_fill(remove, float("-inf")))
    return int(torch.multinomial(functional.softmax(filtered, dim=-1), 1).item())


class GRPOTrainer:
    """Batched rollouts and multiple policy updates per fixed trajectory set."""

    def __init__(
        self,
        model: torch.nn.Module,
        reference_model: torch.nn.Module,
        reward_model: SkyworkRewardModel,
        tokenizer: object,
        dataloader: DataLoader,
        device: torch.device,
        config: GRPOConfig,
        output_dir: str | Path,
        metadata: dict[str, Any],
    ) -> None:
        self.model, self.reference_model, self.reward_model = model, reference_model, reward_model
        self.tokenizer, self.dataloader, self.device, self.config = tokenizer, dataloader, device, config
        self.output_dir, self.metadata = Path(output_dir), metadata
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
        self.reference_model.eval().requires_grad_(False)
        self.vllm_server = self._start_vllm() if config.rollout_backend == "vllm" else None

    def _start_vllm(self):
        if self.device.type != "cuda":
            raise ValueError("rollout_backend='vllm' requires a CUDA training device")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            raise ValueError("vLLM rollout currently supports one policy-training process only")
        from ajllm.utils.vllm_util import VLLMServer

        options = dict(self.config.vllm or {})
        model_id = options.pop("model_id", None)
        if not isinstance(model_id, str):
            raise ValueError("vllm.model_id must name a vLLM-compatible copy of the ajLLM policy")
        server = VLLMServer(model_id=model_id, seed=int(options.pop("seed", 42)), **options)
        server.start()
        server.init_weight_sync(str(self.device))
        return server

    def _autocast(self):
        enabled = self.device.type == "cuda" and self.config.mixed_precision in {"bf16", "fp16"}
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16,
            enabled=enabled,
        )

    @torch.no_grad()
    def _torch_rollouts(self, prompt_ids: list[list[int]]) -> list[list[int]]:
        """Sample every B×G trajectory in parallel using right-padded contexts."""
        repeated_prompts = [prompt for prompt in prompt_ids for _ in range(self.config.num_generations)]
        prompt_lengths = torch.tensor([len(prompt) for prompt in repeated_prompts], device=self.device)
        width = int(prompt_lengths.max().item())
        max_new = min(self.config.max_new_tokens, _base_model(self.model).config.context_length - width)
        if max_new < 1:
            raise ValueError(
                "A rollout prompt fills model context; lower max_prompt_len or use a larger context_length"
            )
        # right-pad all prompts to the same width
        inputs = torch.full(
            (len(repeated_prompts), width + max_new), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device
        )
        for row, prompt in enumerate(repeated_prompts):
            inputs[row, : len(prompt)] = torch.tensor(prompt, device=self.device)
        completion_ids: list[list[int]] = [[] for _ in repeated_prompts]
        active = torch.ones(len(repeated_prompts), dtype=torch.bool, device=self.device)
        was_training = self.model.training
        self.model.eval()
        for offset in range(max_new):
            # input a batch of right-padded prompts
            with self._autocast():
                logits = self.model(inputs[:, : width + offset])
            next_positions = prompt_lengths - 1 + offset
            next_logits = logits[torch.arange(logits.shape[0], device=self.device), next_positions]
            # for each active rollout, sample a token and append it to the right-padded input
            for row in torch.nonzero(active, as_tuple=False).flatten().tolist():
                token_id = _sample(next_logits[row], self.config.temperature, self.config.top_k, self.config.top_p)
                position = int(prompt_lengths[row].item()) + offset
                inputs[row, position] = token_id
                completion_ids[row].append(token_id)
                if token_id == self.tokenizer.eos_token_id:
                    active[row] = False
            if not torch.any(active):
                break
        if was_training:
            self.model.train()
        return completion_ids

    def _vllm_rollouts(self, prompt_ids: list[list[int]]) -> list[list[int]]:
        assert self.vllm_server is not None
        # Synchronize only before a *new* rollout group.  The following
        # updates intentionally compare the changing policy against this fixed
        # behavior policy, so syncing between them would be incorrect.
        self.vllm_server.sync_policy_weights(_base_model(self.model))
        completions = self.vllm_server.generate_completions(
            prompt_ids,
            {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_new_tokens,
                "n": self.config.num_generations,
                "seed": torch.initial_seed(),
                "stop": None,
            },
        )
        expected = len(prompt_ids) * self.config.num_generations
        if len(completions) != expected:
            raise RuntimeError(f"vLLM returned {len(completions)} completions; expected {expected}")
        vocab_size = _base_model(self.model).config.vocab_size
        token_lists = [completion.token_ids or [self.tokenizer.eos_token_id] for completion in completions]
        if any(token < 0 or token >= vocab_size for token_ids in token_lists for token in token_ids):
            raise ValueError("vLLM completion token IDs do not match the ajLLM policy tokenizer/vocabulary")
        return token_lists

    def _make_rollout_batch(
        self,
        prompt_ids: list[list[int]],
        messages: list[list[dict[str, Any]]],
        completion_prefixes: list[str],
    ) -> RolloutBatch:
        # get num_generations completions ids for each prompt of the batch
        completion_lists = self._vllm_rollouts(prompt_ids) if self.vllm_server else self._torch_rollouts(prompt_ids)
        expanded_prompts = [prompt for prompt in prompt_ids for _ in range(self.config.num_generations)]
        expanded_messages = [message for message in messages for _ in range(self.config.num_generations)]
        expanded_prefixes = [prefix for prefix in completion_prefixes for _ in range(self.config.num_generations)]
        if len(completion_lists) != len(expanded_prompts):
            raise AssertionError("rollout completion count does not match expanded prompt count")
        full_sequences = [
            [*prompt, *completion] for prompt, completion in zip(expanded_prompts, completion_lists, strict=True)
        ]
        prompt_lengths = torch.tensor([len(prompt) for prompt in expanded_prompts], device=self.device)
        completion_lengths = torch.tensor([len(completion) for completion in completion_lists], device=self.device)
        max_sequence = max(map(len, full_sequences))
        max_completion = int(completion_lengths.max().item())
        # right-pad all sequences to the same width for a single forward pass
        input_ids = torch.full(
            (len(full_sequences), max_sequence), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device
        )
        completion_ids = torch.full(
            (len(full_sequences), max_completion), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device
        )
        for row, (sequence, completion) in enumerate(zip(full_sequences, completion_lists, strict=True)):
            input_ids[row, : len(sequence)] = torch.tensor(sequence, device=self.device)
            completion_ids[row, : len(completion)] = torch.tensor(completion, device=self.device)
        offsets = torch.arange(max_completion, device=self.device)
        # Shape (B*G, max_completion): valid completion-token mask.
        completion_mask = offsets.unsqueeze(0) < completion_lengths.unsqueeze(1)
        # Shape (B*G, max_completion): completion token positions in input_ids.
        action_positions = prompt_lengths.unsqueeze(1) - 1 + offsets.unsqueeze(0)
        conversations = [
            [
                *history,
                {"role": "assistant", "content": prefix + self.tokenizer.decode(completion, skip_special_tokens=True)},
            ]
            for history, prefix, completion in zip(expanded_messages, expanded_prefixes, completion_lists, strict=True)
        ]
        rewards = self.reward_model.score(conversations)
        advantages = group_relative_advantages(rewards, self.config.num_generations, self.config.advantage_epsilon)
        batch = RolloutBatch(
            input_ids,
            action_positions,
            completion_ids,
            completion_mask,
            rewards,
            advantages,
            torch.empty(0, device=self.device),
        )
        with torch.no_grad():
            batch.reference_logps, _ = self._batched_logps(self.reference_model, batch)
        return batch

    def _batched_logps(self, model: torch.nn.Module, rollout: RolloutBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Score all variable-length trajectories in one right-padded forward pass."""
        with self._autocast():
            logits = model(rollout.input_ids)
            positions = rollout.action_positions.clamp_max(logits.shape[1] - 1)
            # get the padding completion token logits for each rollout
            action_logits = logits.gather(1, positions.unsqueeze(-1).expand(-1, -1, logits.shape[-1]))
            logps = (
                functional.log_softmax(action_logits.float(), dim=-1)
                .gather(-1, rollout.completion_ids.unsqueeze(-1))
                .squeeze(-1)
            )
            # set log-probabilities of padding tokens to zero so they do not contribute to the GRPO loss
            return logps.masked_fill(~rollout.completion_mask, 0.0), _base_model(model).auxiliary_loss()

    def _update_rollout(self, rollout: RolloutBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Rollout and log-probability scoring must share inference behavior.
        # ``eval`` disables any configured dropout but still records gradients.
        self.model.eval()
        current_logps, auxiliary_loss = self._batched_logps(self.model, rollout)
        if rollout.old_logps is None:
            # Cache pi_old from the first policy evaluation.  Before its first
            # optimizer step it equals pi_theta; later updates reuse it.
            rollout.old_logps = current_logps.detach()
        policy_loss, surrogate, kl = grpo_token_loss(
            current_logps,
            rollout.old_logps,
            rollout.reference_logps,
            rollout.advantages,
            rollout.completion_mask,
            self.config.clip_epsilon,
            self.config.kl_coef,
        )
        total_loss = policy_loss + auxiliary_loss
        return total_loss, {
            "policy_loss": policy_loss.detach(),
            "auxiliary_loss": auxiliary_loss.detach(),
            "reward": rollout.rewards.detach().mean(),
            "reward_std": rollout.rewards.detach().std(unbiased=False),
            "group_reward_std": rollout.rewards.detach()
            .view(-1, self.config.num_generations)
            .std(dim=1, unbiased=False)
            .mean(),
            "advantage_mean": rollout.advantages.detach().mean(),
            "advantage_std": rollout.advantages.detach().std(unbiased=False),
            "kl": kl.detach(),
            "surrogate": surrogate.detach(),
            "response_length": rollout.completion_mask.sum(dim=1).float().mean(),
        }

    def _gradient_norm(self) -> float:
        if isinstance(self.model, (TensorParallel, ExpertParallel, TensorExpertParallel)):
            return self.model.clip_grad_norm_(self.config.grad_clip)
        return clip_gradients(self.model.parameters(), self.config.grad_clip, self.use_cuda_kernels)

    def train(self) -> dict[str, float | int | bool]:
        batches_per_epoch = len(self.dataloader)
        if batches_per_epoch == 0:
            raise ValueError("The training dataloader has no batches")
        planned_steps = self.config.epochs * batches_per_epoch * self.config.updates_per_rollout
        total_steps = min(planned_steps, self.config.max_steps or planned_steps)
        start_step = start_epoch = start_batch = 0
        if self.config.resume_from:
            checkpoint = load_checkpoint(self.config.resume_from, self.model, self.optimizer, self.device)
            start_step = int(checkpoint["step"])
            state = checkpoint.get("training_state", {})
            start_epoch = int(state.get("epoch", 0))
            start_batch = int(state.get("batch_in_epoch", 0))
        if start_step >= total_steps:
            raise ValueError("The checkpoint has already completed the requested max_steps or epochs")
        metric_names = (
            "total_loss",
            "policy_loss",
            "auxiliary_loss",
            "reward",
            "reward_std",
            "group_reward_std",
            "advantage_mean",
            "advantage_std",
            "kl",
            "surrogate",
            "response_length",
        )
        progress = tqdm(total=total_steps, initial=start_step, disable=not is_main_process(), unit="step")
        started, completed, logged_steps = time.perf_counter(), start_step, 0
        running = {name: 0.0 for name in metric_names}
        try:
            for epoch in range(start_epoch, self.config.epochs):
                sampler = self.dataloader.sampler
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
                batches_to_skip = start_batch if epoch == start_epoch else 0
                if hasattr(sampler, "set_start_index"):
                    assert isinstance(self.dataloader.batch_size, int)
                    sampler.set_start_index(batches_to_skip * self.dataloader.batch_size)
                iterator = iter(self.dataloader)
                for _ in range(batches_to_skip):
                    next(iterator)
                batches_seen = batches_to_skip
                while batches_seen < batches_per_epoch and completed < total_steps:
                    batch = next(iterator)
                    batches_seen += 1
                    rollout = self._make_rollout_batch(
                        batch["prompt_ids"], batch["messages"], batch["completion_prefix"]
                    )
                    rounds = min(self.config.updates_per_rollout, total_steps - completed)
                    for update_index in range(rounds):
                        lr = warmup_cosine_learning_rate(
                            completed,
                            self.config.learning_rate,
                            self.config.min_lr,
                            self.config.warmup_steps,
                            total_steps,
                        )
                        for group in self.optimizer.param_groups:
                            group["lr"] = lr
                        self.optimizer.zero_grad(set_to_none=True)
                        total_loss, diagnostics = self._update_rollout(rollout)
                        self.scaler.scale(total_loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        if isinstance(self.model, _PARALLEL_WRAPPERS):
                            self.model.finish_gradient_synchronization()
                        gradient_norm = self._gradient_norm()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        completed += 1
                        means = {"total_loss": total_loss.detach(), **diagnostics}
                        means = {name: _mean_across_ranks(value.float()).item() for name, value in means.items()}
                        for name, value in means.items():
                            running[name] += value
                        logged_steps += 1
                        progress.update(1)
                        progress.set_postfix(loss=f"{means['total_loss']:.4f}", reward=f"{means['reward']:.3f}")
                        if completed % self.config.log_interval == 0 or completed == 1 or completed == total_steps:
                            if self.logger:
                                self.logger.log(
                                    completed,
                                    "train",
                                    {
                                        "epoch": epoch + 1,
                                        "epoch_step": batches_seen,
                                        **{name: value / logged_steps for name, value in running.items()},
                                        "learning_rate": lr,
                                        "gradient_norm": gradient_norm,
                                    },
                                )
                            running, logged_steps = {name: 0.0 for name in metric_names}, 0
                        if update_index == rounds - 1 and (
                            completed % self.config.save_interval == 0 or completed == total_steps
                        ):
                            save_checkpoint(
                                self.output_dir / f"step_{completed:08d}.pt",
                                self.model,
                                self.optimizer,
                                completed,
                                self.metadata,
                                {
                                    "epoch": epoch + 1 if batches_seen == batches_per_epoch else epoch,
                                    "batch_in_epoch": 0 if batches_seen == batches_per_epoch else batches_seen,
                                },
                            )
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
        finally:
            progress.close()
            if self.vllm_server is not None:
                self.vllm_server.stop()
