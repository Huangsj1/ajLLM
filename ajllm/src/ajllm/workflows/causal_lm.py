"""Shared wiring for pre-training, SFT, and preference-training workflows."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler

from ajllm.datasets import DPODataset, PretrainDataset, RLAIFDataset, SFTDataset, collate_rlaif
from ajllm.modeling import ModelConfig, build_model, load_model_config
from ajllm.tokenization import MiniMindTokenizer
from ajllm.training.dpo_trainer import DPOConfig, DPOTrainer
from ajllm.training.grpo_trainer import GRPOConfig, GRPOTrainer
from ajllm.training.parallel import (
    ExpertParallel,
    FullyShardedDataParallel,
    ParallelContext,
    TensorExpertParallel,
    TensorParallel,
    destroy_distributed,
    expert_parallelize,
    initialize_distributed,
    tensor_expert_parallelize,
    tensor_parallelize,
)
from ajllm.training.pretrainer import PretrainConfig, Pretrainer, is_main_process
from ajllm.training.rewards import SkyworkRewardModel
from ajllm.training.samplers import ResumableDistributedSampler, ResumableRandomSampler

TrainingStage = Literal["pretrain", "sft", "dpo", "grpo"]
_PARALLEL_WRAPPERS = (FullyShardedDataParallel, TensorParallel, ExpertParallel, TensorExpertParallel)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as source:
        return yaml.safe_load(source) or {}


def _load_tokenizer(config: dict[str, Any]) -> MiniMindTokenizer:
    tokenizer_config = config.get("tokenizer")
    if not isinstance(tokenizer_config, dict) or set(tokenizer_config) != {"path"}:
        raise ValueError("tokenizer must be exactly: {path: assets/tokenizers/minimind}")
    return MiniMindTokenizer.from_pretrained(tokenizer_config["path"])


def _upgrade_legacy_top1_moe_state_dict(
    state_dict: dict[str, torch.Tensor], model: torch.nn.Module
) -> dict[str, torch.Tensor]:
    """Pack old per-expert Top-1 MoE weights into the grouped-expert layout."""
    expected_state = model.state_dict()
    converted = dict(state_dict)
    grouped_suffix = ".feed_forward.grouped_experts"
    for grouped_key, expected_weight in expected_state.items():
        if not grouped_key.endswith(f"{grouped_suffix}.gate_up_proj.weight") or grouped_key in converted:
            continue
        layer_prefix = grouped_key.removesuffix(".grouped_experts.gate_up_proj.weight")
        down_key = f"{layer_prefix}.grouped_experts.down_proj.weight"
        expert_count = expected_weight.shape[0]
        legacy_keys = [
            f"{layer_prefix}.experts.{expert_index}.{projection}_proj.weight"
            for expert_index in range(expert_count)
            for projection in ("gate", "up", "down")
        ]
        present_legacy_keys = [key for key in legacy_keys if key in converted]
        if not present_legacy_keys:
            continue
        missing_legacy_keys = [key for key in legacy_keys if key not in converted]
        if missing_legacy_keys:
            raise ValueError(
                "Legacy MoE checkpoint has an incomplete expert state for "
                f"{layer_prefix}: missing {', '.join(missing_legacy_keys)}"
            )
        grouped_gate_up = torch.stack(
            [
                torch.cat(
                    (
                        converted[f"{layer_prefix}.experts.{expert_index}.gate_proj.weight"],
                        converted[f"{layer_prefix}.experts.{expert_index}.up_proj.weight"],
                    ),
                    dim=0,
                )
                for expert_index in range(expert_count)
            ]
        )
        grouped_down = torch.stack(
            [
                converted[f"{layer_prefix}.experts.{expert_index}.down_proj.weight"]
                for expert_index in range(expert_count)
            ]
        )
        expected_down = expected_state.get(down_key)
        if (
            grouped_gate_up.shape != expected_weight.shape
            or expected_down is None
            or grouped_down.shape != expected_down.shape
        ):
            raise ValueError(
                f"Legacy MoE expert shapes for {layer_prefix} do not match the configured grouped-expert architecture"
            )
        converted[grouped_key] = grouped_gate_up
        converted[down_key] = grouped_down
        for key in legacy_keys:
            del converted[key]
    return converted


def _load_pretrained_weights(model: torch.nn.Module, model_config: ModelConfig, checkpoint_path: str | Path) -> None:
    """Load portable checkpoint weights before applying a parallel wrapper."""
    source = Path(checkpoint_path)
    if not source.is_file():
        raise FileNotFoundError(f"pretrained_checkpoint does not exist: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"{source} is not an ajLLM portable model checkpoint")
    saved_config = checkpoint.get("metadata", {}).get("model_config")
    if saved_config is not None and saved_config != model_config.__dict__:
        raise ValueError(
            "checkpoint model_config differs from model_config; "
            "the initialized model must use the identical architecture"
        )
    model.load_state_dict(_upgrade_legacy_top1_moe_state_dict(state_dict, model))


def _build_dataset(
    stage: TrainingStage,
    data_path: str | Path,
    tokenizer: MiniMindTokenizer,
    sequence_length: int,
    *,
    open_thinking: bool = False,
) -> PretrainDataset | SFTDataset | DPODataset | RLAIFDataset:
    if stage == "pretrain":
        return PretrainDataset(
            data_path,
            tokenizer,
            sequence_length,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    elif stage == "sft":
        return SFTDataset(data_path, tokenizer, sequence_length, pad_token_id=tokenizer.pad_token_id)
    elif stage == "dpo":
        return DPODataset(data_path, tokenizer, sequence_length, pad_token_id=tokenizer.pad_token_id)
    elif stage == "grpo":
        return RLAIFDataset(data_path, tokenizer, sequence_length, open_thinking=open_thinking)
    raise ValueError(f"Unknown training stage: {stage}")


def _wrap_model(
    model: torch.nn.Module,
    config: dict[str, Any],
    parallel_config: dict[str, Any],
    world_size: int,
) -> torch.nn.Module:
    tp_size = int(parallel_config.get("tp_size", 1))
    ep_size = int(parallel_config.get("ep_size", 1))
    if config.get("use_fsdp", False):
        if world_size == 1:
            raise ValueError("use_fsdp requires torchrun and at least two processes")
        fsdp_config = config.get("fsdp_config", {})
        if fsdp_config.get("sharding_strategy", "FULL_SHARD") != "FULL_SHARD":
            raise ValueError("the educational FSDP implementation supports FULL_SHARD only")
        if fsdp_config.get("cpu_offload", False):
            raise ValueError("cpu_offload is not implemented by the educational FSDP wrapper")
        precision = config.get("mixed_precision")
        compute_dtype = torch.bfloat16 if precision == "bf16" else torch.float16 if precision == "fp16" else None
        return FullyShardedDataParallel(
            model, compute_dtype, bool(fsdp_config.get("use_activation_checkpointing", True))
        )
    if tp_size > 1 and ep_size > 1:
        context = ParallelContext.from_distributed(tp_size=tp_size, ep_size=ep_size)
        return tensor_expert_parallelize(
            model, context, expert_backend=str(parallel_config.get("expert_backend", "torch"))
        )
    if tp_size > 1:
        return tensor_parallelize(model)
    if ep_size > 1:
        return expert_parallelize(model, expert_backend=str(parallel_config.get("expert_backend", "torch")))
    return model


def _validate_parallel_config(config: dict[str, Any], world_size: int) -> tuple[dict[str, Any], int, int]:
    parallel_config = config.get("parallel", {})
    if not isinstance(parallel_config, dict):
        raise ValueError("parallel must be a mapping when provided")
    tp_size = int(parallel_config.get("tp_size", 1))
    ep_size = int(parallel_config.get("ep_size", 1))
    if tp_size < 1 or ep_size < 1:
        raise ValueError("parallel.tp_size and parallel.ep_size must be positive")
    if not config.get("use_fsdp", False) and tp_size * ep_size != world_size:
        raise ValueError("parallel.tp_size * parallel.ep_size must equal WORLD_SIZE")
    if world_size > 1 and not config.get("use_fsdp", False) and tp_size == ep_size == 1:
        raise ValueError("torchrun requires use_fsdp: true, parallel.tp_size > 1, or parallel.ep_size > 1")
    if config.get("use_fsdp", False) and (tp_size > 1 or ep_size > 1):
        raise ValueError("combining FSDP with TP/EP is not implemented")
    return parallel_config, tp_size, ep_size


def run_causal_lm(config_path: str | Path, stage: TrainingStage) -> dict[str, float | int | bool]:
    """Run a shared causal-LM stage; data and trainer are stage-specific."""
    config = _load_yaml(config_path)
    if stage == "sft":
        if config.get("resume_from") and config.get("pretrained_checkpoint"):
            raise ValueError("resume_from and pretrained_checkpoint are mutually exclusive")
        if not config.get("resume_from") and not isinstance(config.get("pretrained_checkpoint"), str):
            raise ValueError("SFT requires pretrained_checkpoint unless resuming an SFT checkpoint")
    if stage == "dpo" and not isinstance(config.get("sft_checkpoint"), str):
        raise ValueError("DPO requires sft_checkpoint: path/to/sft_checkpoint.pt")
    if stage == "grpo":
        if not isinstance(config.get("sft_checkpoint"), str):
            raise ValueError("GRPO requires sft_checkpoint: path/to/sft_checkpoint.pt")
        reward_config = config.get("reward_model")
        if not isinstance(reward_config, dict) or not isinstance(reward_config.get("path"), str):
            raise ValueError("GRPO requires reward_model: {path: reward_model/Skywork-Reward-V2-Qwen3-0.6B}")
        if config.get("validation_data_path"):
            raise ValueError("GRPO uses online rollouts and does not support validation_data_path")
    device, rank, world_size = initialize_distributed()
    try:
        if config.get("device", "auto") == "cpu":
            device = torch.device("cpu")
        elif config.get("device") == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("config requests CUDA but CUDA is unavailable")
        parallel_config, tp_size, ep_size = _validate_parallel_config(config, world_size)
        output_dir = Path(config["output_dir"])
        if is_main_process():
            output_dir.mkdir(parents=True, exist_ok=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        use_fsdp = bool(config.get("use_fsdp", False))
        data_replicas = world_size if use_fsdp else ep_size
        data_rank = rank if use_fsdp else rank // tp_size
        seed = int(config.get("seed", 42)) + data_rank
        random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        tokenizer = _load_tokenizer(config)
        model_config = load_model_config(config["model_config"], tokenizer.vocab_size)
        if stage == "grpo":
            sequence_length = int(config.get("max_prompt_len", model_config.context_length))
            max_new_tokens = int(config.get("max_new_tokens", 256))
            if sequence_length + max_new_tokens > model_config.context_length:
                raise ValueError("max_prompt_len + max_new_tokens cannot exceed model context_length")
        else:
            sequence_length = int(config.get("max_seq_len", model_config.context_length))
            if sequence_length > model_config.context_length:
                raise ValueError("max_seq_len cannot exceed model context_length")
        dataset = _build_dataset(
            stage,
            config["data_path"],
            tokenizer,
            sequence_length,
            open_thinking=bool(config.get("open_thinking", False)),
        )
        validation_dataset = (
            _build_dataset(
                stage,
                config["validation_data_path"],
                tokenizer,
                sequence_length,
                open_thinking=bool(config.get("open_thinking", False)),
            )
            if config.get("validation_data_path")
            else None
        )
        sampler = (
            ResumableDistributedSampler(
                dataset, num_replicas=data_replicas, rank=data_rank, shuffle=True, seed=int(config.get("seed", 42))
            )
            if data_replicas > 1
            else ResumableRandomSampler(dataset, seed=int(config.get("seed", 42)))
        )
        dataloader = DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            sampler=sampler,
            num_workers=int(config.get("num_workers", 0)),
            pin_memory=device.type == "cuda",
            collate_fn=collate_rlaif if stage == "grpo" else None,
        )
        validation_sampler = (
            DistributedSampler(validation_dataset, num_replicas=data_replicas, rank=data_rank, shuffle=False)
            if validation_dataset is not None and (data_replicas > 1 or tp_size > 1)
            else None
        )
        validation_dataloader = (
            DataLoader(
                validation_dataset,
                batch_size=int(config.get("eval_batch_size", config["batch_size"])),
                sampler=validation_sampler,
                num_workers=int(config.get("num_workers", 0)),
                pin_memory=device.type == "cuda",
            )
            if validation_dataset is not None
            else None
        )
        model = build_model(model_config).to(device)
        if stage == "sft" and config.get("pretrained_checkpoint"):
            _load_pretrained_weights(model, model_config, config["pretrained_checkpoint"])
        reference_model: torch.nn.Module | None = None
        if stage in {"dpo", "grpo"}:
            reference_model = build_model(model_config).to(device)
            _load_pretrained_weights(model, model_config, config["sft_checkpoint"])
            _load_pretrained_weights(reference_model, model_config, config["sft_checkpoint"])
        model = _wrap_model(model, config, parallel_config, world_size)
        if reference_model is not None:
            reference_model = _wrap_model(reference_model, config, parallel_config, world_size)
        if is_main_process():
            base_model = model.module if isinstance(model, _PARALLEL_WRAPPERS) else model
            parameter_count = (
                model.parameter_count() if isinstance(model, _PARALLEL_WRAPPERS[1:]) else base_model.parameter_count()
            )
            print(f"parameters: {parameter_count:,}")
            print(
                f"{stage} dataset records: {len(dataset):,}; epochs: {config['epochs']}; "
                f"device: {device}; world size: {world_size}"
            )
            (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        metadata: dict[str, Any] = {"model_config": model_config.__dict__, "training_config": config}
        if stage in {"sft", "dpo", "grpo"}:
            metadata["stage"] = stage
        if stage in {"dpo", "grpo"}:
            metadata["reference_checkpoint"] = config["sft_checkpoint"]
        if stage == "grpo":
            metadata["reward_model"] = config["reward_model"]
        if stage == "dpo":
            assert reference_model is not None
            trainer = DPOTrainer(
                model,
                reference_model,
                dataloader,
                device,
                DPOConfig(
                    epochs=int(config["epochs"]),
                    max_steps=int(config["max_steps"]) if config.get("max_steps") is not None else None,
                    learning_rate=float(config["learning_rate"]),
                    min_lr=float(config["min_lr"]),
                    warmup_steps=int(config.get("warmup_steps", 0)),
                    gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
                    weight_decay=float(config.get("weight_decay", 0.1)),
                    beta1=float(config.get("beta1", 0.9)),
                    beta2=float(config.get("beta2", 0.95)),
                    grad_clip=float(config.get("grad_clip", 1.0)),
                    mixed_precision=config.get("mixed_precision"),
                    log_interval=int(config.get("log_interval", 50)),
                    eval_interval=int(config.get("eval_interval", 0)),
                    eval_batches=config.get("eval_batches"),
                    save_interval=int(config.get("save_interval", 2_000)),
                    resume_from=config.get("resume_from"),
                    beta=float(config.get("beta", 0.1)),
                ),
                output_dir,
                metadata,
                validation_dataloader,
            )
            return trainer.train()
        if stage == "grpo":
            assert reference_model is not None
            reward_config = config["reward_model"]
            reward_model = SkyworkRewardModel(
                reward_config["path"],
                device,
                max_length=int(reward_config.get("max_length", 4096)),
                dtype=torch.bfloat16 if config.get("mixed_precision") == "bf16" else torch.float16,
            )
            trainer = GRPOTrainer(
                model,
                reference_model,
                reward_model,
                tokenizer,
                dataloader,
                device,
                GRPOConfig(
                    epochs=int(config["epochs"]),
                    max_steps=int(config["max_steps"]) if config.get("max_steps") is not None else None,
                    learning_rate=float(config["learning_rate"]),
                    min_lr=float(config["min_lr"]),
                    warmup_steps=int(config.get("warmup_steps", 0)),
                    gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
                    weight_decay=float(config.get("weight_decay", 0.1)),
                    beta1=float(config.get("beta1", 0.9)),
                    beta2=float(config.get("beta2", 0.95)),
                    grad_clip=float(config.get("grad_clip", 1.0)),
                    mixed_precision=config.get("mixed_precision"),
                    log_interval=int(config.get("log_interval", 50)),
                    eval_interval=0,
                    eval_batches=None,
                    save_interval=int(config.get("save_interval", 2_000)),
                    resume_from=config.get("resume_from"),
                    num_generations=int(config.get("num_generations", 4)),
                    updates_per_rollout=int(config.get("updates_per_rollout", 4)),
                    max_new_tokens=max_new_tokens,
                    temperature=float(config.get("temperature", 0.8)),
                    top_k=int(config.get("top_k", 50)),
                    top_p=float(config.get("top_p", 0.95)),
                    kl_coef=float(config.get("kl_coef", 0.04)),
                    clip_epsilon=float(config.get("clip_epsilon", 0.2)),
                    advantage_epsilon=float(config.get("advantage_epsilon", 1e-4)),
                    rollout_backend=str(config.get("rollout_backend", "torch")),
                    vllm=config.get("vllm"),
                ),
                output_dir,
                metadata,
            )
            return trainer.train()
        trainer = Pretrainer(
            model,
            dataloader,
            device,
            PretrainConfig(
                epochs=int(config["epochs"]),
                max_steps=int(config["max_steps"]) if config.get("max_steps") is not None else None,
                learning_rate=float(config["learning_rate"]),
                min_lr=float(config["min_lr"]),
                warmup_steps=int(config.get("warmup_steps", 0)),
                gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
                weight_decay=float(config.get("weight_decay", 0.1)),
                beta1=float(config.get("beta1", 0.9)),
                beta2=float(config.get("beta2", 0.95)),
                grad_clip=float(config.get("grad_clip", 1.0)),
                mixed_precision=config.get("mixed_precision"),
                log_interval=int(config.get("log_interval", 50)),
                eval_interval=int(config.get("eval_interval", 0)),
                eval_batches=config.get("eval_batches"),
                save_interval=int(config.get("save_interval", 2_000)),
                resume_from=config.get("resume_from"),
            ),
            output_dir,
            metadata,
            validation_dataloader,
        )
        return trainer.train()
    finally:
        destroy_distributed()


def run_pretrain(config_path: str | Path) -> dict[str, float | int | bool]:
    return run_causal_lm(config_path, "pretrain")


def run_sft(config_path: str | Path) -> dict[str, float | int | bool]:
    return run_causal_lm(config_path, "sft")


def run_dpo(config_path: str | Path) -> dict[str, float | int | bool]:
    return run_causal_lm(config_path, "dpo")


def run_grpo(config_path: str | Path) -> dict[str, float | int | bool]:
    return run_causal_lm(config_path, "grpo")
