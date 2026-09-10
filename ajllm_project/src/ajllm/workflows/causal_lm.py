"""Shared wiring for causal-LM pre-training and supervised fine-tuning."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler

from ajllm.datasets import PretrainDataset, SFTDataset
from ajllm.modeling import ModelConfig, build_model, load_model_config
from ajllm.tokenization import MiniMindTokenizer
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
from ajllm.training.samplers import ResumableDistributedSampler, ResumableRandomSampler

TrainingStage = Literal["pretrain", "sft"]
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


def _load_pretrained_weights(
    model: torch.nn.Module, model_config: ModelConfig, checkpoint_path: str | Path
) -> None:
    """Load portable pre-training weights before applying a parallel wrapper."""
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
            "pretrained_checkpoint model_config differs from model_config; SFT must use the identical architecture"
        )
    model.load_state_dict(_upgrade_legacy_top1_moe_state_dict(state_dict, model))


def _build_dataset(
    stage: TrainingStage, data_path: str | Path, tokenizer: MiniMindTokenizer, sequence_length: int
) -> PretrainDataset | SFTDataset:
    if stage == "pretrain":
        return PretrainDataset(
            data_path,
            tokenizer,
            sequence_length,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return SFTDataset(data_path, tokenizer, sequence_length, pad_token_id=tokenizer.pad_token_id)


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
    """Run either causal-LM stage; the dataset and initialization policy differ."""
    config = _load_yaml(config_path)
    if stage == "sft":
        if config.get("resume_from") and config.get("pretrained_checkpoint"):
            raise ValueError("resume_from and pretrained_checkpoint are mutually exclusive")
        if not config.get("resume_from") and not isinstance(config.get("pretrained_checkpoint"), str):
            raise ValueError("SFT requires pretrained_checkpoint unless resuming an SFT checkpoint")
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
        sequence_length = int(config.get("max_seq_len", model_config.context_length))
        if sequence_length > model_config.context_length:
            raise ValueError("max_seq_len cannot exceed model context_length")
        dataset = _build_dataset(stage, config["data_path"], tokenizer, sequence_length)
        validation_dataset = (
            _build_dataset(stage, config["validation_data_path"], tokenizer, sequence_length)
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
        model = _wrap_model(model, config, parallel_config, world_size)
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
        if stage == "sft":
            metadata["stage"] = "sft"
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
