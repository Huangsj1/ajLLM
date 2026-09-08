"""Command-line entry point for JSONL causal-LM pre-training."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler

from ajllm.datasets import PretrainDataset
from ajllm.modeling import build_model, load_model_config
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


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as source:
        return yaml.safe_load(source) or {}


def _load_tokenizer(config: dict[str, Any]) -> MiniMindTokenizer:
    """Load the one tokenizer supported by the pre-training contract."""
    tokenizer_config = config.get("tokenizer")
    if not isinstance(tokenizer_config, dict) or set(tokenizer_config) != {"path"}:
        raise ValueError("tokenizer must be exactly: {path: assets/tokenizers/minimind}")
    return MiniMindTokenizer.from_pretrained(tokenizer_config["path"])


def run(config_path: str | Path) -> dict[str, float | int | bool]:
    """Construct data/model/trainer and execute one pre-training run."""
    config = _load_yaml(config_path)
    device, rank, world_size = initialize_distributed()
    try:
        if config.get("device", "auto") == "cpu":
            device = torch.device("cpu")
        elif config.get("device") == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("config requests CUDA but CUDA is unavailable")
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
            raise ValueError(
                "torchrun requires use_fsdp: true, parallel.tp_size > 1, or parallel.ep_size > 1"
            )
        if config.get("use_fsdp", False) and (tp_size > 1 or ep_size > 1):
            raise ValueError("combining FSDP with TP/EP will be enabled after the independent combination test suites")
        output_dir = Path(config["output_dir"])
        if is_main_process():
            output_dir.mkdir(parents=True, exist_ok=True)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        # TP replicas share a data/RNG stream; EP replicas process distinct data.
        data_rank = rank // tp_size
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
        dataset = PretrainDataset(
            config["data_path"],
            tokenizer,
            sequence_length,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        validation_dataset = (
            PretrainDataset(
                config["validation_data_path"],
                tokenizer,
                sequence_length,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            if config.get("validation_data_path")
            else None
        )
        data_replicas = ep_size
        sampler = (
            DistributedSampler(
                dataset, num_replicas=data_replicas, rank=data_rank, shuffle=True, seed=int(config.get("seed", 42))
            )
            if data_replicas > 1 or tp_size > 1
            else None
        )
        dataloader = DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=sampler is None,
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
        if config.get("use_fsdp", False):
            if world_size == 1:
                raise ValueError("use_fsdp requires launch with torchrun and at least two processes")
            fsdp_config = config.get("fsdp_config", {})
            if fsdp_config.get("sharding_strategy", "FULL_SHARD") != "FULL_SHARD":
                raise ValueError("the educational FSDP implementation currently supports FULL_SHARD only")
            if fsdp_config.get("cpu_offload", False):
                raise ValueError("cpu_offload is not implemented by the educational FSDP wrapper")
            precision = config.get("mixed_precision")
            compute_dtype = torch.bfloat16 if precision == "bf16" else torch.float16 if precision == "fp16" else None
            model = FullyShardedDataParallel(
                model, compute_dtype, bool(fsdp_config.get("use_activation_checkpointing", True))
            )
        elif tp_size > 1 and ep_size > 1:
            context = ParallelContext.from_distributed(tp_size=tp_size, ep_size=ep_size)
            model = tensor_expert_parallelize(
                model,
                context,
                expert_backend=str(parallel_config.get("expert_backend", "torch")),
            )
        elif tp_size > 1:
            model = tensor_parallelize(model)
        elif ep_size > 1:
            model = expert_parallelize(model, expert_backend=str(parallel_config.get("expert_backend", "torch")))
        if is_main_process():
            base_model = (
                model.module
                if isinstance(model, (FullyShardedDataParallel, TensorParallel, ExpertParallel, TensorExpertParallel))
                else model
            )
            parameter_count = (
                model.parameter_count()
                if isinstance(model, (TensorParallel, ExpertParallel, TensorExpertParallel))
                else base_model.parameter_count()
            )
            print(f"parameters: {parameter_count:,}")
            print(
                f"dataset records: {len(dataset):,}; epochs: {config['epochs']}; "
                f"device: {device}; world size: {world_size}"
            )
            (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
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
            {"model_config": model_config.__dict__, "training_config": config},
            validation_dataloader,
        )
        return trainer.train()
    finally:
        destroy_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train ajLLM on JSONL records containing a text field")
    parser.add_argument("--config", required=True, help="Training YAML path")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
