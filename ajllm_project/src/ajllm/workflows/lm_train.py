"""Create a self-contained language-model training run."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import yaml

from ajllm.artifacts.manifests import write_manifest
from ajllm.artifacts.paths import ArtifactPaths
from ajllm.artifacts.registry import load_encoded_artifact, load_tokenizer_artifact
from ajllm.config.loader import LoadedConfig
from ajllm.config.validation import validate_dataset, validate_model, validate_tokenizer
from ajllm.modeling.factory import build_model
from ajllm.reporting.model_report import write_model_report
from ajllm.training.data import TokenDataset
from ajllm.training.trainer import Trainer
from ajllm.utils.device import explain_device_selection, resolve_device
from ajllm.utils.random import seed_everything


def run(config: LoadedConfig, run_directory: Path | None = None) -> dict[str, Any]:
    data = config.data
    dataset_config = data["dataset"]
    tokenizer_config = data["tokenizer"]
    model_config = data["model"]
    validate_dataset(dataset_config)
    validate_tokenizer(tokenizer_config)
    validate_model(model_config)

    tokenizer_dir, tokenizer_manifest = load_tokenizer_artifact(config.project_root, tokenizer_config["name"])
    encoded_dir, encoded_manifest = load_encoded_artifact(
        config.project_root,
        dataset_config["name"],
        tokenizer_config["name"],
    )

    experiment = str(data.get("experiment", "training"))
    paths = ArtifactPaths(config.project_root)
    run_directory = run_directory or paths.training_run(
        dataset_config["name"],
        tokenizer_config["name"],
        model_config["name"],
        experiment,
        data.get("run_id"),
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    vocab_size = int(tokenizer_manifest["vocab_size"])
    data["_runtime"] = {
        "vocab_size": vocab_size,
        "tokenizer_directory": str(tokenizer_dir),
        "encoded_directory": str(encoded_dir),
        "run_directory": str(run_directory),
    }
    resume_from = data.get("training", {}).get("resume_from")
    if resume_from:
        resume_path = Path(str(resume_from))
        if not resume_path.is_absolute():
            resume_path = (config.project_root / resume_path).resolve()
        data["training"]["resume_from"] = str(resume_path)
    with (run_directory / "resolved_config.yaml").open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(data, output_file, sort_keys=False, allow_unicode=True)

    split_names = data.get("data_splits", {"train": "train", "validation": "validation"})
    train_split = split_names["train"]
    validation_split = split_names.get("validation")
    train_dataset = TokenDataset(encoded_dir / encoded_manifest["splits"][train_split]["path"])
    validation_dataset = (
        TokenDataset(encoded_dir / encoded_manifest["splits"][validation_split]["path"])
        if validation_split and validation_split in encoded_manifest["splits"]
        else None
    )

    runtime = data.get("runtime", {})
    seed = int(runtime.get("seed", 42))
    requested_device = str(runtime.get("device", "auto"))
    device = resolve_device(requested_device)
    seed_everything(seed)

    # Check acceleration settings
    acceleration = data.get("acceleration", {})
    use_flash_attention = bool(acceleration.get("use_flash_attention", False))
    use_fsdp = bool(acceleration.get("use_fsdp", False))
    mixed_precision = acceleration.get("mixed_precision")  # None, "fp16", or "bf16"

    # Initialize distributed environment if FSDP is enabled
    if use_fsdp:
        if not torch.distributed.is_available():
            raise RuntimeError("FSDP requires PyTorch with distributed support")

        # Check if we're in a torchrun environment
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # Initialize process group if not already done
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
                # Set device to local rank
                if device.type == "cuda":
                    local_rank = int(os.environ.get("LOCAL_RANK", 0))
                    torch.cuda.set_device(local_rank)
                    device = torch.device("cuda", local_rank)
        else:
            raise RuntimeError(
                "FSDP requires distributed environment. Launch with torchrun or lm_train_distributed.py"
            )

    # Validate mixed_precision setting
    if mixed_precision is not None:
        if mixed_precision not in ["fp16", "bf16"]:
            raise ValueError(f"mixed_precision must be 'fp16', 'bf16', or null, got: {mixed_precision}")
        if device.type not in ["cuda", "mps"]:
            raise ValueError(f"Mixed precision requires CUDA or MPS device, got: {device.type}")

    # Convert mixed_precision to torch dtype
    compute_dtype = None
    if mixed_precision == "fp16":
        compute_dtype = torch.float16
    elif mixed_precision == "bf16":
        compute_dtype = torch.bfloat16

    model = build_model(model_config, vocab_size, use_flash_attention)

    # Move model to device before FSDP wrapping (FSDP broadcasts tensors during init)
    model = model.to(device)

    # Wrap with FSDP if enabled (distributed environment already initialized above)
    if use_fsdp:
        from ajllm.training.distributed import FullyShardedDataParallel
        model = FullyShardedDataParallel(model, compute_dtype=compute_dtype)

    training_config = data.get("training", {})
    batch_size = int(training_config.get("batch_size", 64))
    # Access context_length through .module if wrapped with FSDP
    base_model = model.module if use_fsdp else model
    context_length = int(base_model.context_length)
    tokens_per_batch = batch_size * context_length
    if "max_steps" in training_config:
        training_steps = int(training_config["max_steps"])
    elif "total_tokens" in training_config:
        training_steps = max(1, int(training_config["total_tokens"]) // tokens_per_batch)
    else:
        raise ValueError("training.max_steps or training.total_tokens is required")
    available_windows = max(0, len(train_dataset) - context_length)
    batches_per_pass = max(1, available_windows // tokens_per_batch) if available_windows else 0
    print("Training setup:")
    print(f"  device: {device} (requested: {requested_device})")
    device_note = explain_device_selection(requested_device, device)
    if device_note:
        print(f"  device note: {device_note}")
        if torch.cuda.is_available():
            print(f"  CUDA devices: {torch.cuda.device_count()}")
    print(f"  train tokens: {len(train_dataset):,}")
    if validation_dataset is not None:
        print(f"  validation tokens: {len(validation_dataset):,}")
    print(f"  batch size: {batch_size:,}")
    print(f"  context length: {context_length:,}")
    print(f"  tokens per batch: {tokens_per_batch:,}")
    print(f"  training batches (steps): {training_steps:,}")
    print(f"  planned training tokens: {training_steps * tokens_per_batch:,}")
    print(f"  batches per nominal dataset pass: {batches_per_pass:,}")
    print(f"  model parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"  flash attention: {'enabled' if use_flash_attention else 'disabled'}")
    print(f"  FSDP: {'enabled' if use_fsdp else 'disabled'}")
    print(f"  mixed precision: {mixed_precision if mixed_precision else 'disabled'}")

    # Only rank 0 writes reports in distributed training
    if not use_fsdp or torch.distributed.get_rank() == 0:
        write_model_report(
            run_directory / "model_report.md",
            base_model,  # Pass unwrapped model
            model_config,
            vocab_size,
            batch_size,
            training_config,
        )

    manifest = {
        "experiment": experiment,
        "dataset": dataset_config["name"],
        "tokenizer": tokenizer_config["name"],
        "model": model_config["name"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
    }

    # Only rank 0 writes manifest in distributed training
    if not use_fsdp or torch.distributed.get_rank() == 0:
        write_manifest(run_directory / "manifest.json", "training_run", manifest)
    trainer = Trainer(model, train_dataset, validation_dataset, run_directory, data, device, seed)
    summary = trainer.train(
        {
            **manifest,
            "resolved_config": data,
            "vocab_size": vocab_size,
        }
    )
    return {"run_directory": str(run_directory), **manifest, **summary}
