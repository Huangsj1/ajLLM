"""Generate text from a completed training run."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from ajllm.artifacts.manifests import write_manifest
from ajllm.artifacts.paths import ArtifactPaths
from ajllm.artifacts.registry import load_tokenizer_artifact
from ajllm.config.loader import LoadedConfig
from ajllm.generation.generator import generate
from ajllm.tokenization.tokenizer import Tokenizer
from ajllm.utils.device import resolve_device
from ajllm.utils.random import seed_everything
from ajllm.workflows.common import load_model_from_run, project_path


def _load_prompts(data: dict[str, Any], project_root: Path) -> list[str]:
    if "prompts" in data:
        return [str(prompt) for prompt in data["prompts"]]
    if "prompt_file" in data:
        path = project_path(project_root, data["prompt_file"])
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise ValueError("Generation config requires prompts or prompt_file")


def run(config: LoadedConfig) -> dict[str, Any]:
    data = config.data
    name = str(data.get("name", "generation"))
    training_run = project_path(config.project_root, data["run_directory"])
    device = resolve_device(str(data.get("device", "auto")))
    seed = int(data.get("seed", 42))
    seed_everything(seed)
    model, training_config, _ = load_model_from_run(training_run, str(data.get("checkpoint", "best")), device)
    tokenizer_name = training_config["tokenizer"]["name"]
    tokenizer_dir, tokenizer_manifest = load_tokenizer_artifact(config.project_root, tokenizer_name)
    tokenizer = Tokenizer.from_files(
        tokenizer_dir / tokenizer_manifest["files"]["vocab"],
        tokenizer_dir / tokenizer_manifest["files"]["merges"],
        tokenizer_manifest.get("special_tokens", []),
    )

    maximum_tokens = int(data.get("max_new_tokens", 100))
    temperature = float(data.get("temperature", 0.8))
    top_p = data.get("top_p", 0.95)
    top_p = None if top_p is None else float(top_p)
    samples = int(data.get("samples_per_prompt", 1))
    eos_token = data.get("eos_token", "<|endoftext|>")
    eos_token_id = tokenizer.token_to_id.get(str(eos_token).encode("utf-8"))
    results = []
    for prompt in _load_prompts(data, config.project_root):
        prompt_ids = tokenizer.encode(prompt)
        completions = []
        for _ in range(samples):
            started = time.perf_counter()
            generated_ids = (
                generate(
                    model,
                    torch.tensor(prompt_ids, dtype=torch.long, device=device),
                    maximum_tokens,
                    temperature,
                    top_p,
                    eos_token_id,
                )
                .cpu()
                .tolist()
            )
            elapsed = time.perf_counter() - started
            completions.append(
                {
                    "text": tokenizer.decode(generated_ids),
                    "token_count": len(generated_ids),
                    "elapsed_seconds": elapsed,
                }
            )
        results.append({"prompt": prompt, "completions": completions})

    output_dir = ArtifactPaths(config.project_root).task_run("generations", name)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "training_run": str(training_run),
        "checkpoint": str(data.get("checkpoint", "best")),
        "parameters": {
            "max_new_tokens": maximum_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "samples_per_prompt": samples,
            "seed": seed,
        },
        "results": results,
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)
    write_manifest(
        output_dir / "manifest.json",
        "generation",
        {key: value for key, value in payload.items() if key != "results"},
    )
    return {"output_dir": str(output_dir), **payload}
