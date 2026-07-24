"""Load composable YAML configuration files."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is missing or inconsistent."""


@dataclass(frozen=True)
class LoadedConfig:
    """A resolved configuration and the project paths used to resolve it."""

    data: dict[str, Any]
    source: Path
    project_root: Path
    config_root: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as config_file:
        value = yaml.safe_load(config_file) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {path}")
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _find_project_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise ConfigError(f"Could not find a project root above {config_path}")


def _load_with_base(path: Path, active: set[Path] | None = None) -> dict[str, Any]:
    active = active or set()
    resolved_path = path.resolve()
    if resolved_path in active:
        raise ConfigError(f"Circular base configuration reference: {resolved_path}")
    active.add(resolved_path)
    config = _load_yaml(resolved_path)
    base_reference = config.pop("base", None)
    if base_reference is not None:
        base_path = (resolved_path.parent / str(base_reference)).resolve()
        config = _deep_merge(_load_with_base(base_path, active), config)
    active.remove(resolved_path)
    return config


def _resolve_component(
    config: dict[str, Any],
    key: str,
    directory: str,
    config_root: Path,
) -> None:
    reference = config.get(key)
    if reference is None or isinstance(reference, dict):
        return
    if not isinstance(reference, str):
        raise ConfigError(f"'{key}' must be a component name or mapping")
    component_path = config_root / directory / f"{reference}.yaml"
    component = _load_with_base(component_path)
    component.setdefault("name", reference)
    config[key] = component


def load_run_config(path: str | Path) -> LoadedConfig:
    """Load a run config, apply inheritance, and resolve named components."""

    source = Path(path).expanduser().resolve()
    project_root = _find_project_root(source)
    config_root = project_root / "configs"
    config = _load_with_base(source)
    _resolve_component(config, "dataset", "datasets", config_root)
    _resolve_component(config, "tokenizer", "tokenizers", config_root)
    _resolve_component(config, "model", "models", config_root)
    config["_meta"] = {
        "source": str(source),
        "project_root": str(project_root),
    }
    return LoadedConfig(config, source, project_root, config_root)


def set_dotted_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested configuration value addressed with dotted notation."""

    keys = dotted_key.split(".")
    current = config
    for key in keys[:-1]:
        child = current.setdefault(key, {})
        if not isinstance(child, dict):
            raise ConfigError(f"Cannot override '{dotted_key}': '{key}' is not a mapping")
        current = child
    current[keys[-1]] = value
