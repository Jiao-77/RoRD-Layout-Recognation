"""Configuration loading utilities using OmegaConf."""
from __future__ import annotations

from pathlib import Path
from typing import Union

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Load a YAML configuration file into a DictConfig."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return OmegaConf.load(path)


def to_absolute_path(path_str: str, base_dir: Union[str, Path]) -> Path:
    """Resolve a possibly relative path against the configuration file directory."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(base_dir) / path).resolve()
