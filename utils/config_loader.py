"""
Configuration loading utilities using OmegaConf.

.. deprecated::
    此模块已废弃，请使用 `utils.config` 模块代替。
    
    迁移示例:
        # 旧方式（已废弃）
        from utils.config_loader import load_config
        cfg = load_config("configs/base_config.yaml")
        
        # 新方式（推荐）
        from utils.config import load_config
        cfg = load_config("configs/base_config.yaml")
        print(cfg.model.backbone.name)  # 类型安全的访问
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Union

from omegaconf import DictConfig, OmegaConf

# 发出废弃警告
warnings.warn(
    "utils/config_loader.py 模块已废弃，请使用 utils.config 模块代替。"
    "新模块提供类型安全的配置访问和 dataclass 支持。",
    DeprecationWarning,
    stacklevel=2
)


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
