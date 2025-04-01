"""Configuration helpers for the parking occupancy project."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from loguru import logger

DEFAULT_CONFIG_PATH = Path(os.getenv("HOTSPOT_CONFIG", "config.yaml"))


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load project configuration from YAML."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        # Resolve relative to project root (cwd)
        path = Path.cwd() / path

    if not path.exists():
        logger.warning("Config file not found: {}", path)
        return {}

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    logger.info("Loaded configuration from {}", path)
    return config
