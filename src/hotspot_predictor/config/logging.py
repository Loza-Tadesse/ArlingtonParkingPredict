"""Logging helpers for pipelines."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger


def setup_logging(log_dir: Path | str = "logs", level: str = "INFO") -> None:
    """Configure loguru sinks for console and rotating file."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level=level,
        colorize=True,
    )

    log_file = log_path / f"hotspot_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level=level,
        rotation="100 MB",
        retention="30 days",
    )

    logger.info("Logging initialized: {}", log_file)
