"""End-to-end pipeline for training the parking occupancy model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from loguru import logger

from hotspot_predictor.config.logging import setup_logging
from hotspot_predictor.config.settings import load_config
from hotspot_predictor.data.transactions import download_months
from hotspot_predictor.features.occupancy import (
    build_hourly_occupancy,
    clean_transactions,
    export_hourly,
    load_months,
)
from hotspot_predictor.models.occupancy_model import (
    evaluate_model,
    save_artifacts,
    split_datasets,
    train_lightgbm,
)


@dataclass
class PipelineConfig:
    raw_dir: Path
    processed_dir: Path
    hourly_output: Path
    artifacts_dir: Path
    months: List[Tuple[int, int]]
    force_download: bool
    test_size: float
    val_size: float
    model_params: dict


DEFAULT_FEATURES = [
    "street",
    "day_of_week",
    "hour_of_day",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
]


def _parse_months(items: Iterable[dict]) -> List[Tuple[int, int]]:
    months: List[Tuple[int, int]] = []
    for item in items:
        months.append((int(item["year"]), int(item["month"])))
    return months


def _build_config(config_dict: dict) -> PipelineConfig:
    data_cfg = config_dict.get("data", {}).get("transactions", {})
    features_cfg = config_dict.get("features", {})
    model_cfg = config_dict.get("model", {})

    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    processed_dir = Path(features_cfg.get("processed_dir", "data/processed"))
    hourly_output = processed_dir / features_cfg.get("hourly_output", "parking_hourly_occupancy.csv")
    artifacts_dir = Path(model_cfg.get("artifacts_dir", "models/occupancy"))

    months = _parse_months(data_cfg.get("months", []))

    return PipelineConfig(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        hourly_output=hourly_output,
        artifacts_dir=artifacts_dir,
        months=months,
        force_download=bool(data_cfg.get("force_download", False)),
        test_size=float(model_cfg.get("test_size", 0.15)),
        val_size=float(model_cfg.get("val_size", 0.15)),
        model_params=model_cfg.get("params", {}),
    )


def run(config_path: str | Path | None = None) -> None:
    """Execute the full training pipeline."""
    config_dict = load_config(config_path)

    log_cfg = config_dict.get("logging", {})
    setup_logging(log_cfg.get("log_dir", "logs"), log_cfg.get("level", "INFO"))

    pipeline_config = _build_config(config_dict)
    logger.info("Running occupancy training with months: {}", pipeline_config.months)

    raw_paths = download_months(pipeline_config.months, pipeline_config.raw_dir, force=pipeline_config.force_download)

    transactions = load_months(raw_paths)
    transactions = clean_transactions(transactions)
    hourly = build_hourly_occupancy(transactions)

    export_hourly(hourly, pipeline_config.hourly_output)

    if hourly.empty:
        logger.warning("No hourly data generated; aborting training.")
        return

    feature_frame = hourly[DEFAULT_FEATURES].copy()
    feature_frame["street"] = feature_frame["street"].astype("category")
    target = hourly["occupancy"].copy()

    splits = split_datasets(
        feature_frame,
        target,
        test_size=pipeline_config.test_size,
        val_size=pipeline_config.val_size,
    )
    x_train, x_val, x_test, y_train, y_val, y_test = splits

    booster = train_lightgbm(
        x_train,
        x_val,
        y_train,
        y_val,
        params=pipeline_config.model_params,
    )

    metrics = {
        "train": evaluate_model(booster, x_train, y_train),
        "validation": evaluate_model(booster, x_val, y_val),
        "test": evaluate_model(booster, x_test, y_test),
        "best_iteration": int(booster.best_iteration),
    }

    save_artifacts(
        booster,
        metrics,
        feature_frame.columns,
        pipeline_config.artifacts_dir,
    )

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    run()
