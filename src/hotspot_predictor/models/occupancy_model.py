"""Training and evaluation utilities for the occupancy regressor."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


@dataclass
class TrainArtifacts:
    """Container for persisted artifacts."""

    model_path: Path
    metrics_path: Path
    feature_importance_path: Path


def split_datasets(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 2025,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split dataset into train/val/test preserving street distribution."""
    x_temp, x_test, y_temp, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=features["street"],
    )

    adjusted_val_size = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=x_temp["street"],
    )

    logger.info(
        "Split data — train: {} rows, val: {} rows, test: {} rows",
        len(x_train),
        len(x_val),
        len(x_test),
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def train_lightgbm(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    *,
    params: Dict,
) -> lgb.Booster:
    """Train LightGBM regressor with validation based early stopping."""
    cat_features = ["street"] if "street" in x_train.columns else []
    train_set = lgb.Dataset(x_train, label=y_train, categorical_feature=cat_features or None, free_raw_data=False)
    val_set = lgb.Dataset(x_val, label=y_val, reference=train_set, categorical_feature=cat_features or None, free_raw_data=False)

    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=50),
    ]

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    logger.info("Trained LightGBM — best iteration {}", booster.best_iteration)
    return booster


def evaluate_model(booster: lgb.Booster, x_data: pd.DataFrame, y_data: pd.Series) -> Dict[str, float]:
    """Return RMSE and MAE metrics."""
    preds = booster.predict(x_data, num_iteration=booster.best_iteration)
    rmse = float(np.sqrt(mean_squared_error(y_data, preds)))
    mae = float(mean_absolute_error(y_data, preds))
    return {"rmse": rmse, "mae": mae}


def save_artifacts(
    booster: lgb.Booster,
    metrics: Dict[str, object],
    feature_columns: Iterable[str],
    artifacts_dir: Path,
) -> TrainArtifacts:
    """Persist model, metrics, and feature importances."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "lightgbm_occupancy.txt"
    metrics_path = artifacts_dir / "metrics.json"
    importance_path = artifacts_dir / "feature_importance.csv"

    booster.save_model(str(model_path))

    import json

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    importance = pd.DataFrame({
        "feature": feature_columns,
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    })
    importance.to_csv(importance_path, index=False)

    logger.info("Saved model artifacts to {}", artifacts_dir)
    return TrainArtifacts(model_path=model_path, metrics_path=metrics_path, feature_importance_path=importance_path)
