"""Streamlit dashboard for visualizing hourly parking occupancy forecasts."""
from __future__ import annotations

import calendar
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hotspot_predictor.config.logging import setup_logging
from hotspot_predictor.config.settings import load_config
from hotspot_predictor.pipelines.train_occupancy import DEFAULT_FEATURES

st.set_page_config(page_title="Arlington Parking Occupancy", page_icon="🅿️", layout="wide")


HOURS = list(range(6, 22 + 1))


@lru_cache(maxsize=1)
def _load_config() -> dict:
    cfg = load_config()
    log_cfg = cfg.get("logging", {})
    setup_logging(log_cfg.get("log_dir", "logs"), log_cfg.get("level", "INFO"))
    return cfg


def _resolve_paths(cfg: dict) -> tuple[Path, Path]:
    features_cfg = cfg.get("features", {})
    model_cfg = cfg.get("model", {})

    processed_dir = Path(features_cfg.get("processed_dir", "data/processed"))
    hourly_output = features_cfg.get("hourly_output", "parking_hourly_occupancy.csv")
    hourly_path = processed_dir / hourly_output

    artifacts_dir = Path(model_cfg.get("artifacts_dir", "models/occupancy"))
    model_path = artifacts_dir / "lightgbm_occupancy.txt"
    return hourly_path, model_path


@st.cache_data(show_spinner=False)
def _load_hourly_frame(hourly_path: Path) -> pd.DataFrame:
    if not hourly_path.exists():
        st.error(f"Hourly occupancy snapshot not found at {hourly_path}.")
        st.stop()
    frame = pd.read_csv(hourly_path)
    frame["street"] = frame["street"].astype(str).str.strip()
    return frame


@st.cache_resource(show_spinner=False)
def _load_model(model_path: Path) -> lgb.Booster:
    if not model_path.exists():
        st.error(f"LightGBM model not found at {model_path}. Train the model before launching the app.")
        st.stop()
    return lgb.Booster(model_file=str(model_path))


def _day_labels() -> List[str]:
    return [calendar.day_name[idx][:3] for idx in range(7)]


def _format_hour_label(hour: int) -> str:
    if hour == 0:
        return "12 AM"
    if hour == 12:
        return "12 PM"
    suffix = "AM" if hour < 12 else "PM"
    value = hour if hour <= 12 else hour - 12
    return f"{value} {suffix}"
