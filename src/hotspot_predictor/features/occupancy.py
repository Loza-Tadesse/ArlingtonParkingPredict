"""Feature engineering for parking occupancy modeling."""
from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from loguru import logger

REQUIRED_COLUMNS = [
    "sourceStreetDisplayName",
    "startDtm",
    "endDtm",
]


def load_months(paths: Iterable[Path]) -> pd.DataFrame:
    """Load raw transaction CSVs into a single DataFrame."""
    dataframes: List[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            logger.warning("Skipping missing raw file {}", path)
            continue
        df = pd.read_csv(path, parse_dates=["startDtm", "endDtm"])
        dataframes.append(df)
        logger.info("Loaded {} rows from {}", len(df), path.name)

    if not dataframes:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    combined = pd.concat(dataframes, ignore_index=True)
    return combined


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and filter unusable rows."""
    if df.empty:
        return df

    data = df.copy()
    data["sourceStreetDisplayName"] = data["sourceStreetDisplayName"].astype(str).str.strip()
    data = data.dropna(subset=REQUIRED_COLUMNS)
    data = data[data["sourceStreetDisplayName"] != ""]

    if data["startDtm"].dt.tz is None:
        data["startDtm"] = data["startDtm"].dt.tz_localize("UTC")
    else:
        data["startDtm"] = data["startDtm"].dt.tz_convert("UTC")

    if data["endDtm"].dt.tz is None:
        data["endDtm"] = data["endDtm"].dt.tz_localize("UTC")
    else:
        data["endDtm"] = data["endDtm"].dt.tz_convert("UTC")

    data = data[data["endDtm"] > data["startDtm"]]
    logger.info("Cleaned transactions: {} rows remaining", len(data))
    return data


def build_hourly_occupancy(transactions: pd.DataFrame) -> pd.DataFrame:
    """Expand transactions into per-street, per-hour occupancy counts."""
    if transactions.empty:
        return pd.DataFrame(columns=["street", "hour", "occupancy"])

    data = transactions.copy()
    data = data.rename(columns={"sourceStreetDisplayName": "street"})
    data["start_hour"] = data["startDtm"].dt.floor("h")
    data["end_hour"] = (data["endDtm"] - pd.Timedelta(seconds=1)).dt.floor("h")
    data = data[data["end_hour"] >= data["start_hour"]]

    durations = ((data["end_hour"] - data["start_hour"]) / pd.Timedelta(hours=1)).astype(int) + 1
    durations = durations.clip(lower=1)

    repeated_index = data.index.repeat(durations)
    expanded = data.loc[repeated_index, ["street", "start_hour"]].copy()

    offsets = list(chain.from_iterable(range(d) for d in durations.to_numpy()))
    expanded["hour"] = expanded["start_hour"].array + pd.to_timedelta(offsets, unit="h")
    expanded = expanded.drop(columns="start_hour")

    hourly = expanded.groupby(["street", "hour"], as_index=False).size().rename(columns={"size": "occupancy"})
    hourly["day_of_week"] = hourly["hour"].dt.dayofweek
    hourly["hour_of_day"] = hourly["hour"].dt.hour
    hourly["month"] = hourly["hour"].dt.month
    hourly["is_weekend"] = (hourly["day_of_week"] >= 5).astype(int)
    hourly["hour_sin"] = np.sin(2 * np.pi * hourly["hour_of_day"] / 24)
    hourly["hour_cos"] = np.cos(2 * np.pi * hourly["hour_of_day"] / 24)

    logger.info("Built hourly occupancy dataset with {} rows", len(hourly))
    return hourly.sort_values(["street", "hour"]).reset_index(drop=True)


def export_hourly(hourly: pd.DataFrame, output_path: Path) -> Path:
    """Persist hourly occupancy data to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hourly_copy = hourly.copy()
    hourly_copy["hour"] = hourly_copy["hour"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    hourly_copy.to_csv(output_path, index=False)
    logger.info("Saved hourly occupancy snapshot to {}", output_path)
    return output_path
