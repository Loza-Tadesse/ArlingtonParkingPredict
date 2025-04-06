"""Lightweight lookup model for parking ticket risk scored by block/hour."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class ParkingRiskModel:
    """Compact model storing risk lookup tables derived from citations."""

    hourly: pd.DataFrame
    day: pd.DataFrame
    blocks: pd.DataFrame
    metadata: Dict[str, Any]

    @classmethod
    def build(cls, df: pd.DataFrame, config: Dict[str, Any]) -> "ParkingRiskModel":
        if df.empty:
            raise ValueError("Parking dataset is empty; cannot build model.")

        parking_cfg = config.get("app", {})
        dt_col = parking_cfg.get("violation_datetime_column", "ISSUE_DATETIME")
        lat_col = parking_cfg.get("latitude_column", "LATITUDE")
        lon_col = parking_cfg.get("longitude_column", "LONGITUDE")

        data = df.copy()
        data[dt_col] = pd.to_datetime(data[dt_col], errors="coerce")
        data = data.dropna(subset=[dt_col])

        date_min = data[dt_col].min().date()
        date_max = data[dt_col].max().date()
        total_days = max((date_max - date_min).days + 1, 1)

        hourly = (
            data.groupby(["block_normalized", "issue_hour"])
            .size()
            .rename("citations")
            .reset_index()
        )
        hourly["citations_per_day"] = hourly["citations"] / total_days
        max_rate = float(hourly["citations_per_day"].max()) if not hourly.empty else 0.0
        if max_rate > 0:
            hourly["likelihood"] = (hourly["citations_per_day"] / max_rate).clip(0.0, 1.0)
        else:
            hourly["likelihood"] = 0.0

        day = (
            data.groupby(["block_normalized", "issue_day_of_week"])
            .size()
            .rename("day_citations")
            .reset_index()
        )
        if not day.empty:
            day["day_ratio"] = day.groupby("block_normalized")["day_citations"].transform(
                lambda series: (series / series.mean()) if series.mean() else 1.0
            )
            day["day_ratio"] = day["day_ratio"].clip(0.3, 2.5)
        else:
            day["day_ratio"] = 1.0

        block_summary = (
            data.groupby("block_normalized")
            .agg(
                total_citations=("block_normalized", "size"),
                latitude=(lat_col, "mean"),
                longitude=(lon_col, "mean"),
            )
            .reset_index()
        )
        block_summary["display_name"] = block_summary["block_normalized"].str.title()
        peak = (
            hourly.groupby("block_normalized")["likelihood"].max().rename("peak_likelihood").reset_index()
        )
        block_summary = block_summary.merge(peak, on="block_normalized", how="left").fillna(
            {"peak_likelihood": 0.0}
        )
        block_summary = block_summary.sort_values("total_citations", ascending=False)

        global_hour = hourly.groupby("issue_hour")["citations_per_day"].mean()
        metadata: Dict[str, Any] = {
            "coverage_start": date_min.isoformat(),
            "coverage_end": date_max.isoformat(),
            "total_days": total_days,
            "max_hourly_rate": max_rate,
            "global_hour_rates": global_hour.to_dict(),
        }

        return cls(hourly=hourly, day=day, blocks=block_summary, metadata=metadata)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.hourly.to_csv(directory / "hourly.csv", index=False)
        self.day.to_csv(directory / "day.csv", index=False)
        self.blocks.to_csv(directory / "blocks.csv", index=False)
        with (directory / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(self.metadata, handle, indent=2)

    @classmethod
    def load(cls, directory: Path) -> "ParkingRiskModel":
        hourly_path = directory / "hourly.csv"
        day_path = directory / "day.csv"
        blocks_path = directory / "blocks.csv"
        meta_path = directory / "metadata.json"

        if not hourly_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Parking risk model not found in {directory}. Run the training pipeline first."
            )

        hourly = pd.read_csv(hourly_path)
        day = pd.read_csv(day_path) if day_path.exists() else pd.DataFrame()
        blocks = pd.read_csv(blocks_path) if blocks_path.exists() else pd.DataFrame()
        with meta_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        return cls(hourly=hourly, day=day, blocks=blocks, metadata=metadata)

    def predict(self, block_id: str, hour: int, day_of_week: int) -> Tuple[float, float, float]:
        block_hour = self.hourly[
            (self.hourly["block_normalized"] == block_id)
            & (self.hourly["issue_hour"] == hour)
        ]

        if not block_hour.empty:
            base_rate = float(block_hour.iloc[0]["citations_per_day"])
        else:
            global_hour = self.metadata.get("global_hour_rates", {})
            values = [float(value) for value in global_hour.values()]
            fallback = float(np.mean(values)) if values else 0.0
            base_rate = float(global_hour.get(str(hour), global_hour.get(hour, fallback)))

        block_day = self.day[
            (self.day["block_normalized"] == block_id)
            & (self.day["issue_day_of_week"] == day_of_week)
        ]
        if not block_day.empty:
            day_ratio = float(block_day.iloc[0]["day_ratio"])
        else:
            block_average = self.day[self.day["block_normalized"] == block_id]["day_ratio"].mean()
            day_ratio = float(block_average) if not math.isnan(block_average) else 1.0

        adjusted_rate = base_rate * day_ratio
        probability = 1.0 - math.exp(-adjusted_rate)
        probability = float(np.clip(probability, 0.0, 0.99))
        return probability, base_rate, day_ratio

    def get_metadata(self) -> Dict[str, Any]:
        return dict(self.metadata)

    def list_blocks(self) -> pd.DataFrame:
        return self.blocks.copy()
