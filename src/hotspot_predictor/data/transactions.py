"""Download parking transaction data from Arlington open data API."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable, List

import requests
from loguru import logger

BASE_URL = "https://datahub-v2.arlingtonva.us/api/ParkingMeter/ParkingTransactions"
PAGE_SIZE = 100_000


def fetch_range(start_iso: str, end_iso: str, max_retries: int = 5, timeout: int = 180) -> Iterable[List[Dict]]:
    """Yield batches of transactions between the provided timestamps."""
    session = requests.Session()
    skip = 0
    total = 0

    while True:
        params = {
            "$top": PAGE_SIZE,
            "$skip": skip,
            "$orderby": "startDtm",
            "$filter": f"startDtm ge {start_iso} and startDtm lt {end_iso}",
        }

        for attempt in range(max_retries):
            response = session.get(BASE_URL, params=params, timeout=timeout)
            if response.status_code >= 500:
                wait = 2 * (attempt + 1)
                logger.warning(
                    "Server error {} — retrying in {} seconds",
                    response.status_code,
                    wait,
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            break
        else:
            response.raise_for_status()

        batch: List[Dict] = response.json()
        total += len(batch)
        logger.info("Fetched batch skip={} size={} total={}", skip, len(batch), total)

        if not batch:
            break

        yield batch

        if len(batch) < PAGE_SIZE:
            break

        skip += PAGE_SIZE

    logger.info("Finished download {} -> {} | total rows={}", start_iso, end_iso, total)


def download_month(year: int, month: int, output_dir: Path, *, force: bool = False) -> Path:
    """Download a calendar month into the raw data directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"parking_transactions_{year:04d}-{month:02d}.csv"
    if output_path.exists() and not force:
        logger.info("Skipping existing raw file {}", output_path)
        return output_path

    if month == 12:
        end_year = year + 1
        end_month = 1
    else:
        end_year = year
        end_month = month + 1

    start_iso = f"{year:04d}-{month:02d}-01T00:00:00Z"
    end_iso = f"{end_year:04d}-{end_month:02d}-01T00:00:00Z"

    rows: List[Dict] = []
    for batch in fetch_range(start_iso, end_iso):
        rows.extend(batch)

    import csv

    if not rows:
        logger.warning("No rows retrieved for {}-{}", year, month)
        output_path.touch()
        return output_path

    fieldnames = sorted(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote {} rows to {}", len(rows), output_path)
    return output_path


def download_months(months: Iterable[tuple[int, int]], output_dir: Path, *, force: bool = False) -> List[Path]:
    """Download multiple months and return written paths."""
    paths: List[Path] = []
    for year, month in months:
        path = download_month(year, month, output_dir, force=force)
        paths.append(path)
    return paths
