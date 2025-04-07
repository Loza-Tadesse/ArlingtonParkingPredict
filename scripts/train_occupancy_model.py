"""CLI entry point for training the parking occupancy model."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the parking occupancy model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration YAML (default: config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Import lazily so IDEs without adjusted PYTHONPATH still succeed
    pipeline = importlib.import_module("hotspot_predictor.pipelines.train_occupancy")
    pipeline.run(args.config)


if __name__ == "__main__":
    main()
