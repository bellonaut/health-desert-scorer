"""Download or prompt for open data inputs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import requests


CONFIG: Dict[str, Dict[str, Optional[str]]] = {
    "lga_boundaries": {
        "filename": "lga_boundaries.geojson",
        "url": None,
    },
    "health_facilities": {
        "filename": "health_facilities.geojson",
        "url": None,
    },
    "population": {
        "filename": "population_lga.csv",
        "url": None,
    },
    "opencellid": {
        "filename": "opencellid.csv.gz",
        "url": None,
    },
}

REQUIRED_KEYS = {"lga_boundaries", "health_facilities"}


def configure_logging() -> None:
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename="logs/download_open_data.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)


def main() -> None:
    configure_logging()
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    missing_required = []

    for key, meta in CONFIG.items():
        filename = meta["filename"]
        url = meta["url"]
        dest = data_dir / filename
        if dest.exists():
            logging.info("Found %s at %s", key, dest)
            continue

        if url:
            logging.info("Downloading %s from %s", key, url)
            try:
                _download_file(url, dest)
                logging.info("Downloaded %s to %s", key, dest)
            except requests.RequestException as exc:
                logging.error("Failed to download %s: %s", key, exc)
        else:
            print(f"MANUAL DOWNLOAD REQUIRED: {key}")
            print(f"Expected filename: {filename}")
            print(f"Place file at: {dest}")

        if not dest.exists() and key in REQUIRED_KEYS:
            missing_required.append(key)

    if missing_required:
        logging.error("Missing required datasets: %s", ", ".join(missing_required))
        sys.exit(2)

    print("OK: Required datasets present.")
    sys.exit(0)


if __name__ == "__main__":
    main()
