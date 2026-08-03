"""Download official Cricsheet archives without committing large data files."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from urllib.request import urlopen

from src.config import RAW_DIR

DATASETS = {
    "ipl": "https://cricsheet.org/downloads/ipl_json.zip",
    "t20i": "https://cricsheet.org/downloads/t20s_json.zip",
}


def download(dataset: str, overwrite: bool = False) -> Path:
    """Download a named official Cricsheet dataset and return its local path."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}. Choose one of {sorted(DATASETS)}")

    destination = RAW_DIR / f"{dataset}_json.zip"
    if destination.exists() and not overwrite:
        return destination

    with urlopen(DATASETS[dataset], timeout=60) as response, destination.open("wb") as file:
        shutil.copyfileobj(response, file)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Cricsheet match archives.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(download(args.dataset, args.overwrite))


if __name__ == "__main__":
    main()

