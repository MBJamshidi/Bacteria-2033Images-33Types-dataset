#!/usr/bin/env python3
"""Download the Bacteria-2033 dataset from Google Drive."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bacteria_classifier.utils import download_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Bacteria-2033 microscopy dataset from Google Drive."
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="data",
        help="Destination directory (default: data/)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Keep the ZIP archive without extracting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = download_dataset(
        dest_dir=args.dest,
        extract=not args.no_extract,
    )
    print(f"\nDataset ready at: {dataset_path.resolve()}")


if __name__ == "__main__":
    main()
