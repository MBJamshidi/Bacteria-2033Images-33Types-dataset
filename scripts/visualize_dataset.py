#!/usr/bin/env python3
"""Generate visualisation figures from the downloaded dataset.

Example::

    python scripts/visualize_dataset.py --data-dir data/bacteria --output-dir assets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bacteria_classifier.utils import plot_class_distribution, visualize_samples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate dataset visualisation figures.")
    p.add_argument("--data-dir", required=True, help="Dataset root directory")
    p.add_argument("--output-dir", default="assets", help="Where to save figures")
    p.add_argument("--n-per-class", type=int, default=4, help="Samples per class in grid")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Generating class distribution chart …")
    plot_class_distribution(
        root=args.data_dir,
        save_path=out / "class_distribution.png",
    )

    print("Generating sample image grid …")
    visualize_samples(
        root=args.data_dir,
        n_per_class=args.n_per_class,
        save_path=out / "sample_grid.png",
    )

    print(f"Figures saved to {out.resolve()}")


if __name__ == "__main__":
    main()
