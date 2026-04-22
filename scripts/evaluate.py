#!/usr/bin/env python3
"""Evaluate a trained model on the test split.

Example::

    python scripts/evaluate.py \\
        --data-dir data/bacteria \\
        --checkpoint runs/exp1/best_model.pt \\
        --arch efficientnet_b3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bacteria_classifier.dataset import get_dataloaders
from bacteria_classifier.models import load_checkpoint
from bacteria_classifier.transforms import get_val_transforms
from bacteria_classifier.utils import (
    plot_confusion_matrix,
    print_classification_report,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a bacteria classifier checkpoint.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--arch", default="efficientnet_b3")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", default="runs/eval")
    return p.parse_args()


@torch.no_grad()
def run_inference(model, loader, device) -> tuple[list[int], list[int]]:
    model.eval()
    all_labels, all_preds = [], []
    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return all_labels, all_preds


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = get_dataloaders(
        root=args.data_dir,
        val_transform=get_val_transforms(args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    class_names = loaders["test"].dataset.dataset.classes

    model = load_checkpoint(args.checkpoint, architecture=args.arch, device=str(device))

    y_true, y_pred = run_inference(model, loaders["test"], device)

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = 100.0 * correct / len(y_true)
    print(f"\nTest Accuracy: {accuracy:.2f}% ({correct}/{len(y_true)})")

    print_classification_report(y_true, y_pred, class_names)
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        normalize=True,
        save_path=output_dir / "confusion_matrix.png",
    )
    print(f"Confusion matrix saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
