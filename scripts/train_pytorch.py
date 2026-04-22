#!/usr/bin/env python3
"""PyTorch training script for bacteria classification.

Example usage::

    python scripts/train_pytorch.py \\
        --data-dir data/bacteria \\
        --arch efficientnet_b3 \\
        --epochs 50 \\
        --batch-size 32 \\
        --lr 3e-4 \\
        --output-dir runs/exp1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bacteria_classifier.dataset import get_dataloaders
from bacteria_classifier.models import build_model, count_parameters
from bacteria_classifier.transforms import get_train_transforms, get_val_transforms
from bacteria_classifier.utils import (
    plot_confusion_matrix,
    plot_training_curves,
    print_classification_report,
)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a bacteria classifier with PyTorch.")
    p.add_argument("--data-dir", required=True, help="Dataset root (ImageFolder layout)")
    p.add_argument("--arch", default="efficientnet_b3", help="Model architecture (timm name)")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--output-dir", default="runs/exp1")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    scaler,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="Eval ", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, 100.0 * correct / total, all_labels, all_preds


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    loaders = get_dataloaders(
        root=args.data_dir,
        train_transform=get_train_transforms(args.image_size),
        val_transform=get_val_transforms(args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    num_classes = len(loaders["train"].dataset.dataset.classes)
    class_names = loaders["train"].dataset.dataset.classes
    print(f"Classes: {num_classes}  |  Train: {len(loaders['train'].dataset)}  "
          f"Val: {len(loaders['val'].dataset)}  Test: {len(loaders['test'].dataset)}")

    # Model
    model = build_model(
        architecture=args.arch,
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        dropout_rate=args.dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    params = count_parameters(model)
    print(f"Parameters — total: {params['total']:,}  trainable: {params['trainable']:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
    }
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, loaders["train"], optimizer, criterion, device, scaler)
        val_loss, val_acc, _, _ = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss {train_loss:.4f}/{val_loss:.4f} | "
            f"Acc {train_acc:.2f}%/{val_acc:.2f}% | "
            f"{elapsed:.0f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_acc": val_acc, "args": vars(args)},
                output_dir / "best_model.pt",
            )

    # Final test evaluation
    model_best = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(model_best["model_state_dict"])
    test_loss, test_acc, y_true, y_pred = evaluate(model, loaders["test"], criterion, device)
    print(f"\nTest accuracy: {test_acc:.2f}%  |  Test loss: {test_loss:.4f}")
    print_classification_report(y_true, y_pred, class_names)

    # Save artefacts
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(
        history["train_loss"], history["val_loss"],
        history["train_acc"], history["val_acc"],
        save_path=output_dir / "training_curves.png",
    )
    plot_confusion_matrix(
        y_true, y_pred, class_names, normalize=True,
        save_path=output_dir / "confusion_matrix.png",
    )
    print(f"\nAll artefacts saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
