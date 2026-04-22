"""PyTorch Dataset and data-loading utilities for the bacteria microscopy dataset."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

# --------------------------------------------------------------------------- #
# Class registry
# --------------------------------------------------------------------------- #

BACTERIA_CLASSES: list[str] = [
    "Acinetobacter_baumannii",
    "Bacillus_anthracis",
    "Bacillus_cereus",
    "Bacteroides_fragilis",
    "Burkholderia_cepacia",
    "Campylobacter_jejuni",
    "Citrobacter_freundii",
    "Clostridium_difficile",
    "Clostridium_perfringens",
    "Corynebacterium_diphtheriae",
    "Enterobacter_cloacae",
    "Enterococcus_faecalis",
    "Enterococcus_faecium",
    "Escherichia_coli",
    "Fusobacterium_nucleatum",
    "Haemophilus_influenzae",
    "Helicobacter_pylori",
    "Klebsiella_pneumoniae",
    "Lactobacillus_acidophilus",
    "Listeria_monocytogenes",
    "Mycobacterium_tuberculosis",
    "Neisseria_gonorrhoeae",
    "Neisseria_meningitidis",
    "Proteus_mirabilis",
    "Pseudomonas_aeruginosa",
    "Salmonella_enterica",
    "Serratia_marcescens",
    "Shigella_dysenteriae",
    "Staphylococcus_aureus",
    "Streptococcus_pneumoniae",
    "Streptococcus_pyogenes",
    "Treponema_pallidum",
    "Vibrio_cholerae",
]

NUM_CLASSES = len(BACTERIA_CLASSES)
CLASS_TO_IDX: dict[str, int] = {cls: idx for idx, cls in enumerate(BACTERIA_CLASSES)}


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class BacteriaDataset(Dataset):
    """Bacteria microscopy image dataset (PyTorch).

    Expects ImageFolder layout::

        root/
        ├── Acinetobacter_baumannii/
        │   ├── img001.jpg
        │   └── ...
        ├── Bacillus_anthracis/
        └── ...

    Args:
        root: Path to the dataset root directory.
        transform: Optional albumentations or torchvision transform pipeline
            applied to each image.
        target_transform: Optional transform applied to each label.
        split: One of ``"train"``, ``"val"``, ``"test"``, or ``"all"``.
            Ignored when the dataset is loaded through
            :func:`get_dataloaders` (which performs the split externally).
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform

        self.samples: list[tuple[Path, int]] = []
        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}
        self._load_samples()

    # ---------------------------------------------------------------------- #

    def _load_samples(self) -> None:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        class_dirs = sorted(
            d for d in self.root.iterdir() if d.is_dir() and not d.name.startswith(".")
        )
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for class_dir in class_dirs:
            idx = self.class_to_idx[class_dir.name]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in extensions:
                    self.samples.append((img_path, idx))

    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple:
        img_path, label = self.samples[index]
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    # ---------------------------------------------------------------------- #

    def class_counts(self) -> dict[str, int]:
        """Return per-class sample counts."""
        counts: dict[str, int] = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            counts[self.classes[label]] += 1
        return counts


# --------------------------------------------------------------------------- #
# DataLoader factory
# --------------------------------------------------------------------------- #

def get_dataloaders(
    root: str | Path,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 4,
    random_state: int = 42,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    """Create stratified train / val / test :class:`~torch.utils.data.DataLoader` splits.

    Args:
        root: Dataset root containing one sub-folder per class.
        train_transform: Albumentations transform pipeline for training.
        val_transform: Albumentations transform pipeline for validation/test.
        test_size: Fraction of data reserved for testing.
        val_size: Fraction of *remaining* data reserved for validation.
        batch_size: Mini-batch size.
        num_workers: DataLoader worker processes.
        random_state: RNG seed for reproducibility.
        pin_memory: Pin tensors to CUDA-pinned memory for faster GPU transfer.

    Returns:
        Dictionary with keys ``"train"``, ``"val"``, ``"test"``.
    """
    full_ds = BacteriaDataset(root, transform=train_transform)
    labels = [lbl for _, lbl in full_ds.samples]
    indices = list(range(len(full_ds)))

    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=random_state
    )
    train_val_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (1.0 - test_size),
        stratify=train_val_labels,
        random_state=random_state,
    )

    val_ds = BacteriaDataset(root, transform=val_transform)
    test_ds = BacteriaDataset(root, transform=val_transform)

    loaders: dict[str, DataLoader] = {
        "train": DataLoader(
            Subset(full_ds, train_idx),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        ),
        "val": DataLoader(
            Subset(val_ds, val_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            Subset(test_ds, test_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    return loaders
