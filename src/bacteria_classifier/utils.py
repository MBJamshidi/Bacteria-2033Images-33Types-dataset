"""Utility functions: download, visualisation, and metrics helpers."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

GDRIVE_FILE_ID = "1aR7Dz11wKV3t7awnnnO32UE_37MYF6wX"


def download_dataset(
    dest_dir: str | Path = "data",
    file_id: str = GDRIVE_FILE_ID,
    extract: bool = True,
) -> Path:
    """Download the dataset ZIP from Google Drive using *gdown*.

    Args:
        dest_dir: Directory where the ZIP is saved (and extracted).
        file_id: Google Drive file ID.
        extract: Unzip the archive after downloading.

    Returns:
        Path to the extracted dataset root.
    """
    try:
        import gdown
    except ImportError as exc:
        raise ImportError("Install gdown: pip install gdown") from exc

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dest_dir / "bacteria_dataset.zip"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(zip_path), quiet=False)

    if extract:
        print(f"Extracting {zip_path} …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        zip_path.unlink()
        print("Done.")

    dataset_dirs = [d for d in dest_dir.iterdir() if d.is_dir()]
    return dataset_dirs[0] if dataset_dirs else dest_dir


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #

def visualize_samples(
    root: str | Path,
    classes: Optional[list[str]] = None,
    n_per_class: int = 4,
    figsize: tuple[int, int] = (18, 12),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Display a grid of sample images from each class.

    Args:
        root: Dataset root (one sub-folder per class).
        classes: Subset of class names to display. Defaults to all.
        n_per_class: Images to show per class row.
        figsize: Matplotlib figure size.
        save_path: If given, save the figure here instead of showing it.

    Returns:
        The :class:`~matplotlib.figure.Figure` object.
    """
    root = Path(root)
    all_classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    display_classes = classes if classes is not None else all_classes

    fig, axes = plt.subplots(
        len(display_classes), n_per_class, figsize=figsize
    )
    if len(display_classes) == 1:
        axes = axes[np.newaxis, :]

    for row, cls in enumerate(display_classes):
        cls_dir = root / cls
        images = sorted(cls_dir.iterdir())[: n_per_class]
        for col, img_path in enumerate(images):
            img = Image.open(img_path).convert("RGB")
            ax = axes[row, col]
            ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(cls.replace("_", " "), fontsize=7, rotation=0, labelpad=80, va="center")

    plt.suptitle("Bacteria Microscopy Dataset — Sample Images", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_class_distribution(
    root: str | Path,
    figsize: tuple[int, int] = (14, 7),
    palette: str = "viridis",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Horizontal bar chart of per-class image counts.

    Args:
        root: Dataset root containing class sub-folders.
        figsize: Matplotlib figure size.
        palette: Seaborn colour palette name.
        save_path: If given, save the figure here.

    Returns:
        The :class:`~matplotlib.figure.Figure` object.
    """
    root = Path(root)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    counts: dict[str, int] = {}
    for cls_dir in sorted(root.iterdir()):
        if cls_dir.is_dir():
            counts[cls_dir.name.replace("_", " ")] = sum(
                1 for f in cls_dir.iterdir() if f.suffix.lower() in exts
            )

    fig, ax = plt.subplots(figsize=figsize)
    classes = list(counts.keys())
    values = list(counts.values())
    colors = sns.color_palette(palette, n_colors=len(classes))

    bars = ax.barh(classes, values, color=colors, edgecolor="white", height=0.7)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)
    ax.set_xlabel("Number of Images", fontsize=11)
    ax.set_title("Class Distribution — Bacteria 2033 Dataset", fontsize=13, fontweight="bold")
    ax.axvline(np.mean(values), color="crimson", linestyle="--", linewidth=1.2, label=f"Mean ({np.mean(values):.0f})")
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    sns.despine(ax=ax, left=True)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    normalize: bool = True,
    figsize: tuple[int, int] = (20, 18),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot a labelled confusion matrix.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Ordered list of class name strings.
        normalize: If True, display row-normalised percentages.
        figsize: Matplotlib figure size.
        save_path: If given, save the figure here.

    Returns:
        The :class:`~matplotlib.figure.Figure` object.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt, vmax = ".2f", 1.0
    else:
        cm_display = cm
        fmt, vmax = "d", cm.max()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        xticklabels=[c.replace("_", " ") for c in class_names],
        yticklabels=[c.replace("_", " ") for c in class_names],
        ax=ax,
        linewidths=0.3,
        linecolor="lightgray",
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    title = "Confusion Matrix" + (" (Normalised)" if normalize else "")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def print_classification_report(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> str:
    """Return a formatted sklearn classification report."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=[c.replace("_", " ") for c in class_names],
        digits=4,
    )
    print(report)
    return report


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot loss and accuracy training curves side by side.

    Args:
        train_losses: Per-epoch training losses.
        val_losses: Per-epoch validation losses.
        train_accs: Per-epoch training accuracies.
        val_accs: Per-epoch validation accuracies.
        save_path: If given, save the figure here.

    Returns:
        The :class:`~matplotlib.figure.Figure` object.
    """
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, "b-o", ms=4, label="Train")
    ax1.plot(epochs, val_losses, "r-o", ms=4, label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves", fontweight="bold")
    ax1.legend()
    sns.despine(ax=ax1)

    ax2.plot(epochs, train_accs, "b-o", ms=4, label="Train")
    ax2.plot(epochs, val_accs, "r-o", ms=4, label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curves", fontweight="bold")
    ax2.legend()
    sns.despine(ax=ax2)

    plt.suptitle("Training Progress", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
