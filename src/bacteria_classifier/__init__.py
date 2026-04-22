"""Bacteria Classifier — ML utilities for the Bacteria-2033Images-33Types dataset."""

__version__ = "1.0.0"
__author__ = "Mohammad Behdad Jamshidi"

from .dataset import BacteriaDataset, get_dataloaders
from .transforms import get_train_transforms, get_val_transforms
from .utils import download_dataset, visualize_samples, plot_class_distribution

__all__ = [
    "BacteriaDataset",
    "get_dataloaders",
    "get_train_transforms",
    "get_val_transforms",
    "download_dataset",
    "visualize_samples",
    "plot_class_distribution",
]
