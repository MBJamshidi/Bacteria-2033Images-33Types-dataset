"""Unit tests for BacteriaDataset and get_dataloaders."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def fake_dataset(tmp_path_factory) -> Path:
    """Create a minimal fake dataset: 3 classes × 8 images each."""
    root = tmp_path_factory.mktemp("fake_bacteria")
    classes = ["Class_A", "Class_B", "Class_C"]
    rng = np.random.default_rng(0)

    for cls in classes:
        cls_dir = root / cls
        cls_dir.mkdir()
        for i in range(8):
            arr = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i:03d}.jpg")

    return root


# --------------------------------------------------------------------------- #
# BacteriaDataset tests
# --------------------------------------------------------------------------- #

class TestBacteriaDataset:
    def test_import(self):
        from bacteria_classifier.dataset import BacteriaDataset  # noqa: F401

    def test_length(self, fake_dataset):
        from bacteria_classifier.dataset import BacteriaDataset
        ds = BacteriaDataset(fake_dataset)
        assert len(ds) == 24  # 3 classes × 8 images

    def test_classes_detected(self, fake_dataset):
        from bacteria_classifier.dataset import BacteriaDataset
        ds = BacteriaDataset(fake_dataset)
        assert set(ds.classes) == {"Class_A", "Class_B", "Class_C"}

    def test_getitem_shape(self, fake_dataset):
        from bacteria_classifier.dataset import BacteriaDataset
        ds = BacteriaDataset(fake_dataset)
        img, label = ds[0]
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_label_range(self, fake_dataset):
        from bacteria_classifier.dataset import BacteriaDataset
        ds = BacteriaDataset(fake_dataset)
        labels = {lbl for _, lbl in ds.samples}
        assert labels == {0, 1, 2}

    def test_class_counts(self, fake_dataset):
        from bacteria_classifier.dataset import BacteriaDataset
        ds = BacteriaDataset(fake_dataset)
        counts = ds.class_counts()
        assert all(v == 8 for v in counts.values())

    def test_transform_applied(self, fake_dataset):
        """If a transform returns a dict[image], __getitem__ should return a tensor."""
        import torch
        from bacteria_classifier.dataset import BacteriaDataset
        from bacteria_classifier.transforms import get_val_transforms

        ds = BacteriaDataset(fake_dataset, transform=get_val_transforms(64))
        img, _ = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 64, 64)


# --------------------------------------------------------------------------- #
# get_dataloaders tests
# --------------------------------------------------------------------------- #

class TestGetDataloaders:
    def test_returns_three_splits(self, fake_dataset):
        from bacteria_classifier.dataset import get_dataloaders
        from bacteria_classifier.transforms import get_val_transforms

        loaders = get_dataloaders(
            root=fake_dataset,
            val_transform=get_val_transforms(64),
            batch_size=4,
            num_workers=0,
        )
        assert set(loaders.keys()) == {"train", "val", "test"}

    def test_no_overlap(self, fake_dataset):
        from bacteria_classifier.dataset import get_dataloaders
        from bacteria_classifier.transforms import get_val_transforms

        loaders = get_dataloaders(
            root=fake_dataset,
            val_transform=get_val_transforms(64),
            batch_size=4,
            num_workers=0,
        )
        train_idx = set(loaders["train"].dataset.indices)
        val_idx   = set(loaders["val"].dataset.indices)
        test_idx  = set(loaders["test"].dataset.indices)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)

    def test_total_coverage(self, fake_dataset):
        from bacteria_classifier.dataset import get_dataloaders
        from bacteria_classifier.transforms import get_val_transforms

        loaders = get_dataloaders(
            root=fake_dataset,
            val_transform=get_val_transforms(64),
            batch_size=4,
            num_workers=0,
        )
        total = (
            len(loaders["train"].dataset)
            + len(loaders["val"].dataset)
            + len(loaders["test"].dataset)
        )
        assert total == 24
