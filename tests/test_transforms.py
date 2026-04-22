"""Unit tests for augmentation pipelines."""

from __future__ import annotations

import numpy as np
import pytest
import torch


class TestTransforms:
    def test_imports(self):
        from bacteria_classifier.transforms import (  # noqa: F401
            get_train_transforms,
            get_val_transforms,
            get_tta_transforms,
        )

    def test_val_transform_output_shape(self):
        from bacteria_classifier.transforms import get_val_transforms

        transform = get_val_transforms(image_size=224)
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        result = transform(image=img)["image"]

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)
        assert result.dtype == torch.float32

    def test_train_transform_output_shape(self):
        from bacteria_classifier.transforms import get_train_transforms

        transform = get_train_transforms(image_size=224)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = transform(image=img)["image"]

        assert result.shape == (3, 224, 224)

    def test_custom_image_size(self):
        from bacteria_classifier.transforms import get_val_transforms

        for size in [64, 128, 299, 384]:
            transform = get_val_transforms(image_size=size)
            img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            result = transform(image=img)["image"]
            assert result.shape == (3, size, size), f"Failed for size={size}"

    def test_tta_returns_list(self):
        from bacteria_classifier.transforms import get_tta_transforms

        tta = get_tta_transforms(image_size=224, n_augments=4)
        assert len(tta) == 4

    def test_tta_each_produces_tensor(self):
        from bacteria_classifier.transforms import get_tta_transforms

        tta = get_tta_transforms(image_size=224, n_augments=4)
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        for t in tta:
            out = t(image=img)["image"]
            assert isinstance(out, torch.Tensor)
            assert out.shape == (3, 224, 224)
