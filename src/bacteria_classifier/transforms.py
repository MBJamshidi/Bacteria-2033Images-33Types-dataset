"""Albumentations-based augmentation pipelines for bacteria microscopy images."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet statistics are a reasonable prior for Gram-stained RGB images used
# in the published transfer-learning benchmarks for this dataset.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Return the training augmentation pipeline.

    Includes spatial, colour, and microscopy-specific augmentations that
    improve generalisation without distorting biologically relevant morphology.
    """
    return A.Compose(
        [
            A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=1.0
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0
                    ),
                    A.CLAHE(clip_limit=4.0, p=1.0),
                ],
                p=0.5,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=0,
                p=0.3,
            ),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Return the deterministic validation / test transform pipeline."""
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ]
    )


def get_tta_transforms(image_size: int = 224, n_augments: int = 5) -> list[A.Compose]:
    """Return a list of *n_augments* test-time augmentation (TTA) pipelines."""
    tta = []
    for flip_h in [False, True]:
        for flip_v in [False, True]:
            if len(tta) >= n_augments:
                break
            ops = [A.Resize(height=image_size, width=image_size)]
            if flip_h:
                ops.append(A.HorizontalFlip(p=1.0))
            if flip_v:
                ops.append(A.VerticalFlip(p=1.0))
            ops += [A.Normalize(mean=_MEAN, std=_STD), ToTensorV2()]
            tta.append(A.Compose(ops))
    return tta
