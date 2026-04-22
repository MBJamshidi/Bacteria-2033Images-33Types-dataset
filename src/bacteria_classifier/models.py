"""Model factory — transfer-learning backbones for bacteria classification."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import timm


# Curated backbones that achieve strong results on small medical datasets with
# limited compute.  All are available via timm >= 1.0.
SUPPORTED_ARCHITECTURES = [
    "resnet50",
    "resnet101",
    "efficientnet_b3",
    "efficientnet_b4",
    "convnext_small",
    "convnext_base",
    "vit_base_patch16_224",
    "swin_small_patch4_window7_224",
    "densenet121",
]


def build_model(
    architecture: str = "efficientnet_b3",
    num_classes: int = 33,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a pretrained classification model.

    Args:
        architecture: timm model name. Must be one of
            :data:`SUPPORTED_ARCHITECTURES`.
        num_classes: Number of output classes (33 for the bacteria dataset).
        pretrained: Load ImageNet-pretrained weights.
        dropout_rate: Dropout applied before the final linear head.
        freeze_backbone: Freeze all layers except the classification head.

    Returns:
        Configured :class:`~torch.nn.Module`.
    """
    model = timm.create_model(
        architecture,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout_rate,
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name and "head" not in name and "fc" not in name:
                param.requires_grad = False

    return model


def load_checkpoint(
    checkpoint_path: str,
    architecture: str = "efficientnet_b3",
    num_classes: int = 33,
    device: Optional[str] = None,
) -> nn.Module:
    """Load a model from a saved checkpoint.

    Args:
        checkpoint_path: Path to the ``.pt`` or ``.pth`` checkpoint file.
        architecture: Architecture used when the checkpoint was saved.
        num_classes: Number of output classes.
        device: Target device (``"cpu"``, ``"cuda"``, or ``None`` to auto-detect).

    Returns:
        Model with loaded weights set to eval mode.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(architecture, num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
