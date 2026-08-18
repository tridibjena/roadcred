"""Model factory for the segmentation architectures compared in this project."""

from __future__ import annotations

import torch.nn as nn

#: Architectures available to the comparison, mapped to their smp constructor name.
SMP_ARCHITECTURES = {
    "deeplabv3plus": "DeepLabV3Plus",
    "unet": "Unet",
    "fpn": "FPN",
    "segformer": "Segformer",
}


def build_model(
    name: str = "deeplabv3plus",
    encoder: str = "resnet34",
    n_classes: int = 7,
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    """Build a segmentation model.

    Args:
        name: Architecture key from :data:`SMP_ARCHITECTURES`.
        encoder: Encoder/backbone name, e.g. ``resnet34``, ``resnet18``, ``mobilenet_v2``.
        n_classes: Number of output channels.
        encoder_weights: ``imagenet`` for pretrained, or ``None`` for random init.
            The architecture comparison holds this fixed so the encoder initialisation
            is not confounded with the decoder design.

    Returns:
        A model mapping ``(N, 3, H, W)`` to ``(N, n_classes, H, W)`` logits.
    """
    import segmentation_models_pytorch as smp

    key = name.lower()
    if key not in SMP_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {name!r}; expected one of {sorted(SMP_ARCHITECTURES)}"
        )
    constructor = getattr(smp, SMP_ARCHITECTURES[key])
    return constructor(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=n_classes,
    )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Total parameter count, for the size/accuracy trade-off table."""
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


__all__ = ["build_model", "count_parameters", "SMP_ARCHITECTURES"]
