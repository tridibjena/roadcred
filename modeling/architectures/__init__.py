"""Model factory for the segmentation architectures compared in this project."""

from __future__ import annotations

import torch.nn as nn

#: Architectures whose encoder stride can be reduced. Dilating the encoder to stride 8
#: doubles the resolution of the features the decoder sees, at roughly 2x the compute --
#: the direct test of whether thin classes are limited by the network or by the source
#: image. Only the atrous architectures support it; U-Net and FPN carry their own
#: full-resolution skip paths instead.
STRIDE_CONFIGURABLE = frozenset({"deeplabv3plus"})

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
    encoder_output_stride: int | None = None,
) -> nn.Module:
    """Build a segmentation model.

    Args:
        name: Architecture key from :data:`SMP_ARCHITECTURES`.
        encoder: Encoder/backbone name, e.g. ``resnet34``, ``resnet18``, ``mobilenet_v2``.
        n_classes: Number of output channels.
        encoder_weights: ``imagenet`` for pretrained, or ``None`` for random init.
            The architecture comparison holds this fixed so the encoder initialisation
            is not confounded with the decoder design.
        encoder_output_stride: Downsampling factor of the encoder's output features,
            ``8`` or ``16``. ``None`` keeps the architecture's own default (16 for
            DeepLabV3+), which is what every previously recorded run used -- so leaving
            it unset reproduces those runs exactly. Only meaningful for architectures in
            :data:`STRIDE_CONFIGURABLE`; passing it to any other raises rather than being
            silently dropped, since a silently ignored setting would invalidate an
            ablation without ever failing.

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
    kwargs: dict = {}
    if encoder_output_stride is not None:
        if key not in STRIDE_CONFIGURABLE:
            raise ValueError(
                f"encoder_output_stride is not supported by {name!r}; "
                f"supported: {sorted(STRIDE_CONFIGURABLE)}"
            )
        if encoder_output_stride not in (8, 16):
            raise ValueError(
                f"encoder_output_stride must be 8 or 16, got {encoder_output_stride}"
            )
        kwargs["encoder_output_stride"] = encoder_output_stride
    return constructor(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=n_classes,
        **kwargs,
    )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Total parameter count, for the size/accuracy trade-off table."""
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


__all__ = ["build_model", "count_parameters", "SMP_ARCHITECTURES", "STRIDE_CONFIGURABLE"]
