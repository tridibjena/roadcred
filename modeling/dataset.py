"""PyTorch dataset and augmentation pipeline for the prepared IDD variants.

Reads the ``data/processed/<variant>/`` layout written by :mod:`data.prepare_masks`.
Augmentation is deliberately conservative and geometry-preserving where it matters: no
vertical flip (a road scene is never upside down) and no rotation beyond a few degrees,
since a horizon that tilts 30 degrees is not a sample from the deployment distribution
and mostly teaches the model to hallucinate sky.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

IGNORE_INDEX = 255
#: ImageNet statistics -- both backbones use ImageNet-pretrained encoders by default.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(
    imgsz: tuple[int, int] = (224, 320), train: bool = True
) -> Callable:
    """Build the albumentations pipeline.

    Args:
        imgsz: Target ``(height, width)``. Both must be multiples of 32 so the encoder's
            successive downsamplings stay exact.
        train: Whether to include stochastic augmentation.

    Returns:
        An albumentations ``Compose`` producing ``image`` (float CHW tensor) and
        ``mask`` (long HW tensor).
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    height, width = imgsz
    if train:
        stages = [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.85, 1.20),
                translate_percent=(-0.05, 0.05),
                rotate=(-7, 7),
                # Fill masks with ignore, never with class 0: inventing "drivable" in the
                # corners created by a warp would be a silent label error.
                fill_mask=IGNORE_INDEX,
                p=0.6,
            ),
            A.RandomBrightnessContrast(0.25, 0.25, p=0.5),
            A.HueSaturationValue(10, 20, 10, p=0.3),
            # Weather/blur are held out of training: they are the *test-time* corruptions
            # used in the robustness evaluation, so training on them would invalidate it.
            A.Resize(height, width, interpolation=1, mask_interpolation=0),
        ]
    else:
        stages = [A.Resize(height, width, interpolation=1, mask_interpolation=0)]

    stages += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(stages)


class IDDSegmentation(Dataset):
    """Image/mask pairs from a prepared variant directory.

    Args:
        root: Variant directory, e.g. ``data/processed/level1_official``.
        split: ``train`` or ``val``.
        imgsz: Target ``(height, width)``.
        train: Whether to apply training augmentation. Defaults to ``split == "train"``.
        transform: Override the built-in pipeline entirely.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        imgsz: tuple[int, int] = (224, 320),
        train: bool | None = None,
        transform: Callable | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / "images" / split
        self.mask_dir = self.root / "masks" / split
        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"{self.image_dir} not found. Run: python -m data.prepare_masks"
            )
        self.images = sorted(self.image_dir.glob("*.jpg"))
        if not self.images:
            raise FileNotFoundError(f"No images in {self.image_dir}")
        is_train = train if train is not None else (split == "train")
        self.transform = transform or build_transforms(imgsz, is_train)

    def __len__(self) -> int:
        return len(self.images)

    @property
    def drives(self) -> list[str]:
        """Drive-sequence id per sample, recoverable from the ``<drive>_<frame>`` key."""
        return [p.stem.rsplit("_", 1)[0] for p in self.images]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        import cv2

        image_path = self.images[index]
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_dir / f"{image_path.stem}.png"), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Missing mask for {image_path.name}")
        if mask.ndim == 3:
            mask = mask[..., 0]
        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].long()


def build_loaders(
    root: str | Path,
    imgsz: tuple[int, int] = (224, 320),
    batch_size: int = 16,
    workers: int = 0,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation loaders for a prepared variant.

    The validation loader is unshuffled and undropped so every validation frame is scored
    exactly once, which keeps mIoU comparable across runs.
    """
    generator = torch.Generator().manual_seed(seed)
    # pin_memory is a CUDA-only optimisation; MPS warns and ignores it.
    pin = torch.cuda.is_available()
    train_set = IDDSegmentation(root, "train", imgsz, train=True)
    val_set = IDDSegmentation(root, "val", imgsz, train=False)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        drop_last=True,
        generator=generator,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        drop_last=False,
        persistent_workers=workers > 0,
    )
    return train_loader, val_loader


def training_class_counts(root: str | Path, n_classes: int) -> np.ndarray:
    """Count pixels per class over the **training** split only.

    Used to derive weighted-cross-entropy weights. Reading the raw mask files directly
    (rather than the augmented tensors) keeps the counts deterministic.
    """
    import cv2

    counts = np.zeros(n_classes, dtype=np.int64)
    for path in sorted((Path(root) / "masks" / "train").glob("*.png")):
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        valid = mask[mask != IGNORE_INDEX]
        counts += np.bincount(valid.ravel(), minlength=n_classes)[:n_classes]
    return counts


__all__ = ["IDDSegmentation", "build_loaders", "build_transforms", "training_class_counts"]
