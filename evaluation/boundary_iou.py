"""Boundary IoU -- accuracy restricted to a band around object boundaries.

Standard mIoU is dominated by region interiors. A model can score well while producing
boundaries that are systematically thick, thin, or displaced, because the interior pixels
outnumber the boundary pixels by an order of magnitude. Boundary IoU (Cheng et al., 2021)
recomputes IoU using only pixels within ``d`` of a mask's boundary, which makes boundary
quality visible.

Reported alongside mIoU so the two can be read together: a large gap means the model gets
the *shape* roughly right but the *edges* wrong.
"""

from __future__ import annotations

import numpy as np

IGNORE_INDEX = 255


def boundary_region(mask: np.ndarray, dilation: int = 2) -> np.ndarray:
    """Pixels of a binary mask lying within ``dilation`` of its boundary.

    Computed as ``mask XOR erode(mask)``, which keeps the band strictly inside the mask
    so that two masks' bands are directly comparable.

    Args:
        mask: Boolean array.
        dilation: Band width in pixels.

    Returns:
        Boolean array of boundary-band pixels.
    """
    import cv2

    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    binary = mask.astype(np.uint8)
    # Pad so that regions touching the image border are not treated as having a boundary
    # there -- the image edge is not an object edge.
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(padded, kernel, iterations=dilation)[1:-1, 1:-1]
    return (binary - eroded).astype(bool)


def boundary_iou_per_class(
    pred: np.ndarray, target: np.ndarray, n_classes: int, dilation: int = 2
) -> np.ndarray:
    """Boundary IoU for each class.

    Args:
        pred: ``(H, W)`` predicted class indices.
        target: ``(H, W)`` ground-truth indices; ``255`` excluded.
        n_classes: Number of classes.
        dilation: Boundary band width.

    Returns:
        Length-``n_classes`` array; ``nan`` for classes absent from the target.
    """
    valid = target != IGNORE_INDEX
    out = np.full(n_classes, np.nan, dtype=np.float64)

    for cls in range(n_classes):
        target_mask = (target == cls) & valid
        if not target_mask.any():
            continue
        pred_mask = (pred == cls) & valid
        target_band = boundary_region(target_mask, dilation)
        pred_band = boundary_region(pred_mask, dilation)
        intersection = np.logical_and(target_band, pred_band).sum()
        union = np.logical_or(target_band, pred_band).sum()
        out[cls] = intersection / union if union else np.nan
    return out


class BoundaryIoU:
    """Accumulates boundary IoU across a dataset.

    Unlike region IoU this cannot be derived from a confusion matrix -- the boundary band
    depends on spatial structure -- so per-image values are averaged, skipping images
    where a class is absent.
    """

    def __init__(self, n_classes: int, dilation: int = 2):
        self.n_classes = n_classes
        self.dilation = dilation
        self._sum = np.zeros(n_classes, dtype=np.float64)
        self._count = np.zeros(n_classes, dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        """Accumulate one image, or a batch of shape ``(N, H, W)``."""
        if pred.ndim == 3:
            for p, t in zip(pred, target):
                self.update(p, t)
            return
        values = boundary_iou_per_class(pred, target, self.n_classes, self.dilation)
        present = ~np.isnan(values)
        self._sum[present] += values[present]
        self._count[present] += 1

    def per_class(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self._count > 0, self._sum / np.maximum(self._count, 1), np.nan)

    def mean(self) -> float:
        return float(np.nanmean(self.per_class()))

    def summary(self, class_names: list[str] | None = None) -> dict[str, float]:
        values = self.per_class()
        names = class_names or [f"class{i}" for i in range(self.n_classes)]
        out = {"boundary_miou": self.mean()}
        for name, value in zip(names, values):
            out[f"boundary_iou/{name}"] = float(value)
        return out


__all__ = ["BoundaryIoU", "boundary_iou_per_class", "boundary_region"]
