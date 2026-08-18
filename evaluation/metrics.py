"""Segmentation metrics built on a streaming confusion matrix.

Accumulating one confusion matrix over the whole validation set and deriving every metric
from it is both cheaper and more correct than averaging per-batch IoUs: a per-batch mean
over-weights batches where a rare class happens to appear, which is exactly the regime
this dataset lives in.
"""

from __future__ import annotations

import numpy as np
import torch

IGNORE_INDEX = 255


class ConfusionMatrix:
    """Streaming ``(C, C)`` confusion matrix, rows = ground truth, columns = prediction."""

    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    def reset(self) -> None:
        self.matrix[:] = 0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate a batch.

        Args:
            pred: ``(N, H, W)`` predicted class indices, or ``(N, C, H, W)`` logits.
            target: ``(N, H, W)`` ground-truth indices; ``255`` pixels are dropped.
        """
        if pred.ndim == 4:
            pred = pred.argmax(dim=1)
        pred = pred.detach().flatten().cpu().numpy()
        target = target.detach().flatten().cpu().numpy()

        keep = target != IGNORE_INDEX
        pred, target = pred[keep], target[keep]
        # Guard against a model emitting an out-of-range index; counting it would
        # silently corrupt another class's row.
        keep = (pred >= 0) & (pred < self.n_classes)
        pred, target = pred[keep], target[keep]

        index = target.astype(np.int64) * self.n_classes + pred.astype(np.int64)
        self.matrix += np.bincount(index, minlength=self.n_classes**2).reshape(
            self.n_classes, self.n_classes
        )

    def iou_per_class(self) -> np.ndarray:
        """Intersection-over-union per class; ``nan`` for classes absent from the target."""
        tp = np.diag(self.matrix).astype(np.float64)
        support = self.matrix.sum(axis=1)
        union = support + self.matrix.sum(axis=0) - tp
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, tp / union, np.nan)
        # A class with no ground-truth pixels is undefined, not zero: scoring it 0 would
        # drag mIoU down for a class the split never asked the model about.
        return np.where(support > 0, iou, np.nan)

    def miou(self) -> float:
        """Mean IoU over classes present in the ground truth."""
        return float(np.nanmean(self.iou_per_class()))

    def pixel_accuracy(self) -> float:
        """Fraction of labelled pixels classified correctly."""
        total = self.matrix.sum()
        return float(np.diag(self.matrix).sum() / total) if total else 0.0

    def mean_accuracy(self) -> float:
        """Mean per-class recall -- the imbalance-sensitive counterpart to pixel accuracy."""
        support = self.matrix.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            recall = np.where(support > 0, np.diag(self.matrix) / support, np.nan)
        return float(np.nanmean(recall))

    def frequency_weighted_iou(self) -> float:
        """IoU weighted by class frequency; dominated by the common classes."""
        support = self.matrix.sum(axis=1).astype(np.float64)
        iou = self.iou_per_class()
        valid = ~np.isnan(iou)
        if not valid.any() or support.sum() == 0:
            return 0.0
        return float((support[valid] * iou[valid]).sum() / support.sum())

    def dice_per_class(self) -> np.ndarray:
        """Dice / F1 per class."""
        tp = np.diag(self.matrix).astype(np.float64)
        support = self.matrix.sum(axis=1)
        denominator = support + self.matrix.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            dice = np.where(denominator > 0, 2 * tp / denominator, np.nan)
        return np.where(support > 0, dice, np.nan)

    def normalised(self) -> np.ndarray:
        """Row-normalised confusion matrix, for plotting which classes get confused."""
        support = self.matrix.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(support > 0, self.matrix / support, 0.0)

    def summary(self, class_names: list[str] | None = None) -> dict[str, float]:
        """All scalar metrics, plus ``iou/<class>`` entries."""
        iou = self.iou_per_class()
        names = class_names or [f"class{i}" for i in range(self.n_classes)]
        out: dict[str, float] = {
            "miou": self.miou(),
            "pixel_acc": self.pixel_accuracy(),
            "mean_acc": self.mean_accuracy(),
            "fw_iou": self.frequency_weighted_iou(),
        }
        for name, value in zip(names, iou):
            out[f"iou/{name}"] = float(value)
        return out


__all__ = ["ConfusionMatrix", "IGNORE_INDEX"]
