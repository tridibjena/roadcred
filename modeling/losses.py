"""Segmentation losses, written from scratch in PyTorch.

All losses share one interface -- ``forward(logits, target)`` where ``logits`` is
``(N, C, H, W)`` and ``target`` is ``(N, H, W)`` of class indices with
:data:`IGNORE_INDEX` marking pixels excluded from the loss -- so the ablation can swap
one for another with no other change to the training recipe.

The ablation exists because IDD Lite is genuinely imbalanced: ``living-thing`` is 1.31%
of pixels and ``non-drivable`` 2.19%, against ``construction-vegetation`` at 26%. Plain
cross-entropy optimises a pixel-weighted objective and is therefore free to ignore the
rare classes almost entirely; the region and boundary losses here are the standard
answers to that, and the ablation measures which one actually pays off.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = 255


def compute_class_weights(
    counts: np.ndarray, scheme: str = "inv_sqrt", normalise: bool = True
) -> torch.Tensor:
    """Derive per-class loss weights from training-split pixel counts.

    Args:
        counts: Pixel count per class, from the **training** split only. Deriving weights
            from validation would leak label statistics into model selection.
        scheme: ``inv_sqrt`` (weight ∝ 1/sqrt(freq); the usual middle ground),
            ``inv`` (weight ∝ 1/freq; aggressive, can destabilise training), or
            ``enet`` (the ENet formulation, ``1 / ln(c + freq)``).
        normalise: Scale weights to mean 1.0 so the loss magnitude stays comparable
            across schemes and the learning rate does not need retuning per scheme.

    Returns:
        ``float32`` tensor of shape ``(C,)``.
    """
    counts = np.asarray(counts, dtype=np.float64)
    freq = counts / max(counts.sum(), 1.0)
    freq = np.clip(freq, 1e-8, None)

    if scheme == "inv":
        weights = 1.0 / freq
    elif scheme == "inv_sqrt":
        weights = 1.0 / np.sqrt(freq)
    elif scheme == "enet":
        weights = 1.0 / np.log(1.02 + freq)
    else:
        raise ValueError(f"Unknown weighting scheme {scheme!r}")

    # Classes absent from training get zero weight rather than a huge one -- otherwise a
    # class with no pixels would dominate the gradient purely through its 1/freq clamp.
    weights[counts == 0] = 0.0
    if normalise and weights.sum() > 0:
        weights = weights * (len(weights) / weights.sum())
    return torch.tensor(weights, dtype=torch.float32)


def _valid_mask(target: torch.Tensor) -> torch.Tensor:
    """Boolean mask of pixels that participate in the loss."""
    return target != IGNORE_INDEX


def _one_hot(target: torch.Tensor, n_classes: int) -> torch.Tensor:
    """One-hot encode ``(N, H, W)`` indices to ``(N, C, H, W)``, zeroing ignored pixels."""
    safe = torch.where(_valid_mask(target), target, torch.zeros_like(target))
    onehot = F.one_hot(safe.long(), n_classes).permute(0, 3, 1, 2).float()
    return onehot * _valid_mask(target).unsqueeze(1).float()


class CrossEntropy(nn.Module):
    """Standard pixel-wise cross-entropy, optionally class-weighted.

    Args:
        weight: Per-class weights, or ``None`` for unweighted. Passing weights from
            :func:`compute_class_weights` gives the ``weighted_ce`` ablation arm.
        label_smoothing: Softens targets; kept at 0 by default so the ablation isolates
            the loss family rather than confounding it with regularisation.
    """

    def __init__(self, weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            target.long(),
            weight=self.weight,
            ignore_index=IGNORE_INDEX,
            label_smoothing=self.label_smoothing,
        )


class TverskyLoss(nn.Module):
    r"""Soft Tversky loss, which subsumes Dice.

    Per class, with :math:`p` the softmax probability and :math:`g` the one-hot target:

    .. math::
        T = \frac{\sum pg + \epsilon}
                 {\sum pg + \alpha \sum p(1-g) + \beta \sum (1-p)g + \epsilon}

    ``alpha`` penalises false positives and ``beta`` false negatives.
    ``alpha = beta = 0.5`` recovers the Dice coefficient exactly; raising ``beta`` above
    ``alpha`` trades precision for recall, which is the point for rare classes whose
    false negatives otherwise cost almost nothing.

    Args:
        alpha: False-positive weight.
        beta: False-negative weight.
        eps: Numerical floor, also preventing a 0/0 for classes absent from a batch.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.alpha, self.beta, self.eps = alpha, beta, eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = logits.shape[1]
        probs = logits.softmax(dim=1) * _valid_mask(target).unsqueeze(1).float()
        onehot = _one_hot(target, n_classes)

        dims = (0, 2, 3)  # sum over batch and space, keep per-class
        tp = (probs * onehot).sum(dims)
        fp = (probs * (1 - onehot)).sum(dims)
        fn = ((1 - probs) * onehot).sum(dims)
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)

        # Average only over classes present in this batch, so absent classes do not
        # contribute a constant 1.0 that dilutes the gradient signal.
        present = onehot.sum(dims) > 0
        if present.any():
            return 1.0 - tversky[present].mean()
        return logits.sum() * 0.0


class DiceLoss(TverskyLoss):
    """Soft Dice loss -- :class:`TverskyLoss` with ``alpha = beta = 0.5``."""

    def __init__(self, eps: float = 1e-6):
        super().__init__(alpha=0.5, beta=0.5, eps=eps)


class BoundaryWeightedCE(nn.Module):
    """Cross-entropy up-weighted in a band around ground-truth class boundaries.

    Semantic segmentation errors concentrate at object boundaries, but boundary pixels
    are a small fraction of the image and so contribute little to a pixel-mean loss. This
    finds boundary pixels by morphological gradient -- a pixel is on a boundary when
    max-pooling and min-pooling the label map disagree -- and scales their contribution.

    .. note::
       This is the *boundary-weighted cross-entropy* formulation, not Kervadec et al.'s
       distance-map boundary loss. The distance-map version needs a Euclidean distance
       transform per class per sample per step, which at this batch size and resolution
       dominates step time on MPS for no measured benefit. The name in ``RESULTS.md``
       reflects what is actually computed.

    Args:
        weight: Optional per-class weights, applied on top of the boundary weighting.
        boundary_weight: Multiplier applied to boundary-band pixels.
        kernel_size: Width of the band, in pixels.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        boundary_weight: float = 5.0,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.boundary_weight = boundary_weight
        self.kernel_size = kernel_size

    def _boundary_band(self, target: torch.Tensor) -> torch.Tensor:
        """Boolean map of pixels lying on a class boundary."""
        valid = _valid_mask(target)
        # Map ignore to a sentinel that cannot equal a real class, so image borders and
        # void regions do not register as boundaries of a real class.
        labels = torch.where(valid, target, torch.full_like(target, 254)).float().unsqueeze(1)
        pad = self.kernel_size // 2
        dilated = F.max_pool2d(labels, self.kernel_size, stride=1, padding=pad)
        eroded = -F.max_pool2d(-labels, self.kernel_size, stride=1, padding=pad)
        return ((dilated != eroded).squeeze(1)) & valid

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_pixel = F.cross_entropy(
            logits,
            target.long(),
            weight=self.weight,
            ignore_index=IGNORE_INDEX,
            reduction="none",
        )
        valid = _valid_mask(target)
        pixel_weight = torch.ones_like(per_pixel)
        pixel_weight[self._boundary_band(target)] = self.boundary_weight
        pixel_weight = pixel_weight * valid.float()

        denominator = pixel_weight.sum().clamp_min(1.0)
        return (per_pixel * pixel_weight).sum() / denominator


class CombinedLoss(nn.Module):
    """Convex blend of two losses: ``alpha * primary + (1 - alpha) * secondary``.

    Region losses (Dice/Tversky) are unstable on their own early in training because
    their gradient vanishes when predictions are near-uniform; pairing them with
    cross-entropy is standard practice and is what the ablation actually compares.
    """

    def __init__(self, primary: nn.Module, secondary: nn.Module, alpha: float = 0.5):
        super().__init__()
        self.primary, self.secondary, self.alpha = primary, secondary, alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.primary(logits, target) + (1 - self.alpha) * self.secondary(
            logits, target
        )


def make_loss(
    name: str,
    class_weights: torch.Tensor | None = None,
    alpha: float = 0.5,
    tversky_beta: float = 0.7,
    boundary_weight: float = 5.0,
) -> nn.Module:
    """Build a loss by name -- the single entry point used by the ablation driver.

    Args:
        name: ``ce``, ``weighted_ce``, ``dice``, ``tversky``, or ``boundary``.
        class_weights: Required for ``weighted_ce``; ignored by ``ce``.
        alpha: Blend weight for the region losses against cross-entropy.
        tversky_beta: False-negative weight for ``tversky``. ``alpha`` for the Tversky
            term is set to ``1 - tversky_beta`` so the two sum to 1.
        boundary_weight: Boundary-band multiplier for ``boundary``.

    Returns:
        A ready-to-use loss module.
    """
    name = name.lower()
    if name == "ce":
        return CrossEntropy()
    if name == "weighted_ce":
        if class_weights is None:
            raise ValueError("weighted_ce requires class_weights from the training split")
        return CrossEntropy(weight=class_weights)
    if name == "dice":
        return CombinedLoss(DiceLoss(), CrossEntropy(), alpha=alpha)
    if name == "tversky":
        return CombinedLoss(
            TverskyLoss(alpha=1.0 - tversky_beta, beta=tversky_beta), CrossEntropy(), alpha=alpha
        )
    if name == "boundary":
        return BoundaryWeightedCE(boundary_weight=boundary_weight)
    raise ValueError(
        f"Unknown loss {name!r}; expected ce/weighted_ce/dice/tversky/boundary"
    )


__all__ = [
    "IGNORE_INDEX",
    "CrossEntropy",
    "DiceLoss",
    "TverskyLoss",
    "BoundaryWeightedCE",
    "CombinedLoss",
    "make_loss",
    "compute_class_weights",
]
