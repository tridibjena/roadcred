"""Metric tests, checked against values computed by hand.

mIoU has several defensible definitions that differ on edge cases (absent classes, ignore
pixels). These pin down the one this project reports, so the numbers in RESULTS.md mean a
specific thing.
"""

from __future__ import annotations

import numpy as np
import torch

from evaluation.metrics import ConfusionMatrix


def cm_from(pred: list[int], target: list[int], n_classes: int) -> ConfusionMatrix:
    matrix = ConfusionMatrix(n_classes)
    matrix.update(torch.tensor([[pred]]), torch.tensor([[target]]))
    return matrix


def test_perfect_prediction_scores_one():
    matrix = cm_from([0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2], 3)
    assert matrix.miou() == 1.0
    assert matrix.pixel_accuracy() == 1.0
    assert matrix.mean_accuracy() == 1.0


def test_iou_matches_hand_computation():
    # target [0,0,1,1], pred [0,1,1,1]
    #   class0: tp=1 fp=0 fn=1 -> 1/2      class1: tp=2 fp=1 fn=0 -> 2/3
    matrix = cm_from([0, 1, 1, 1], [0, 0, 1, 1], 2)
    np.testing.assert_allclose(matrix.iou_per_class(), [0.5, 2 / 3], rtol=1e-6)
    assert abs(matrix.miou() - (0.5 + 2 / 3) / 2) < 1e-9


def test_ignore_pixels_are_dropped_entirely():
    matrix = cm_from([0, 1], [0, 255], 2)
    assert matrix.matrix.sum() == 1
    assert matrix.matrix[0, 0] == 1


def test_absent_class_is_nan_not_zero():
    """A class the split never asked about must not drag mIoU down."""
    matrix = cm_from([0, 0], [0, 0], 3)
    iou = matrix.iou_per_class()
    assert iou[0] == 1.0
    assert np.isnan(iou[1]) and np.isnan(iou[2])
    assert matrix.miou() == 1.0


def test_out_of_range_prediction_is_discarded():
    """A prediction outside [0, C) must not corrupt another class's row."""
    matrix = ConfusionMatrix(2)
    matrix.update(torch.tensor([[[0, 5]]]), torch.tensor([[[0, 1]]]))
    assert matrix.matrix.sum() == 1


def test_accepts_logits_and_indices_equivalently():
    target = torch.tensor([[[0, 1, 1]]])
    logits = torch.zeros(1, 2, 1, 3)
    logits[0, 0, 0, 0] = 5.0
    logits[0, 1, 0, 1] = 5.0
    logits[0, 1, 0, 2] = 5.0
    from_logits = ConfusionMatrix(2)
    from_logits.update(logits, target)
    from_indices = ConfusionMatrix(2)
    from_indices.update(torch.tensor([[[0, 1, 1]]]), target)
    np.testing.assert_array_equal(from_logits.matrix, from_indices.matrix)


def test_pixel_accuracy_differs_from_mean_accuracy_under_imbalance():
    """The two must diverge on imbalanced data, or the imbalance story is unmeasurable."""
    # 98 pixels of class 0 all correct; 2 pixels of class 1 all wrong.
    matrix = ConfusionMatrix(2)
    matrix.update(torch.tensor([[[0] * 98 + [0, 0]]]), torch.tensor([[[0] * 98 + [1, 1]]]))
    assert matrix.pixel_accuracy() == 0.98
    assert matrix.mean_accuracy() == 0.5


def test_summary_includes_per_class_keys():
    matrix = cm_from([0, 1], [0, 1], 2)
    summary = matrix.summary(["a", "b"])
    assert summary["iou/a"] == 1.0 and summary["iou/b"] == 1.0
    assert {"miou", "pixel_acc", "mean_acc", "fw_iou"} <= set(summary)
