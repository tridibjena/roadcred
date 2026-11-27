"""Boundary-band tests. The frame edge must never count as an object edge."""

from __future__ import annotations

import numpy as np

from evaluation.boundary_iou import boundary_iou_per_class, boundary_region


def test_a_class_filling_the_frame_has_no_boundary():
    """Regression: zero-padding before erosion made a 2px frame around every
    border-touching class count as boundary."""
    assert not boundary_region(np.ones((8, 10), dtype=bool), dilation=2).any()


def test_band_follows_the_internal_edge_only():
    mask = np.zeros((6, 10), dtype=bool)
    mask[:, :5] = True  # left half; its one object edge is the line at column 5
    band = boundary_region(mask, dilation=2)
    assert band[:, 3:5].all()  # the two columns beside the edge
    assert not band[:, :3].any()  # interior, including the frame edge on the left
    assert not band[:, 5:].any()  # outside the mask


def test_matching_frame_edges_do_not_inflate_boundary_iou():
    """Two predictions that agree everywhere except along the one internal edge must score
    the same however much frame edge they share."""
    target = np.zeros((20, 20), dtype=np.uint8)
    target[:, 10:] = 1
    pred = target.copy()
    pred[:, 9] = 1  # the internal edge moves by one pixel
    iou = boundary_iou_per_class(pred, target, n_classes=2, dilation=2)
    # Each class's band is the two columns beside its edge; shifting the edge by one
    # column overlaps one of three columns. Counting the frame edge as boundary scored
    # this ~0.70, from border pixels that match trivially.
    assert np.allclose(iou, 1 / 3)
