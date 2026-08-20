"""A drivable-path score for the forward corridor of a frame.

**What this is.** A descriptive statistic computed from the segmentation the model just
produced: how much of the region ahead of the vehicle was predicted drivable, how confident
the model was about it, and how much of it something else is standing in.

**What this is not.** It is not a safety assessment, and it does not become one by being a
single number. It describes the *model's output*, not the road. `MODEL_CARD.md` states that
this model must not be used for vehicle control, driver assistance or navigation; a score
named "drivability" does not create an exemption. The components are therefore reported
alongside the score rather than collapsed into it, so a reader can see *why* a frame scored
what it did -- and a low score caused by low confidence means something entirely different
from one caused by an obstruction.

**The known failure mode is surfaced deliberately.** This project measured that 17-20% of
true `non-drivable` pixels -- sidewalk, shoulder, the road edge -- are absorbed into
`drivable`, and that six independent interventions failed to fix it. The score is therefore
*optimistic at exactly the boundary that matters most*, and
:func:`drivability` reports the share of the corridor sitting near a predicted class border
so that optimism is visible rather than hidden.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Class indices in the level-1 space, mirroring data.label_utils.LEVEL1_NAMES.
DRIVABLE = 0
NON_DRIVABLE = 1
LIVING_THING = 2
VEHICLES = 3

#: Classes that physically block a path the vehicle would otherwise take.
OBSTRUCTING = (LIVING_THING, VEHICLES)

#: Forward-corridor trapezoid, as fractions of image width/height. Wide at the bottom of
#: the frame (near the vehicle) and narrow further up (further away), which is the shape a
#: forward path projects to under a normal windscreen-mounted camera. Deliberately stops
#: well short of the horizon: at 320x227 the far field is a handful of pixels and the
#: model's per-class reliability there is not something this score should be asserting on.
CORRIDOR_TOP_Y = 0.58
CORRIDOR_TOP_HALF_WIDTH = 0.09
CORRIDOR_BOTTOM_HALF_WIDTH = 0.30


def corridor_mask(height: int, width: int) -> np.ndarray:
    """Boolean mask of the forward corridor for an image of this size.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        ``(height, width)`` boolean array, ``True`` inside the corridor.
    """
    rows = np.arange(height)[:, None]
    columns = np.arange(width)[None, :]

    top = CORRIDOR_TOP_Y * height
    # 0 at the top of the trapezoid, 1 at the bottom of the frame.
    depth = np.clip((rows - top) / max(height - top, 1.0), 0.0, 1.0)
    half = (
        CORRIDOR_TOP_HALF_WIDTH
        + (CORRIDOR_BOTTOM_HALF_WIDTH - CORRIDOR_TOP_HALF_WIDTH) * depth
    ) * width

    centre = width / 2.0
    return (rows >= top) & (np.abs(columns - centre) <= half)


def _boundary_fraction(prediction: np.ndarray, region: np.ndarray) -> float:
    """Share of the region whose predicted class differs from a 4-neighbour.

    Boundary pixels are where this model is least reliable -- per-class IoU correlates at
    r = -0.92 with a class's boundary share -- so a corridor made mostly of boundary is a
    corridor whose score should be trusted less.
    """
    if not region.any():
        return 0.0
    differs = np.zeros_like(prediction, dtype=bool)
    differs[:, :-1] |= prediction[:, :-1] != prediction[:, 1:]
    differs[:, 1:] |= prediction[:, 1:] != prediction[:, :-1]
    differs[:-1, :] |= prediction[:-1, :] != prediction[1:, :]
    differs[1:, :] |= prediction[1:, :] != prediction[:-1, :]
    return float(differs[region].mean())


def drivability(
    prediction: np.ndarray,
    confidence: np.ndarray,
    low_confidence_threshold: float = 0.6,
) -> dict[str, Any]:
    """Score the forward corridor of one segmented frame.

    The score is the product of three measured quantities, each in ``[0, 1]``::

        score = coverage * mean_confidence * (1 - obstruction)

    A product rather than a weighted sum, because these are not interchangeable: a corridor
    that is 100% road but blocked by a vehicle is not "mostly fine", and averaging would
    say it was. Any one component collapsing should collapse the score.

    Args:
        prediction: ``(H, W)`` predicted class indices, level-1 space.
        confidence: ``(H, W)`` per-pixel confidence, ideally temperature-calibrated.
        low_confidence_threshold: Pixels below this count toward ``low_confidence``.

    Returns:
        The score and every component that produced it, plus the caveat fields.
    """
    prediction = np.asarray(prediction)
    confidence = np.asarray(confidence, dtype=np.float64)
    region = corridor_mask(*prediction.shape[:2])
    area = int(region.sum())
    if area == 0:  # degenerate image; report nothing rather than divide by zero
        return {
            "score": 0.0, "coverage": 0.0, "mean_confidence": 0.0, "obstruction": 0.0,
            "boundary_fraction": 0.0, "low_confidence": 0.0, "corridor_pixels": 0,
            "road_edge_caveat": True,
        }

    in_corridor = prediction[region]
    corridor_confidence = confidence[region]

    is_drivable = in_corridor == DRIVABLE
    coverage = float(is_drivable.mean())
    # Confidence is averaged over the drivable pixels only: the question is how sure the
    # model is about the road it claims to see, not about the whole corridor.
    mean_confidence = float(corridor_confidence[is_drivable].mean()) if is_drivable.any() else 0.0
    obstruction = float(np.isin(in_corridor, OBSTRUCTING).mean())

    score = coverage * mean_confidence * (1.0 - obstruction)
    return {
        "score": score,
        "coverage": coverage,
        "mean_confidence": mean_confidence,
        "obstruction": obstruction,
        "boundary_fraction": _boundary_fraction(prediction, region),
        "low_confidence": float((corridor_confidence < low_confidence_threshold).mean()),
        "corridor_pixels": area,
        # True whenever the corridor touches a predicted drivable/non-drivable border --
        # the one transition this model is measurably optimistic about.
        "road_edge_caveat": bool(
            (in_corridor == DRIVABLE).any() and (in_corridor == NON_DRIVABLE).any()
        ),
    }


__all__ = ["drivability", "corridor_mask", "DRIVABLE", "NON_DRIVABLE", "OBSTRUCTING"]
