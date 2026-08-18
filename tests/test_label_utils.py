"""Label-mapping tests.

The IDD ID schemes are the most error-prone part of this pipeline: the same numeric value
means different classes at different levels, and a silent mis-mapping would corrupt every
downstream number without ever raising.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.label_utils import (
    DRIVABLE_NAMES,
    LEVEL1_NAMES,
    build_lut,
    detect_level,
    drivable_names,
    max_id_for_level,
    remap_mask,
)


def test_drivable_names_match_idd_category():
    """The frozen name set must agree with IDD's own ``drivable`` category."""
    assert drivable_names() == DRIVABLE_NAMES
    assert DRIVABLE_NAMES == {"road", "parking", "drivable fallback"}


@pytest.mark.parametrize(
    "level,expected_drivable",
    [("level1Ids", [0]), ("level3Ids", [0, 1]), ("id", [0, 1, 2])],
)
def test_binary_lut_drivable_ids(level, expected_drivable):
    """Exactly the drivable IDs map to class 1, at every ID encoding."""
    lut = build_lut(level, "binary")
    assert np.where(lut == 1)[0].tolist() == expected_drivable


def test_level1_lut_is_identity_for_level1_input():
    """level1 IDs 0-6 pass through unchanged; void maps to ignore."""
    lut = build_lut("level1Ids", "level1")
    assert lut[:7].tolist() == list(range(7))
    assert lut[255] == 255


def test_level3_collapses_into_seven_classes():
    """Full-IDD level3 IDs collapse to the same 7-class space."""
    lut = build_lut("level3Ids", "level1")
    # level3: 0=road, 1=parking/drivable-fallback -> both level1 0 (drivable)
    assert lut[0] == 0 and lut[1] == 0
    # level3: 2=sidewalk, 3=rail track/non-drivable fallback -> level1 1
    assert lut[2] == 1 and lut[3] == 1
    assert set(lut[: max_id_for_level("level3Ids") + 1].tolist()) <= set(range(len(LEVEL1_NAMES)))


def test_unassigned_ids_map_to_ignore_not_a_class():
    """IDs IDD never assigns must be ignored, never scored as a real class."""
    lut = build_lut("level1Ids", "level1")
    assert lut[7] == 255 and lut[100] == 255


def test_detect_level_handles_idd_lite():
    """IDD Lite ships bare ``_label.png`` files that nonetheless hold level1 IDs."""
    assert detect_level("data/raw/idd_seg/idd20k_lite/gtFine/train/135/330189_label.png") == "level1Ids"
    assert detect_level("x/y_gtFine_labellevel3Ids.png") == "level3Ids"
    assert detect_level("x/y_gtFine_labelids.png") == "id"


def test_remap_rejects_out_of_range_ids():
    """A level mis-detection must fail loudly rather than silently mis-map."""
    mask = np.array([[0, 25]], dtype=np.uint8)  # 25 is valid at level3, not at level1
    with pytest.raises(ValueError, match="exceeds the maximum"):
        remap_mask(mask, level="level1Ids")


def test_remap_preserves_ignore(synthetic_mask):
    """255 stays 255 through remapping."""
    out = remap_mask(synthetic_mask, level="level1Ids", target="level1")
    assert (out[:2, :] == 255).all()
    assert out.shape == synthetic_mask.shape
