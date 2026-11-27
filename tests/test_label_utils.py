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


def test_range_check_also_runs_with_a_precomputed_lut():
    """Regression: the dataset build passes a cached lookup table, and the range check
    used to run only without one -- so it never ran on real data."""
    level3_mask = np.array([[0, 20], [24, 25]], dtype=np.uint8)
    with pytest.raises(ValueError, match="exceeds the maximum"):
        remap_mask(level3_mask, level="level1Ids", lut=build_lut("level1Ids"))


def test_level1_masks_read_as_level3_are_rejected_dataset_wide():
    """The IDD Lite bug: level-1 IDs read as level-3 map non-drivable onto drivable. One
    mask cannot reveal it (0-6 is a legal level-3 range); the whole dataset can."""
    from data.label_utils import check_level_plausible

    with pytest.raises(ValueError, match="look like level-1"):
        check_level_plausible("level3Ids", 6)
    check_level_plausible("level3Ids", 25)  # real level-3 data reaches sky (25)
    check_level_plausible("level1Ids", 6)  # level-1 data read as level-1 is fine


def test_real_idd_lite_labels_pass_the_level_audit(idd_available):
    if not idd_available:
        pytest.skip("IDD not downloaded")
    from data.idd_frames import audit_label_levels, discover_frames

    assert audit_label_levels(discover_frames("data/raw/idd_seg")) == {"level1Ids": 6}
