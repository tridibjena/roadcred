"""Split-strategy tests, including the leakage assertion.

The headline generalization claim in this project is that a drive-disjoint split is
harder and more honest than a random frame split. These tests keep that claim enforced:
if a future change lets a drive straddle the train/val boundary, CI fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data.idd_frames import FramePaths
from data.sequence_split import (
    frame_split,
    group_by_drive,
    leakage_report,
    make_splits,
    sequence_split,
)


def make_frames(n_drives: int = 20, per_drive: int = 5) -> list[FramePaths]:
    """Synthetic frames: ``n_drives`` drives with ``per_drive`` correlated frames each."""
    return [
        FramePaths(
            image=Path(f"img/d{d}/f{i}_image.jpg"),
            label=Path(f"lbl/d{d}/f{i}_label.png"),
            split="train",
            drive=f"d{d}",
        )
        for d in range(n_drives)
        for i in range(per_drive)
    ]


def test_group_by_drive():
    groups = group_by_drive(make_frames(4, 3))
    assert len(groups) == 4 and all(len(v) == 3 for v in groups.values())


def test_sequence_split_has_zero_leakage():
    """No drive may appear on both sides of a sequence split."""
    splits = sequence_split(make_frames(), val_fraction=0.2, seed=0)
    report = leakage_report(splits)
    assert report["shared_drives"] == 0
    assert report["contaminated_val_frames"] == 0
    assert report["contaminated_fraction"] == 0.0


def test_sequence_split_respects_requested_fraction():
    frames = make_frames(40, 5)
    splits = sequence_split(frames, val_fraction=0.2, seed=0)
    ratio = len(splits["val"]) / len(frames)
    assert 0.15 <= ratio <= 0.30  # drives are granular, so allow slack


def test_sequence_split_is_deterministic_per_seed():
    a = sequence_split(make_frames(), 0.2, seed=7)
    b = sequence_split(make_frames(), 0.2, seed=7)
    assert [f.key for f in a["val"]] == [f.key for f in b["val"]]


def test_frame_split_leaks_and_is_labelled_as_the_control():
    """The random split is expected to leak -- that is the whole point of the control."""
    splits = frame_split(make_frames(), val_fraction=0.2, seed=0)
    assert leakage_report(splits)["shared_drives"] > 0


def test_splits_partition_without_loss_or_duplication():
    frames = make_frames()
    for mode in ("sequence", "frame"):
        splits = make_splits(frames, mode, 0.2, 0)
        keys = [f.key for f in splits["train"]] + [f.key for f in splits["val"]]
        assert len(keys) == len(frames)
        assert len(set(keys)) == len(frames)


@pytest.mark.parametrize("mode", ["official", "sequence"])
def test_real_idd_splits_are_drive_disjoint(mode, idd_available):
    """On the real download, both honest splits must show zero leakage."""
    if not idd_available:
        pytest.skip("IDD not downloaded")
    from data.idd_frames import discover_frames

    frames = discover_frames("data/raw/idd_seg")
    assert frames, "IDD present but no frames discovered"
    report = leakage_report(make_splits(frames, mode, 0.15, 0))
    assert report["shared_drives"] == 0, f"{mode} split leaked drives"
    assert report["contaminated_val_frames"] == 0
