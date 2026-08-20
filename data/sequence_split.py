"""Train/val splitting strategies, and the leakage they do or do not cause.

IDD groups frames into *drive sequences* -- one folder per drive/route. Frames from the
same drive are captured seconds apart on the same road in the same lighting, so they are
strongly correlated. Splitting frames at random therefore puts near-duplicates on both
sides of the train/val boundary, and the resulting mIoU measures memorisation as much as
generalisation.

This module provides three strategies so that effect can be *measured* rather than
asserted:

* :func:`official_split`  -- IDD's own train/val directories. Already drive-disjoint.
* :func:`sequence_split`  -- re-split the pooled frames, holding out whole drives.
* :func:`frame_split`     -- re-split the pooled frames at random, ignoring drives.
  Deliberately leaky, and included as the *control*: the gap between this and
  :func:`sequence_split` is the headline generalization result.

:func:`leakage_report` quantifies the overlap for any split, and is asserted in the test
suite so the claim stays enforced rather than merely documented.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence

from data.idd_frames import FramePaths

SplitMap = dict[str, list[FramePaths]]


def group_by_drive(frames: Iterable[FramePaths]) -> dict[str, list[FramePaths]]:
    """Group frames by their drive-sequence id."""
    groups: dict[str, list[FramePaths]] = defaultdict(list)
    for frame in frames:
        groups[frame.drive].append(frame)
    return dict(groups)


def official_split(frames: Iterable[FramePaths]) -> SplitMap:
    """Use IDD's own ``train``/``val`` directories unchanged.

    For IDD Lite these are already drive-disjoint (309 train drives vs 61 val drives,
    no overlap), so this is a legitimate generalization split as shipped.
    """
    splits: SplitMap = defaultdict(list)
    for frame in frames:
        splits[frame.split].append(frame)
    return dict(splits)


def sequence_split(
    frames: Sequence[FramePaths], val_fraction: float = 0.15, seed: int = 0
) -> SplitMap:
    """Hold out whole drive sequences, so no validation drive is seen in training.

    Drives are shuffled and assigned to validation until the *frame* quota is met, which
    keeps the split near the requested fraction despite drives varying from 1 to 36
    frames.

    Args:
        frames: All labelled frames, pooled across IDD's own splits.
        val_fraction: Approximate fraction of frames to hold out.
        seed: Shuffling seed.

    Returns:
        ``{"train": [...], "val": [...]}``.
    """
    groups = group_by_drive(frames)
    drives = sorted(groups)
    random.Random(seed).shuffle(drives)

    target = int(round(len(frames) * val_fraction))
    val_drives: set[str] = set()
    count = 0
    for drive in drives:
        if count >= target:
            break
        val_drives.add(drive)
        count += len(groups[drive])

    return {
        "train": [f for f in frames if f.drive not in val_drives],
        "val": [f for f in frames if f.drive in val_drives],
    }


def sequence_split_3way(
    frames: Sequence[FramePaths],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> SplitMap:
    """Hold out whole drives for *both* validation and a final test set.

    Why this exists: with a single held-out split, validation does double duty -- it is
    the early-stopping criterion *and* the set every number is reported on -- so model
    selection has seen the reporting set. IDD ships test images with the labels withheld,
    so a genuine test set has to be carved out of the labelled frames here.

    All three partitions are drive-disjoint from each other, not merely val-from-train:
    a test set sharing drives with validation would be contaminated by model selection
    just as surely as one sharing drives with training.

    Args:
        frames: All labelled frames, pooled across IDD's own splits.
        val_fraction: Approximate frame fraction for validation.
        test_fraction: Approximate frame fraction for test.
        seed: Shuffling seed.

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}``.
    """
    groups = group_by_drive(frames)
    drives = sorted(groups)
    random.Random(seed).shuffle(drives)

    # Fill test first, then val, from one shuffled ordering -- so changing val_fraction
    # cannot silently reshuffle which drives are in test.
    assigned: dict[str, str] = {}
    for name, fraction in (("test", test_fraction), ("val", val_fraction)):
        target = int(round(len(frames) * fraction))
        count = 0
        for drive in drives:
            if count >= target:
                break
            if drive in assigned:
                continue
            assigned[drive] = name
            count += len(groups[drive])

    out: SplitMap = {"train": [], "val": [], "test": []}
    for frame in frames:
        out[assigned.get(frame.drive, "train")].append(frame)
    return out


def frame_split(
    frames: Sequence[FramePaths], val_fraction: float = 0.15, seed: int = 0
) -> SplitMap:
    """Split individual frames at random, ignoring drive membership.

    .. warning::
       This split **leaks**. Frames from one drive land on both sides, so validation
       frames have near-duplicates in training. It exists purely as the control arm that
       shows how much a naive random split inflates reported mIoU -- it must never be
       used to report a headline number.
    """
    shuffled = list(frames)
    random.Random(seed).shuffle(shuffled)
    cut = int(round(len(shuffled) * val_fraction))
    return {"train": shuffled[cut:], "val": shuffled[:cut]}


def make_splits(
    frames: Sequence[FramePaths],
    mode: str = "official",
    val_fraction: float = 0.15,
    seed: int = 0,
) -> SplitMap:
    """Dispatch by name (``official`` / ``sequence`` / ``sequence3`` / ``frame``)."""
    if mode == "official":
        return official_split(frames)
    if mode == "sequence":
        return sequence_split(frames, val_fraction, seed)
    if mode == "sequence3":
        return sequence_split_3way(frames, val_fraction, val_fraction, seed)
    if mode == "frame":
        return frame_split(frames, val_fraction, seed)
    raise ValueError(
        f"Unknown split mode {mode!r}; expected official/sequence/sequence3/frame"
    )


def leakage_report(splits: SplitMap) -> dict[str, int | float | list[str]]:
    """Quantify train/val overlap for a split.

    Returns:
        A dict with the number of shared drives, how many validation frames sit in a
        drive that also appears in training, and that count as a fraction. For a correct
        sequence split every value is zero.
    """
    train_drives = {f.drive for f in splits.get("train", [])}
    val_frames = splits.get("val", [])
    shared = sorted(train_drives & {f.drive for f in val_frames})
    contaminated = [f for f in val_frames if f.drive in train_drives]
    report: dict[str, int | float | list[str]] = {
        "shared_drives": len(shared),
        "shared_drive_ids": shared[:10],
        "contaminated_val_frames": len(contaminated),
        "contaminated_fraction": (len(contaminated) / len(val_frames)) if val_frames else 0.0,
        "n_train": len(splits.get("train", [])),
        "n_val": len(val_frames),
    }
    # A test set must be disjoint from *both* other partitions: sharing drives with
    # validation contaminates it through model selection, not just through training.
    test_frames = splits.get("test")
    if test_frames is not None:
        selection_drives = train_drives | {f.drive for f in val_frames}
        leaked = [f for f in test_frames if f.drive in selection_drives]
        report["n_test"] = len(test_frames)
        report["contaminated_test_frames"] = len(leaked)
        report["contaminated_test_fraction"] = (
            len(leaked) / len(test_frames) if test_frames else 0.0
        )
    return report


__all__ = [
    "group_by_drive",
    "official_split",
    "sequence_split",
    "sequence_split_3way",
    "frame_split",
    "make_splits",
    "leakage_report",
    "SplitMap",
]
