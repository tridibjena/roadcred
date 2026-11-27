"""Discovery and label reading for IDD frames.

Kept separate from :mod:`data.prepare_masks` and :mod:`data.sequence_split` so both can
depend on frame discovery without importing each other.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.label_utils import (
    IGNORE_INDEX,
    build_lut,
    check_level_plausible,
    detect_level,
    polygons_to_mask,
    remap_mask,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class FramePaths:
    """A single IDD frame: its image, its label file, and its drive sequence."""

    image: Path
    label: Path
    split: str
    #: Drive-sequence id. IDD groups frames into folders, one per drive/route; frames
    #: from one drive are highly correlated, which is why splits must respect this.
    drive: str

    @property
    def frame_id(self) -> str:
        """Frame identifier within the drive, e.g. ``330189`` or ``frame7540``."""
        return self.image.stem.split("_")[0]

    @property
    def key(self) -> str:
        """Flat, collision-free name that preserves the drive id."""
        return f"{self.drive}_{self.frame_id}"


def find_dataset_root(idd_root: str | Path) -> Path:
    """Locate the directory containing ``leftImg8bit``/``gtFine``.

    Releases nest one or two levels deep (e.g. ``idd_seg/idd20k_lite/leftImg8bit``),
    so the caller can point at the download folder and not at the archive's inner root.
    """
    root = Path(idd_root)
    if (root / "leftImg8bit").exists():
        return root
    for pattern in ("*/leftImg8bit", "*/*/leftImg8bit"):
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0].parent
    return root


def _match_label(label_dir: Path, prefix: str) -> Path | None:
    """Find the semantic label file for a frame prefix.

    ``_inst_label.png`` sorts before ``_label.png`` and also contains "label", so
    instance-label files must be excluded explicitly or they would win the match.
    """
    if not label_dir.is_dir():
        return None
    matches = sorted(
        p
        for p in label_dir.glob(f"{prefix}_*")
        if p.suffix in {".png", ".json"} and "inst" not in p.stem.lower()
    )
    if not matches:
        return None
    for path in matches:  # polygons are authoritative (name-keyed, no ID indirection)
        if path.suffix == ".json" and "polygon" in path.stem:
            return path
    for path in matches:
        if path.suffix == ".png" and "label" in path.stem.lower():
            return path
    return matches[0]


def discover_frames(
    idd_root: str | Path, splits: Sequence[str] = ("train", "val")
) -> list[FramePaths]:
    """Pair IDD images with their label files across releases.

    Handles IDD Lite (``<frame>_image.jpg`` + ``<frame>_label.png``) and full IDD
    Segmentation (``<frame>_leftImg8bit.png`` + ``<frame>_gtFine_polygons.json``) by
    matching on the numeric frame prefix rather than a fixed suffix, since the suffix
    convention differs between releases.

    Args:
        idd_root: Download root; the real dataset root is located automatically.
        splits: Which of IDD's own split directories to read.

    Returns:
        Frames sorted by ``(split, drive, frame)``. Frames whose label is missing are
        skipped, so the label-withheld ``test`` split yields nothing here.
    """
    root = find_dataset_root(idd_root)
    images_root, labels_root = root / "leftImg8bit", root / "gtFine"
    frames: list[FramePaths] = []
    for split in splits:
        split_dir = images_root / split
        if not split_dir.is_dir():
            continue
        for image_path in sorted(split_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES or not image_path.is_file():
                continue
            drive = image_path.parent.name
            label = _match_label(labels_root / split / drive, image_path.stem.split("_")[0])
            if label is not None:
                frames.append(FramePaths(image_path, label, split, drive))
    return sorted(frames, key=lambda f: (f.split, f.drive, f.image.stem))


def read_mask(
    frame: FramePaths, target: str = "level1", lut_cache: dict[str, np.ndarray] | None = None
) -> np.ndarray:
    """Read one frame's label file into the target class space.

    Args:
        frame: The frame to read.
        target: ``level1`` (7 classes) or ``binary``.
        lut_cache: Optional cache keyed by level, to avoid rebuilding the lookup table
            once per image in a dataset-wide loop.

    Returns:
        ``uint8`` mask of class indices, 255 = ignore.
    """
    if frame.label.suffix == ".json":
        mask = polygons_to_mask(json.loads(frame.label.read_text()))
        if target == "binary":
            binary_lut = build_lut("level1Ids", "binary")
            mask = binary_lut[mask]
        return mask

    raw = _read_raw_label(frame)
    level = detect_level(frame.label)
    if lut_cache is None:
        return remap_mask(raw, level=level, target=target)
    cache_key = f"{level}:{target}"
    if cache_key not in lut_cache:
        lut_cache[cache_key] = build_lut(level, target)
    return remap_mask(raw, level=level, lut=lut_cache[cache_key])


def _read_raw_label(frame: FramePaths) -> np.ndarray:
    import cv2

    raw = cv2.imread(str(frame.label), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise OSError(f"Could not read label {frame.label}")
    return raw[..., 0] if raw.ndim == 3 else raw


def audit_label_levels(frames: Sequence[FramePaths]) -> dict[str, int]:
    """Check, before anything is written, that every rendered mask set has a plausible level.

    Reads each rendered label once and records the largest raw ID per detected level, then
    applies :func:`data.label_utils.check_level_plausible`, which raises if masks read as a
    fine encoding never leave the level-1 range.

    Returns:
        Largest raw ID observed per detected level.
    """
    max_raw: dict[str, int] = {}
    for frame in frames:
        if frame.label.suffix == ".json":
            continue  # polygons are rasterised by class name; no ID level involved
        raw = _read_raw_label(frame)
        observed = raw[raw != IGNORE_INDEX]
        level = detect_level(frame.label)
        max_raw[level] = max(max_raw.get(level, 0), int(observed.max()) if observed.size else 0)
    for level, max_id in max_raw.items():
        check_level_plausible(level, max_id)
    return max_raw


__all__ = [
    "FramePaths",
    "discover_frames",
    "read_mask",
    "audit_label_levels",
    "find_dataset_root",
    "IMAGE_SUFFIXES",
]
