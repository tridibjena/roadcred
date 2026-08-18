"""Materialise IDD frames into a trainable dataset (images + class-index PNG masks).

Writes the layout Ultralytics' ``semantic`` task expects, which the hand-written PyTorch
loop in :mod:`modeling.train` also reads::

    data/processed/<variant>/
      images/{train,val}/<drive>_<frame>.jpg
      masks/{train,val}/<drive>_<frame>.png   # class indices, 255 = ignore
    configs/<variant>.yaml

One variant is produced per (target, split-mode) pair, so the leakage experiment compares
datasets that differ *only* in how frames were assigned to train/val.

Every dataset here comes from IDD. There are no external data sources in this project.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from data.idd_frames import FramePaths, discover_frames, read_mask
from data.sequence_split import SplitMap, leakage_report, make_splits


def build_dataset(
    splits: SplitMap,
    out_root: str | Path,
    variant: str,
    *,
    target: str = "level1",
    max_size: int | None = None,
) -> dict[str, int]:
    """Write images and masks for every split of one variant.

    Args:
        splits: Mapping of split name to frames, from :func:`data.sequence_split.make_splits`.
        out_root: Parent directory; the variant gets its own subdirectory.
        variant: Directory name for this variant, e.g. ``level1_official``.
        target: ``level1`` (7 classes) or ``binary``.
        max_size: If set, the longest image side is downscaled to this. Masks are resized
            with nearest-neighbour so class indices are never interpolated into
            nonexistent classes.

    Returns:
        Frame counts per split.
    """
    import cv2

    out_root = Path(out_root) / variant
    lut_cache: dict[str, np.ndarray] = {}
    stats: dict[str, int] = {}

    for split, frames in splits.items():
        for sub in ("images", "masks"):
            (out_root / sub / split).mkdir(parents=True, exist_ok=True)
        written = 0
        for frame in frames:
            image = cv2.imread(str(frame.image))
            if image is None:
                continue
            mask = read_mask(frame, target=target, lut_cache=lut_cache)
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(
                    mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
                )
            if max_size and max(image.shape[:2]) > max_size:
                scale = max_size / max(image.shape[:2])
                size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
                image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(out_root / "images" / split / f"{frame.key}.jpg"), image)
            cv2.imwrite(str(out_root / "masks" / split / f"{frame.key}.png"), mask)
            written += 1
        stats[split] = written
    return stats


def write_dataset_yaml(
    out_root: str | Path, variant: str, class_names: Sequence[str], config_dir: str | Path
) -> Path:
    """Write the Ultralytics dataset YAML for a prepared variant."""
    import yaml

    dataset_dir = (Path(out_root) / variant).resolve()
    config_path = Path(config_dir) / f"{variant}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_dir),
                "train": "images/train",
                "val": "images/val",
                "masks_dir": "masks",
                "nc": len(class_names),
                "names": list(class_names),
            },
            sort_keys=False,
        )
    )
    return config_path


def class_pixel_counts(
    splits: SplitMap, n_classes: int, target: str = "level1"
) -> dict[str, np.ndarray]:
    """Count pixels per class per split.

    Used to derive weighted-cross-entropy class weights from the *training* split only
    (deriving them from validation would leak), and to report the imbalance that
    motivates the loss ablation.
    """
    lut_cache: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for split, frames in splits.items():
        total = np.zeros(n_classes, dtype=np.int64)
        for frame in frames:
            mask = read_mask(frame, target=target, lut_cache=lut_cache)
            valid = mask[mask != 255]
            total += np.bincount(valid, minlength=n_classes)[:n_classes]
        counts[split] = total
    return counts


def main() -> None:
    """CLI: ``python -m data.prepare_masks --split-mode official --target level1``."""
    from modeling.config import REPO_ROOT, load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional YAML config")
    parser.add_argument("--target", default="level1", choices=["level1", "binary"])
    parser.add_argument(
        "--split-mode",
        default="official",
        choices=["official", "sequence", "frame", "all"],
        help="'frame' is the deliberately leaky control; 'all' builds every variant",
    )
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    idd_root = cfg.data.resolved("idd_root")
    out_root = cfg.data.resolved("out_root")
    val_fraction = (
        args.val_fraction if args.val_fraction is not None else cfg.data.val_sequence_fraction
    )
    class_names = (
        cfg.data.class_names if args.target == "level1" else ["nondrivable", "drivable"]
    )

    frames = discover_frames(idd_root)
    if not frames:
        raise SystemExit(f"No labelled frames under {idd_root}. See DOWNLOADS.md.")
    print(
        f"Discovered {len(frames)} labelled frames across "
        f"{len({f.drive for f in frames})} drive sequences"
    )

    modes = ["official", "sequence", "frame"] if args.split_mode == "all" else [args.split_mode]
    for mode in modes:
        splits = make_splits(frames, mode, val_fraction, args.seed)
        report = leakage_report(splits)
        variant = f"{args.target}_{mode}"
        stats = build_dataset(
            splits,
            out_root,
            variant,
            target=args.target,
            max_size=args.max_size if args.max_size is not None else cfg.data.max_size,
        )
        path = write_dataset_yaml(out_root, variant, class_names, REPO_ROOT / "configs")
        leak = 100 * report["contaminated_fraction"]
        print(
            f"[{variant}] {stats} | shared_drives={report['shared_drives']} "
            f"leaked_val={report['contaminated_val_frames']} ({leak:.1f}%) -> {path.name}"
        )


if __name__ == "__main__":
    main()
