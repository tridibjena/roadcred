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
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from data.idd_frames import FramePaths, audit_label_levels, discover_frames, read_mask
from data.sequence_split import SplitMap, frame_gap_stats, leakage_report, make_splits
from modeling.config import REPO_ROOT

#: Every split strategy, in the order ``--split-mode all`` builds them.
SPLIT_MODES = ("official", "sequence", "sequence3", "frame")


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
    # Written repo-relative where possible. An absolute path here bakes one machine's
    # filesystem layout into a committed file and breaks on every other checkout; only
    # the `names` field is read back by this project, so portability costs nothing.
    try:
        dataset_path = str(dataset_dir.relative_to(REPO_ROOT))
    except ValueError:
        dataset_path = str(dataset_dir)
    config_path = Path(config_dir) / f"{variant}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "path": dataset_path,
                "train": "images/train",
                "val": "images/val",
                **({"test": "images/test"} if (dataset_dir / "images" / "test").is_dir() else {}),
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


def dataset_stats(
    frames: Sequence[FramePaths], val_fraction: float, seed: int
) -> dict[str, object]:
    """The dataset facts RESULTS.md quotes, measured rather than transcribed.

    Leakage is reported for every split strategy, built exactly as the variants are, so
    the contamination rate quoted for the random split is the one the trained models saw.
    """
    return {
        "n_frames": len(frames),
        "n_drives": len({f.drive for f in frames}),
        "val_fraction": val_fraction,
        "seed": seed,
        "splits": {
            mode: leakage_report(make_splits(frames, mode, val_fraction, seed))
            for mode in SPLIT_MODES
        },
        "frame_gaps": frame_gap_stats(frames),
    }


def main() -> None:
    """CLI: ``python -m data.prepare_masks --split-mode official --target level1``."""
    import json

    from modeling.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional YAML config")
    parser.add_argument("--target", default="level1", choices=["level1", "binary"])
    parser.add_argument(
        "--split-mode",
        default="official",
        choices=[*SPLIT_MODES, "all"],
        help="'frame' is the deliberately leaky control; 'all' builds every variant",
    )
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stats-out", default="results/dataset_stats.json",
        help="Where the measured dataset facts (leakage, frame gaps) are written",
    )
    parser.add_argument(
        "--stats-only", action="store_true", help="Write the dataset facts and build nothing"
    )
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
    # Before anything is written: a mis-detected label level corrupts every variant.
    print(f"Label levels: {audit_label_levels(frames)} (largest raw ID per detected level)")

    stats_path = REPO_ROOT / args.stats_out
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(dataset_stats(frames, val_fraction, args.seed), indent=2) + "\n"
    )
    print(f"Dataset facts -> {stats_path.relative_to(REPO_ROOT)}")
    if args.stats_only:
        return

    modes = list(SPLIT_MODES) if args.split_mode == "all" else [args.split_mode]
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
