"""Drivers for the experiment suites reported in RESULTS.md.

Each suite trains a set of models that differ in exactly one factor, and writes a tidy
CSV to ``results/``. Every run also appends to ``results/runs.csv`` via
:func:`modeling.tracking.log_run`, so an individual run can always be traced back to its
config, seed and git SHA.

Results are written after *every* run rather than at the end, so a suite that is
interrupted still leaves usable partial output.

Run::

    python scripts/run_ablations.py --suite split
    python scripts/run_ablations.py --suite loss --seeds 0
    python scripts/run_ablations.py --suite arch
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modeling.train import train  # noqa: E402

#: Columns kept in the per-suite CSVs, in order. Per-class IoU columns are appended.
BASE_COLUMNS = [
    "experiment",
    "data",
    "architecture",
    "encoder",
    "encoder_weights",
    "loss",
    "seed",
    "miou",
    "mean_acc",
    "pixel_acc",
    "fw_iou",
    "val_loss",
    "params",
    "epochs_ran",
    "train_seconds",
    "checkpoint",
]


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write suite results, ordering known columns first and per-class IoU after."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return path
    extra = sorted({k for row in rows for k in row} - set(BASE_COLUMNS))
    fieldnames = [c for c in BASE_COLUMNS if any(c in row for row in rows)] + extra
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_suite(
    jobs: list[dict[str, Any]], out_csv: str | Path, common: dict[str, Any]
) -> list[dict[str, Any]]:
    """Run a list of training jobs sequentially, checkpointing the CSV after each."""
    rows: list[dict[str, Any]] = []
    started = time.time()
    for index, job in enumerate(jobs, 1):
        label = job.pop("_label")
        print(f"\n[{index}/{len(jobs)}] {label}", flush=True)
        try:
            result = train(**{**common, **job})
        except Exception as exc:  # noqa: BLE001 - one bad cell must not kill the suite
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            rows.append({"experiment": label, "error": f"{type(exc).__name__}: {exc}"})
            write_csv(rows, out_csv)
            continue
        result["experiment"] = label
        rows.append(result)
        write_csv(rows, out_csv)
        print(f"  -> mIoU {result['miou']:.4f}  ({result['train_seconds']:.0f}s)", flush=True)
    print(f"\nSuite finished in {(time.time() - started) / 60:.1f} min -> {out_csv}", flush=True)
    return rows


def suite_split(seeds: Iterable[int]) -> tuple[list[dict], str]:
    """Does a random frame split overstate generalization vs a drive-disjoint split?

    Trains the identical model on three datasets that differ *only* in how frames were
    assigned to train/val. ``level1_frame`` is the leaky control.
    """
    jobs = [
        {
            "_label": f"{mode}_s{seed}",
            "data_root": f"data/processed/level1_{mode}",
            "seed": seed,
            "run_name": f"split_{mode}_s{seed}",
        }
        for mode in ("official", "sequence", "frame")
        for seed in seeds
    ]
    return jobs, "results/split_experiment.csv"


def suite_loss(seeds: Iterable[int]) -> tuple[list[dict], str]:
    """Which loss best handles IDD Lite's class imbalance?

    Held fixed: architecture, encoder, schedule, augmentation, split, seed.
    """
    jobs = [
        {
            "_label": f"{loss}_s{seed}",
            "loss": loss,
            "seed": seed,
            "run_name": f"loss_{loss}_s{seed}",
        }
        for loss in ("ce", "weighted_ce", "dice", "tversky", "boundary")
        for seed in seeds
    ]
    return jobs, "results/loss_ablation.csv"


def suite_arch(seeds: Iterable[int]) -> tuple[list[dict], str]:
    """Accuracy vs size across decoder architectures at a fixed encoder and loss."""
    jobs = [
        {
            "_label": f"{arch}_{encoder}_s{seed}",
            "architecture": arch,
            "encoder": encoder,
            "seed": seed,
            "run_name": f"arch_{arch}_{encoder}_s{seed}",
        }
        for arch, encoder in (
            ("deeplabv3plus", "resnet34"),
            ("unet", "resnet34"),
            ("fpn", "resnet34"),
            ("deeplabv3plus", "mobilenet_v2"),
        )
        for seed in seeds
    ]
    return jobs, "results/architecture_comparison.csv"


def suite_pretrain(seeds: Iterable[int]) -> tuple[list[dict], str]:
    """How much does ImageNet initialisation buy at this labelled-data scale?"""
    jobs = [
        {
            "_label": f"{'imagenet' if weights else 'scratch'}_s{seed}",
            "encoder_weights": weights,
            "seed": seed,
            "run_name": f"init_{'imagenet' if weights else 'scratch'}_s{seed}",
        }
        for weights in ("imagenet", None)
        for seed in seeds
    ]
    return jobs, "results/pretrain_ablation.csv"


SUITES = {
    "split": suite_split,
    "loss": suite_loss,
    "arch": suite_arch,
    "pretrain": suite_pretrain,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=sorted(SUITES) + ["all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--data", default="data/processed/level1_official")
    args = parser.parse_args()

    common = {
        "data_root": args.data,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "tensorboard": True,
        "verbose": True,
    }

    names = sorted(SUITES) if args.suite == "all" else [args.suite]
    for name in names:
        jobs, out_csv = SUITES[name](args.seeds)
        print(f"\n{'=' * 70}\nSUITE: {name}  ({len(jobs)} runs)\n{'=' * 70}", flush=True)
        run_suite(jobs, out_csv, common)


if __name__ == "__main__":
    main()
