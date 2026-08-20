"""Error analysis: what the model gets wrong, where, and whether the pattern is systematic.

An aggregate mIoU says how much is wrong but nothing about *what*. This module produces
the three artefacts that answer that:

* a row-normalised confusion matrix, showing which class pairs are actually confused;
* per-class IoU against class frequency, showing whether errors track rarity;
* the worst frames by per-image mIoU, rendered as image / ground truth / prediction /
  error map, so failures can be inspected rather than guessed at.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.metrics import ConfusionMatrix
from modeling.dataset import IMAGENET_MEAN, IMAGENET_STD, IDDSegmentation

IGNORE_INDEX = 255

#: Distinct colours for the 7 level-1 classes, chosen to stay distinguishable when
#: rendered side by side and in greyscale print.
CLASS_COLOURS = np.array(
    [
        [128, 64, 128],   # drivable            - purple
        [244, 35, 232],   # non-drivable        - magenta
        [220, 20, 60],    # living-thing        - crimson
        [0, 0, 230],      # vehicles            - blue
        [190, 153, 153],  # barrier-structures  - dusty rose
        [107, 142, 35],   # construction-veg    - olive
        [70, 130, 180],   # sky                 - steel blue
    ],
    dtype=np.uint8,
)


def colourise(mask: np.ndarray) -> np.ndarray:
    """Map a class-index mask to RGB, rendering ignore pixels black."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for index, colour in enumerate(CLASS_COLOURS):
        out[mask == index] = colour
    return out


def denormalise(tensor: torch.Tensor) -> np.ndarray:
    """Invert ImageNet normalisation to a displayable ``uint8`` RGB image."""
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    array = tensor.detach().cpu().numpy() * std + mean
    return np.clip(array.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)


def per_image_miou(pred: np.ndarray, target: np.ndarray, n_classes: int) -> float:
    """mIoU for a single image, over classes present in its ground truth."""
    confusion = ConfusionMatrix(n_classes)
    confusion.update(torch.from_numpy(pred[None]), torch.from_numpy(target[None]))
    return confusion.miou()


def class_geometry(
    data_root: str | Path, n_classes: int, split: str = "train", dilation: int = 2
) -> dict[str, list[float]]:
    """Per-class shape statistics, computed from ground-truth masks alone.

    Motivates a distinction the per-class IoU table cannot make on its own. The obvious
    reading of that table is that IoU tracks class *rarity*; but rarity is confounded with
    *thinness* in this dataset, and the two make different predictions. ``barrier-structures``
    is 8.5x more common than ``living-thing`` and scores the same IoU, which a pure frequency
    story cannot explain and a geometry story predicts.

    Needs no model and no GPU -- it reads the label PNGs only -- so it is cheap enough to
    run alongside the model-dependent analysis.

    Args:
        data_root: Prepared variant directory.
        n_classes: Number of classes.
        split: Which split's masks to measure. Defaults to ``train``, the distribution the
            model actually learned from.
        dilation: Band width, in pixels, defining "near a boundary".

    Returns:
        Per-class ``boundary_fraction`` (share of a class's pixels within ``dilation`` of a
        class border), ``mean_component_pixels`` (mean connected-component area), and
        ``pixel_fraction``, each a list indexed by class.
    """
    import cv2

    kernel = np.ones((3, 3), np.uint8)
    edge = np.zeros(n_classes)
    total = np.zeros(n_classes)
    components = np.zeros(n_classes)

    for path in sorted((Path(data_root) / "masks" / split).glob("*.png")):
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = mask[..., 0]
        for cls in range(n_classes):
            binary = (mask == cls).astype(np.uint8)
            count = int(binary.sum())
            if not count:
                continue
            total[cls] += count
            # Interior = what survives erosion; the rest of the class is within
            # `dilation` px of a border with some other class.
            edge[cls] += count - int(cv2.erode(binary, kernel, iterations=dilation).sum())
            components[cls] += cv2.connectedComponents(binary)[0] - 1

    safe = np.maximum(total, 1)
    return {
        "boundary_fraction": [float(v) for v in edge / safe],
        "mean_component_pixels": [float(v) for v in total / np.maximum(components, 1)],
        "pixel_fraction": [float(v) for v in total / max(total.sum(), 1)],
        "split": split,
        "dilation": dilation,
    }


def geometry_vs_frequency(report: dict[str, Any]) -> dict[str, float] | None:
    """Correlate per-class IoU against thinness and against rarity, and compare.

    Both are reported because the comparison is the point: whichever explains more of the
    per-class spread is the one worth acting on. With only ``n_classes`` points these are
    weak estimates, and the collinearity between the two predictors is returned alongside
    so the correlations are not read as independent.
    """
    geometry = report.get("class_geometry")
    iou = report.get("iou_per_class")
    if not geometry or not iou:
        return None
    iou = np.asarray(iou, dtype=float)
    boundary = np.asarray(geometry["boundary_fraction"], dtype=float)
    frequency = np.asarray(geometry["pixel_fraction"], dtype=float)
    valid = ~np.isnan(iou) & (frequency > 0)
    if valid.sum() < 3:
        return None
    log_frequency = np.log10(frequency[valid])
    return {
        "n_classes": int(valid.sum()),
        "r_iou_vs_boundary_fraction": float(np.corrcoef(boundary[valid], iou[valid])[0, 1]),
        "r_iou_vs_log_frequency": float(np.corrcoef(log_frequency, iou[valid])[0, 1]),
        "r_boundary_vs_log_frequency": float(np.corrcoef(boundary[valid], log_frequency)[0, 1]),
    }


@torch.no_grad()
def analyse(
    checkpoint: str | Path,
    data_root: str | Path,
    split: str = "val",
    device: str = "auto",
    batch_size: int = 8,
    worst_k: int = 8,
) -> dict[str, Any]:
    """Run the model over a split and collect everything the report needs."""
    from evaluation.eval import load_checkpoint
    from modeling.config import resolve_device

    device_t = resolve_device(device)
    model, payload = load_checkpoint(checkpoint, device_t)
    class_names = payload["class_names"]
    n_classes = len(class_names)
    imgsz = tuple(payload.get("imgsz", (224, 320)))

    dataset = IDDSegmentation(data_root, split, imgsz, train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    confusion = ConfusionMatrix(n_classes)
    per_image: list[dict[str, Any]] = []
    index = 0

    for images, targets in loader:
        preds = model(images.to(device_t)).argmax(dim=1).cpu().numpy()
        target_array = targets.numpy()
        confusion.update(torch.from_numpy(preds), targets)
        for i in range(preds.shape[0]):
            per_image.append(
                {
                    "index": index,
                    "name": dataset.images[index].stem,
                    "drive": dataset.drives[index],
                    "miou": per_image_miou(preds[i], target_array[i], n_classes),
                }
            )
            index += 1

    mious = np.array([r["miou"] for r in per_image])
    order = np.argsort(mious)
    support = confusion.matrix.sum(axis=1)

    normalised = confusion.normalised()
    # Strongest off-diagonal confusions, as "true class was predicted as other class".
    pairs = []
    for true_index in range(n_classes):
        for pred_index in range(n_classes):
            if true_index != pred_index and support[true_index] > 0:
                pairs.append(
                    {
                        "true": class_names[true_index],
                        "predicted": class_names[pred_index],
                        "rate": float(normalised[true_index, pred_index]),
                        "pixels": int(confusion.matrix[true_index, pred_index]),
                    }
                )
    pairs.sort(key=lambda p: p["rate"], reverse=True)

    return {
        "checkpoint": str(checkpoint),
        "data": Path(data_root).name,
        "split": split,
        "class_names": class_names,
        "confusion_matrix": confusion.matrix.tolist(),
        "iou_per_class": [float(v) for v in confusion.iou_per_class()],
        "support_pixels": [int(v) for v in support],
        "support_fraction": [float(v / max(1, support.sum())) for v in support],
        "miou": confusion.miou(),
        "top_confusions": pairs[:10],
        "per_image": per_image,
        "worst_indices": [int(i) for i in order[:worst_k]],
        "best_indices": [int(i) for i in order[-worst_k:][::-1]],
        "miou_percentiles": {
            str(p): float(np.percentile(mious, p)) for p in (5, 25, 50, 75, 95)
        },
        "class_geometry": class_geometry(data_root, n_classes),
        "_dataset": dataset,
        "_model": model,
        "_device": device_t,
    }


def plot_confusion(report: dict[str, Any], out_path: str | Path) -> Path:
    """Render the row-normalised confusion matrix."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.array(report["confusion_matrix"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalised = np.where(matrix.sum(1, keepdims=True) > 0,
                              matrix / matrix.sum(1, keepdims=True), 0.0)
    names = report["class_names"]

    fig, ax = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
    image = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title(f"Row-normalised confusion  (mIoU {report['miou']:.3f})")
    for i in range(len(names)):
        for j in range(len(names)):
            value = normalised[i, j]
            if value > 0.005:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if value > 0.5 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_iou_vs_frequency(report: dict[str, Any], out_path: str | Path) -> Path:
    """Scatter per-class IoU against class frequency, to test whether errors track rarity."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = report["class_names"]
    frequency = np.array(report["support_fraction"])
    iou = np.array(report["iou_per_class"], dtype=float)
    valid = ~np.isnan(iou) & (frequency > 0)

    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    ax.scatter(frequency[valid] * 100, iou[valid], s=70, c="#4C72B0", zorder=3)
    for name, f, value in zip(
        np.array(names)[valid], frequency[valid], iou[valid], strict=True
    ):
        ax.annotate(name, (f * 100, value), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    if valid.sum() > 2:
        correlation = np.corrcoef(np.log10(frequency[valid]), iou[valid])[0, 1]
        ax.set_title(f"Per-class IoU vs frequency   (Pearson r = {correlation:.2f} on log freq)")
    ax.set_xscale("log")
    # Matplotlib's default log formatter renders "2 x 10^0" for percentages, which is
    # unreadable here; the range is narrow enough for plain decimal labels.
    ticks = [1, 2, 5, 10, 20, 40]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}%" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("share of labelled validation pixels")
    ax.set_ylabel("IoU")
    ax.grid(alpha=0.3, zorder=0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


@torch.no_grad()
def plot_worst_cases(report: dict[str, Any], out_path: str | Path, k: int = 6) -> Path:
    """Render the worst frames as image / ground truth / prediction / error map."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset = report["_dataset"]
    model = report["_model"]
    device = report["_device"]
    indices = report["worst_indices"][:k]

    fig, axes = plt.subplots(len(indices), 4, figsize=(13, 2.5 * len(indices)),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    for row, index in enumerate(indices):
        image_tensor, target = dataset[index]
        pred = model(image_tensor[None].to(device)).argmax(1)[0].cpu().numpy()
        target = target.numpy()
        rgb = denormalise(image_tensor)
        error = (pred != target) & (target != IGNORE_INDEX)

        record = report["per_image"][index]
        panels = [
            (rgb, f"{record['name']}\nmIoU {record['miou']:.3f}"),
            (colourise(target), "ground truth"),
            (colourise(pred), "prediction"),
            (np.stack([error * 255] * 3, -1).astype(np.uint8),
             f"errors ({100 * error.mean():.1f}% of pixels)"),
        ]
        for column, (panel, title) in enumerate(panels):
            axes[row, column].imshow(panel)
            axes[row, column].set_title(title, fontsize=8)
            axes[row, column].axis("off")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed/level1_official")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--worst-k", type=int, default=8)
    parser.add_argument("--out", default="results/error_analysis.json")
    parser.add_argument("--figure-dir", default="results/figures")
    args = parser.parse_args()

    report = analyse(args.checkpoint, args.data, args.split, args.device, worst_k=args.worst_k)
    report["geometry_vs_frequency"] = geometry_vs_frequency(report)
    figure_dir = Path(args.figure_dir)

    plot_confusion(report, figure_dir / "confusion_matrix.png")
    plot_iou_vs_frequency(report, figure_dir / "iou_vs_frequency.png")
    plot_worst_cases(report, figure_dir / "worst_cases.png")

    serialisable = {k: v for k, v in report.items() if not k.startswith("_")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(serialisable, indent=2))

    print(f"mIoU {report['miou']:.4f}   per-image mIoU percentiles: {report['miou_percentiles']}")
    print("\nTop confusions (true -> predicted):")
    for pair in report["top_confusions"][:6]:
        print(f"  {pair['true']:24s} -> {pair['predicted']:24s} {100 * pair['rate']:5.1f}%")
    print(f"\nfigures -> {figure_dir}/")


if __name__ == "__main__":
    main()
