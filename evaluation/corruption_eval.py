"""Robustness under test-time corruption, with test-time BatchNorm adaptation and FGSM.

Three questions, in order of how much they matter for a deployed model:

1. **How far does accuracy fall under adverse conditions?** Measured as mIoU per
   corruption per severity, and summarised by mean corruption error.
2. **How much of that is recoverable for free?** Test-time BatchNorm adaptation
   (:func:`adapt_batchnorm`) re-estimates BN statistics on the shifted data using no
   labels and no gradient steps. It is the cheapest domain-adaptation baseline there is,
   and it isolates how much of the degradation is mere feature-statistic drift.
3. **How does it behave under adversarial rather than natural shift?** A single-step FGSM
   perturbation gives the worst-case counterpart to the natural corruptions.
"""

from __future__ import annotations

import argparse
import copy
import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.corruptions import CORRUPTIONS, WEATHER_LIKE, make_corruption
from evaluation.metrics import ConfusionMatrix
from modeling.dataset import IMAGENET_MEAN, IMAGENET_STD, IDDSegmentation


def build_corrupted_loader(
    data_root: str | Path,
    imgsz: tuple[int, int],
    corruption: str | None,
    severity: int,
    batch_size: int = 8,
    split: str = "val",
) -> DataLoader:
    """Validation loader with an optional test-time corruption applied."""
    hook = make_corruption(corruption, severity) if corruption else None
    dataset = IDDSegmentation(data_root, split, imgsz, train=False, corruption=hook)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def adapt_batchnorm(model: nn.Module, loader: DataLoader, device: torch.device) -> nn.Module:
    """Re-estimate BatchNorm running statistics on the (unlabelled) target data.

    Under distribution shift, a network's learned affine parameters usually remain valid
    while its BN running mean/variance -- estimated on the source distribution -- no
    longer describe the activations. Recomputing them on the target data requires no
    labels, no gradients and one forward pass over the set.

    Only BN layers are put in training mode; every other layer stays in eval, so dropout
    and the like are unaffected. ``momentum = None`` makes each layer accumulate a
    cumulative moving average over the whole pass rather than an exponential one that
    would over-weight the final batches.

    Returns:
        A deep copy of ``model`` with adapted statistics. The original is untouched, so
        adapted and unadapted numbers come from the same weights.
    """
    adapted = copy.deepcopy(model).to(device)
    adapted.eval()

    bn_layers = [m for m in adapted.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bn_layers:
        return adapted
    for layer in bn_layers:
        layer.reset_running_stats()
        layer.momentum = None
        layer.train()

    for images, _ in loader:
        adapted(images.to(device))

    adapted.eval()
    return adapted


@torch.no_grad()
def measure(model: nn.Module, loader: DataLoader, device: torch.device, n_classes: int) -> float:
    """Mean IoU of ``model`` over ``loader``."""
    confusion = ConfusionMatrix(n_classes)
    for images, targets in loader:
        confusion.update(model(images.to(device)).cpu(), targets)
    return confusion.miou()


def fgsm_attack(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
    epsilon: float = 2 / 255,
) -> float:
    """mIoU under a single-step FGSM perturbation.

    ``epsilon`` is specified in *pixel* space and converted into the normalised space the
    model consumes, so the perturbation budget means the same thing regardless of the
    normalisation constants.
    """
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    step = epsilon / std  # pixel-space epsilon expressed in normalised units

    confusion = ConfusionMatrix(n_classes)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    model.eval()

    for images, targets in loader:
        images = images.to(device).requires_grad_(True)
        targets = targets.to(device)
        loss = criterion(model(images), targets)
        model.zero_grad(set_to_none=True)
        loss.backward()

        perturbed = images + step * images.grad.sign()
        # Keep the result a valid image: clamp in pixel space, then re-normalise.
        pixels = (perturbed * std + mean).clamp(0.0, 1.0)
        perturbed = ((pixels - mean) / std).detach()

        with torch.no_grad():
            confusion.update(model(perturbed).cpu(), targets.cpu())
    return confusion.miou()


def run(
    checkpoint: str | Path,
    data_root: str | Path,
    corruptions: Iterable[str],
    severities: Iterable[int],
    device: str = "auto",
    batch_size: int = 8,
    adapt: bool = True,
    out_csv: str | Path = "results/corruption_robustness.csv",
) -> list[dict[str, Any]]:
    """Evaluate clean, corrupted, and BN-adapted mIoU across corruptions and severities."""
    from evaluation.eval import load_checkpoint
    from modeling.config import resolve_device

    device_t = resolve_device(device)
    model, payload = load_checkpoint(checkpoint, device_t)
    n_classes = payload["n_classes"]
    imgsz = tuple(payload.get("imgsz", (224, 320)))

    clean_loader = build_corrupted_loader(data_root, imgsz, None, 1, batch_size)
    clean = measure(model, clean_loader, device_t, n_classes)
    print(f"clean mIoU = {clean:.4f}", flush=True)

    rows: list[dict[str, Any]] = [
        {"corruption": "clean", "severity": 0, "miou": clean, "miou_bn_adapted": clean,
         "retention": 1.0, "recovered": 0.0, "weather_like": False}
    ]

    for name in corruptions:
        for severity in severities:
            loader = build_corrupted_loader(data_root, imgsz, name, severity, batch_size)
            corrupted = measure(model, loader, device_t, n_classes)
            adapted_miou = float("nan")
            if adapt:
                adapted_model = adapt_batchnorm(model, loader, device_t)
                adapted_miou = measure(adapted_model, loader, device_t, n_classes)
                del adapted_model
            rows.append(
                {
                    "corruption": name,
                    "severity": severity,
                    "miou": corrupted,
                    "miou_bn_adapted": adapted_miou,
                    # Fraction of clean performance retained -- comparable across models.
                    "retention": corrupted / clean if clean else 0.0,
                    # mIoU points recovered by BN adaptation alone.
                    "recovered": (adapted_miou - corrupted) if adapt else 0.0,
                    "weather_like": name in WEATHER_LIKE,
                }
            )
            print(
                f"  {name:16s} s{severity}  mIoU={corrupted:.4f}"
                + (f"  +BN={adapted_miou:.4f} ({adapted_miou - corrupted:+.4f})" if adapt else ""),
                flush=True,
            )
            _write(rows, out_csv)
    return rows


def _write(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_degradation(csv_path: str | Path, out_path: str | Path) -> Path:
    """Plot mIoU against severity for each corruption, with the BN-adapted curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.read_csv(csv_path)
    clean = float(frame.loc[frame["corruption"] == "clean", "miou"].iloc[0])
    corrupted = frame[frame["corruption"] != "clean"]
    names = list(dict.fromkeys(corrupted["corruption"]))

    columns = 5
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 2.8 * rows),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names, strict=False):  # axes is padded to a full grid
        subset = corrupted[corrupted["corruption"] == name].sort_values("severity")
        ax.axhline(clean, color="#888", ls="--", lw=1, label="clean")
        ax.plot(subset["severity"], subset["miou"], "o-", color="#C44E52", ms=4, label="corrupted")
        if subset["miou_bn_adapted"].notna().any():
            ax.plot(subset["severity"], subset["miou_bn_adapted"], "s-",
                    color="#4C72B0", ms=4, label="+ BN adapt")
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0, max(0.05, clean * 1.15))
        # Severity is an integer level; matplotlib's default locator would otherwise
        # draw meaningless half-severities like 1.5.
        ax.set_xticks(sorted(subset["severity"].unique()))
        ax.grid(alpha=0.25)
    for ax in axes[len(names):]:
        ax.axis("off")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.supxlabel("severity")
    fig.supylabel("mIoU")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed/level1_official")
    parser.add_argument("--corruptions", nargs="+", default=sorted(CORRUPTIONS))
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-adapt", action="store_true", help="Skip BN adaptation")
    parser.add_argument("--fgsm", action="store_true", help="Also run an FGSM attack")
    parser.add_argument("--out", default="results/corruption_robustness.csv")
    parser.add_argument("--figure", default="results/figures/corruption_degradation.png")
    args = parser.parse_args()

    rows = run(
        args.checkpoint, args.data, args.corruptions, args.severities,
        args.device, args.batch_size, not args.no_adapt, args.out,
    )
    figure = plot_degradation(args.out, args.figure)
    print(f"\ndegradation curves -> {figure}")

    if args.fgsm:
        from evaluation.eval import load_checkpoint
        from modeling.config import resolve_device

        device = resolve_device(args.device)
        model, payload = load_checkpoint(args.checkpoint, device)
        loader = build_corrupted_loader(
            args.data, tuple(payload.get("imgsz", (224, 320))), None, 1, args.batch_size
        )
        clean = rows[0]["miou"]
        for epsilon in (1 / 255, 2 / 255, 4 / 255):
            miou = fgsm_attack(model, loader, device, payload["n_classes"], epsilon)
            print(f"FGSM eps={epsilon * 255:.0f}/255  mIoU={miou:.4f}  "
                  f"({100 * miou / clean:.1f}% of clean)")


if __name__ == "__main__":
    main()
