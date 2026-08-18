"""Evaluate a trained checkpoint: mIoU, per-class IoU, boundary IoU, optional TTA.

Run::

    python -m evaluation.eval --checkpoint checkpoints/loss_ce_s0.pt
    python -m evaluation.eval --checkpoint ... --tta --data data/processed/level1_sequence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.boundary_iou import BoundaryIoU
from evaluation.metrics import ConfusionMatrix
from modeling.architectures import build_model, count_parameters
from modeling.dataset import IDDSegmentation


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild a model from a checkpoint written by :func:`modeling.train.train`.

    The checkpoint stores the architecture, encoder and class names alongside the weights,
    so evaluation never needs to be told what it is loading.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    model = build_model(
        payload["architecture"],
        payload["encoder"],
        payload["n_classes"],
        encoder_weights=None,  # weights come from the checkpoint, not from ImageNet
    )
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), payload


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    images: torch.Tensor,
    tta: bool = False,
    scales: tuple[float, ...] = (0.75, 1.0, 1.25),
) -> torch.Tensor:
    """Predict logits, optionally averaging over test-time augmentations.

    TTA averages *softmax probabilities* rather than logits: logits from different scales
    are not on a common scale, so averaging them lets the most confident branch dominate
    for reasons unrelated to correctness.

    Args:
        model: A model in eval mode.
        images: ``(N, 3, H, W)`` normalised input.
        tta: Whether to apply horizontal-flip and multi-scale augmentation.
        scales: Scale factors used when ``tta`` is set.

    Returns:
        ``(N, C, H, W)`` logits, or averaged probabilities when ``tta`` is set.
    """
    if not tta:
        return model(images)

    height, width = images.shape[-2:]
    accumulated = None
    count = 0
    for scale in scales:
        size = (int(round(height * scale / 32)) * 32, int(round(width * scale / 32)) * 32)
        scaled = F.interpolate(images, size=size, mode="bilinear", align_corners=False)
        for flip in (False, True):
            batch = torch.flip(scaled, dims=[3]) if flip else scaled
            logits = model(batch)
            if flip:
                logits = torch.flip(logits, dims=[3])
            logits = F.interpolate(
                logits, size=(height, width), mode="bilinear", align_corners=False
            )
            probs = logits.softmax(dim=1)
            accumulated = probs if accumulated is None else accumulated + probs
            count += 1
    return accumulated / count


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: str | Path,
    data_root: str | Path,
    split: str = "val",
    device: str = "auto",
    batch_size: int = 8,
    tta: bool = False,
    boundary_dilation: int = 2,
) -> dict[str, Any]:
    """Evaluate a checkpoint on a prepared variant and return all metrics."""
    from torch.utils.data import DataLoader

    from modeling.config import resolve_device

    device_t = resolve_device(device)
    model, payload = load_checkpoint(checkpoint, device_t)
    class_names = payload.get("class_names") or [f"class{i}" for i in range(payload["n_classes"])]
    imgsz = tuple(payload.get("imgsz", (224, 320)))

    dataset = IDDSegmentation(data_root, split, imgsz, train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    confusion = ConfusionMatrix(len(class_names))
    boundary = BoundaryIoU(len(class_names), boundary_dilation)

    for images, targets in loader:
        images = images.to(device_t)
        logits = predict_logits(model, images, tta)
        pred = logits.argmax(dim=1).cpu().numpy()
        target = targets.numpy()
        confusion.update(torch.from_numpy(pred), torch.from_numpy(target))
        boundary.update(pred, target)

    result: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "data": Path(data_root).name,
        "split": split,
        "tta": tta,
        "n_images": len(dataset),
        "params": count_parameters(model),
        **confusion.summary(class_names),
        **boundary.summary(class_names),
    }
    result["boundary_gap"] = result["miou"] - result["boundary_miou"]
    result["confusion_matrix"] = confusion.matrix.tolist()
    result["class_names"] = class_names
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed/level1_official")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tta", action="store_true", help="Flip + multi-scale averaging")
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    result = evaluate_checkpoint(
        args.checkpoint, args.data, args.split, args.device, args.batch_size, args.tta
    )
    printable = {k: v for k, v in result.items() if k != "confusion_matrix"}
    print(json.dumps(printable, indent=2, default=float))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
