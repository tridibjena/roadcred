"""Prediction stability under small, label-preserving input perturbations.

Motivation: a deployed segmenter sees a near-identical scene many times a second, and a
model whose prediction flips between classes under imperceptible input changes is unusable
even at a good mIoU. The natural way to measure that is frame-to-frame flicker on video.

**IDD Lite cannot support that measurement.** Its frames are grouped by drive sequence,
but frames within a drive are not temporally adjacent: the median gap between consecutive
frames of a drive is ~4,400 frame indices, and only 0.5% of gaps are within 30 frames.
Treating them as consecutive would produce a "flicker" number that is really just scene
change.

This module measures the same underlying property honestly instead. Each image is
perturbed by a small, label-preserving transform -- a one or two pixel shift, a slight
brightness change, mild noise, light JPEG -- and the prediction is compared against the
unperturbed one. Geometric perturbations are inverted before comparison so that only
genuine label *flips* are counted, not the shift itself.

The reported quantity is the **flip rate**: the fraction of pixels whose predicted class
changes. Lower is better, and it is measured without any ground truth, so it can also be
run on unlabelled footage.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

IGNORE_INDEX = 255


def shift_image(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate an image by whole pixels, replicating the edge."""
    import cv2

    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE
    )


def _brightness(delta: float) -> Callable[[np.ndarray], np.ndarray]:
    def apply(image: np.ndarray) -> np.ndarray:
        return np.clip(image.astype(np.float32) * (1 + delta), 0, 255).astype(np.uint8)

    return apply


def _noise(sigma: float) -> Callable[[np.ndarray], np.ndarray]:
    def apply(image: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(0)
        return np.clip(
            image.astype(np.float32) + rng.normal(0, sigma * 255, image.shape), 0, 255
        ).astype(np.uint8)

    return apply


def _jpeg(quality: int) -> Callable[[np.ndarray], np.ndarray]:
    def apply(image: np.ndarray) -> np.ndarray:
        import cv2

        ok, buffer = cv2.imencode(".jpg", image[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return image if not ok else cv2.imdecode(buffer, cv2.IMREAD_COLOR)[:, :, ::-1]

    return apply


#: name -> (image transform, inverse applied to the *prediction*, valid-region cropper)
#: The inverse exists so a shift is not counted as a flip; the crop drops the replicated
#: border a shift creates, whose content is genuinely different rather than merely moved.
PERTURBATIONS: dict[str, tuple[Callable, Callable, int]] = {
    "shift_1px": (lambda im: shift_image(im, 1, 0), lambda p: shift_image(p, -1, 0), 2),
    "shift_2px": (lambda im: shift_image(im, 2, 2), lambda p: shift_image(p, -2, -2), 3),
    "brightness_5pct": (_brightness(0.05), lambda p: p, 0),
    "brightness_-5pct": (_brightness(-0.05), lambda p: p, 0),
    "noise_sigma0.01": (_noise(0.01), lambda p: p, 0),
    "jpeg_q90": (_jpeg(90), lambda p: p, 0),
}


@torch.no_grad()
def measure_stability(
    checkpoint: str | Path,
    data_root: str | Path,
    split: str = "val",
    device: str = "auto",
    batch_size: int = 8,
    perturbations: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Flip rate and confidence drift for each perturbation.

    Returns:
        One row per perturbation with ``flip_rate`` (fraction of pixels changing class),
        ``flip_rate_confident`` (restricted to pixels the model was confident about, which
        are the ones a downstream consumer would act on), and ``mean_confidence_delta``.
    """
    from evaluation.eval import load_checkpoint
    from modeling.dataset import IDDSegmentation
    from modeling.train import resolve_device

    device_t = resolve_device(device)
    model, payload = load_checkpoint(checkpoint, device_t)
    imgsz = tuple(payload.get("imgsz", (224, 320)))
    names = perturbations or list(PERTURBATIONS)

    base_dataset = IDDSegmentation(data_root, split, imgsz, train=False)
    base_loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Cache the unperturbed predictions once; every perturbation compares against them.
    base_predictions: list[np.ndarray] = []
    base_confidence: list[np.ndarray] = []
    for images, _ in base_loader:
        probs = model(images.to(device_t)).softmax(dim=1)
        base_predictions.append(probs.argmax(1).cpu().numpy())
        base_confidence.append(probs.max(1).values.cpu().numpy())
    base_pred = np.concatenate(base_predictions)
    base_conf = np.concatenate(base_confidence)

    rows: list[dict[str, Any]] = []
    for name in names:
        transform, inverse, crop = PERTURBATIONS[name]
        dataset = IDDSegmentation(data_root, split, imgsz, train=False, corruption=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        flips_total = 0
        pixels_total = 0
        flips_confident = 0
        pixels_confident = 0
        confidence_delta = 0.0
        seen = 0

        for index, (images, _) in enumerate(loader):
            probs = model(images.to(device_t)).softmax(dim=1)
            pred = probs.argmax(1).cpu().numpy()
            conf = probs.max(1).values.cpu().numpy()

            start = index * batch_size
            reference = base_pred[start : start + pred.shape[0]]
            reference_conf = base_conf[start : start + pred.shape[0]]

            for i in range(pred.shape[0]):
                aligned = inverse(pred[i].astype(np.uint8)).astype(np.int64)
                aligned_conf = inverse(conf[i])
                if crop:
                    sl = (slice(crop, -crop), slice(crop, -crop))
                    aligned, ref = aligned[sl], reference[i][sl]
                    aligned_conf, ref_conf = aligned_conf[sl], reference_conf[i][sl]
                else:
                    ref, ref_conf = reference[i], reference_conf[i]

                changed = aligned != ref
                flips_total += int(changed.sum())
                pixels_total += changed.size

                confident = ref_conf >= 0.8
                flips_confident += int((changed & confident).sum())
                pixels_confident += int(confident.sum())

                confidence_delta += float(np.abs(aligned_conf - ref_conf).mean())
                seen += 1

        rows.append(
            {
                "perturbation": name,
                "flip_rate": flips_total / max(1, pixels_total),
                "flip_rate_confident": flips_confident / max(1, pixels_confident),
                "mean_confidence_delta": confidence_delta / max(1, seen),
                "n_images": seen,
            }
        )
        print(
            f"  {name:18s} flip={rows[-1]['flip_rate'] * 100:5.2f}%  "
            f"confident-flip={rows[-1]['flip_rate_confident'] * 100:5.2f}%  "
            f"Δconf={rows[-1]['mean_confidence_delta']:.4f}",
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed/level1_official")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", default="results/stability.csv")
    args = parser.parse_args()

    rows = measure_stability(args.checkpoint, args.data, args.split, args.device, args.batch_size)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n-> {path}")
    print(json.dumps({r["perturbation"]: round(r["flip_rate"], 5) for r in rows}, indent=2))


if __name__ == "__main__":
    main()
