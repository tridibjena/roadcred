"""Confidence calibration: expected calibration error, reliability diagrams, temperature scaling.

A segmentation model's softmax maximum is routinely treated as a confidence score -- the
serving layer returns one -- but modern networks are systematically overconfident, so that
number is not a probability unless it has been checked. This module measures the gap and
corrects it.

**Temperature is fitted on a held-out half of the validation set and evaluated on the
other half.** Fitting and reporting on the same pixels would make the correction look
better than it is, which is the usual way calibration results get overstated.
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

IGNORE_INDEX = 255


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15
) -> tuple[float, dict[str, np.ndarray]]:
    r"""Expected calibration error with equal-width confidence bins.

    .. math:: \mathrm{ECE} = \sum_b \frac{|B_b|}{N} \bigl| \mathrm{acc}(B_b) - \mathrm{conf}(B_b) \bigr|

    Args:
        confidence: Predicted confidence per sample, in ``[0, 1]``.
        correct: Boolean array, whether each prediction was right.
        n_bins: Number of equal-width bins.

    Returns:
        ``(ece, bins)`` where ``bins`` holds per-bin ``accuracy``, ``confidence``,
        ``count`` and ``edges``, ready for a reliability diagram.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin by upper edge so that a confidence of exactly 1.0 lands in the last bin.
    index = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)

    accuracy = np.zeros(n_bins)
    mean_confidence = np.zeros(n_bins)
    count = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        selected = index == b
        count[b] = selected.sum()
        if count[b]:
            accuracy[b] = correct[selected].mean()
            mean_confidence[b] = confidence[selected].mean()

    total = max(1, count.sum())
    ece = float((count / total * np.abs(accuracy - mean_confidence)).sum())
    return ece, {
        "accuracy": accuracy,
        "confidence": mean_confidence,
        "count": count,
        "edges": edges,
    }


def maximum_calibration_error(bins: dict[str, np.ndarray]) -> float:
    """Largest gap between accuracy and confidence over non-empty bins."""
    populated = bins["count"] > 0
    if not populated.any():
        return 0.0
    return float(np.abs(bins["accuracy"] - bins["confidence"])[populated].max())


class TemperatureScaler(nn.Module):
    """Single-parameter post-hoc calibration: ``logits / T``.

    Temperature scaling cannot change which class is predicted -- dividing by a positive
    scalar is monotone -- so mIoU is provably unchanged and only the confidence
    distribution moves. That is what makes it safe to apply after model selection.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(float(np.log(temperature))))

    @property
    def temperature(self) -> float:
        """The fitted temperature. ``> 1`` means the model was overconfident."""
        return float(self.log_temperature.detach().exp())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_temperature.exp()

    def fit(self, logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 100) -> float:
        """Fit temperature by minimising NLL on held-out logits.

        Optimises ``log T`` rather than ``T`` so the temperature cannot go negative.
        """
        valid = targets != IGNORE_INDEX
        logits, targets = logits[valid], targets[valid]
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.05, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = F.cross_entropy(self(logits), targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.temperature


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    max_pixels: int = 4_000_000,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather logits and targets over a loader, subsampling pixels.

    A full validation split is hundreds of millions of pixels; calibration statistics
    converge long before that. Pixels are subsampled uniformly at random with a fixed
    seed so the estimate is reproducible and unbiased.
    """
    generator = torch.Generator().manual_seed(seed)
    logit_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    collected = 0

    for images, targets in loader:
        logits = model(images.to(device)).cpu()
        n_classes = logits.shape[1]
        flat_logits = logits.permute(0, 2, 3, 1).reshape(-1, n_classes)
        flat_targets = targets.reshape(-1)

        valid = flat_targets != IGNORE_INDEX
        flat_logits, flat_targets = flat_logits[valid], flat_targets[valid]
        if flat_targets.numel() == 0:
            continue

        budget = max_pixels - collected
        if flat_targets.numel() > budget:
            pick = torch.randperm(flat_targets.numel(), generator=generator)[:budget]
            flat_logits, flat_targets = flat_logits[pick], flat_targets[pick]

        logit_chunks.append(flat_logits)
        target_chunks.append(flat_targets)
        collected += flat_targets.numel()
        if collected >= max_pixels:
            break

    return torch.cat(logit_chunks), torch.cat(target_chunks)


def calibration_report(
    logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 15, seed: int = 0
) -> dict[str, Any]:
    """Measure calibration before and after temperature scaling, without leakage.

    Splits the supplied pixels in half: temperature is fitted on the first half and every
    reported number is computed on the second, which the fit never saw.

    Returns:
        Uncalibrated and calibrated ECE/MCE, the fitted temperature, and the bin contents
        for both reliability diagrams.
    """
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(targets.numel(), generator=generator)
    half = targets.numel() // 2
    fit_idx, eval_idx = order[:half], order[half:]

    scaler = TemperatureScaler()
    temperature = scaler.fit(logits[fit_idx], targets[fit_idx])

    eval_logits, eval_targets = logits[eval_idx], targets[eval_idx]

    def measure(values: torch.Tensor) -> tuple[float, dict[str, np.ndarray], float]:
        probs = values.softmax(dim=1)
        confidence, prediction = probs.max(dim=1)
        correct = (prediction == eval_targets).numpy()
        ece, bins = expected_calibration_error(confidence.numpy(), correct, n_bins)
        return ece, bins, maximum_calibration_error(bins)

    ece_before, bins_before, mce_before = measure(eval_logits)
    ece_after, bins_after, mce_after = measure(eval_logits / temperature)

    accuracy = float((eval_logits.argmax(dim=1) == eval_targets).float().mean())
    return {
        "temperature": temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "mce_before": mce_before,
        "mce_after": mce_after,
        "ece_reduction_pct": 100 * (1 - ece_after / ece_before) if ece_before else 0.0,
        "accuracy": accuracy,
        "n_fit_pixels": int(half),
        "n_eval_pixels": int(eval_targets.numel()),
        "bins_before": {k: v.tolist() for k, v in bins_before.items()},
        "bins_after": {k: v.tolist() for k, v in bins_after.items()},
    }


def plot_reliability(report: dict[str, Any], out_path: str | Path) -> Path:
    """Render before/after reliability diagrams."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax, key, label in (
        (axes[0], "bins_before", f"Uncalibrated (ECE {report['ece_before']:.4f})"),
        (axes[1], "bins_after", f"T = {report['temperature']:.3f} (ECE {report['ece_after']:.4f})"),
    ):
        bins = report[key]
        edges = np.asarray(bins["edges"])
        centres = (edges[:-1] + edges[1:]) / 2
        counts = np.asarray(bins["count"], dtype=float)
        accuracy = np.asarray(bins["accuracy"])
        width = edges[1] - edges[0]

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ax.bar(
            centres, accuracy, width=width * 0.9,
            color="#4C72B0", alpha=0.85, edgecolor="white", label="accuracy",
        )
        ax.plot(centres[counts > 0], np.asarray(bins["confidence"])[counts > 0],
                "o-", color="#DD8452", ms=4, lw=1.4, label="mean confidence")
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.set_title(label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=8)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed/level1_official")
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bins", type=int, default=15)
    parser.add_argument("--max-pixels", type=int, default=4_000_000)
    parser.add_argument("--out", default="results/calibration_report.json")
    parser.add_argument("--figure", default="results/figures/reliability.png")
    args = parser.parse_args()

    from torch.utils.data import DataLoader

    from evaluation.eval import load_checkpoint
    from modeling.dataset import IDDSegmentation
    from modeling.train import resolve_device

    device = resolve_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    dataset = IDDSegmentation(args.data, args.split, tuple(payload.get("imgsz", (224, 320))), train=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    logits, targets = collect_predictions(model, loader, device, args.max_pixels)
    report = calibration_report(logits, targets, args.bins)
    report["checkpoint"] = str(args.checkpoint)

    figure = plot_reliability(report, args.figure)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    print(json.dumps({k: v for k, v in report.items() if not k.startswith("bins_")}, indent=2))
    print(f"\nreliability diagram -> {figure}")


if __name__ == "__main__":
    main()
