"""Does the confidence score survive the conditions it was not calibrated on?

:mod:`evaluation.calibration` fits a temperature on clean validation pixels and reports a
well-calibrated model. :mod:`evaluation.corruption_eval` shows accuracy collapsing under
corruption and recovering under BatchNorm adaptation. Nothing connects the two -- and the
question sitting between them is the one a deployed system actually depends on:

**The API ships a confidence number calibrated in clear weather. Is it still a probability
in the rain?**

Three measurements, in the order they matter:

1. **Calibration under shift.** Take the temperature fitted on *clean* data -- the one the
   server actually applies, since a deployed model cannot refit against conditions it has
   not seen -- and measure its calibration error on corrupted data. A refit-on-shift
   temperature is reported alongside as the unreachable ceiling: the gap between them is
   the cost of calibrating in the lab and deploying in the world.

2. **Does BatchNorm adaptation restore confidence, or only accuracy?** Test-time BN
   adaptation recovers a large fraction of the mIoU lost to corruption. Whether it also
   repairs the confidence distribution is a separate question with no obvious answer: it
   could hand back an accurate model that is still lying about its certainty.

3. **Selective prediction.** If confidence means something, thresholding on it should buy
   accuracy. The risk-coverage curve answers what a perception stack asks: *at the
   threshold where this model is 95% accurate, how much of the frame can it commit to?*
   That also replaces the hardcoded 0.6 in the serving layer with a measured operating
   point.

Run::

    python -m evaluation.shift_calibration --checkpoint checkpoints/<best>.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from evaluation.calibration import (
    TemperatureScaler,
    collect_predictions,
    expected_calibration_error,
)
from evaluation.corruption_eval import adapt_batchnorm, build_corrupted_loader
from evaluation.corruptions import WEATHER_LIKE

IGNORE_INDEX = 255


def _confidence_and_correctness(
    logits: torch.Tensor, targets: torch.Tensor, temperature: float
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated max-softmax confidence and whether each pixel was right."""
    probs = (logits / temperature).softmax(dim=1)
    confidence, prediction = probs.max(dim=1)
    return confidence.numpy(), (prediction == targets).numpy()


def risk_coverage(
    confidence: np.ndarray, correct: np.ndarray, targets: tuple[float, ...] = (0.95, 0.99)
) -> dict[str, Any]:
    """Selective-prediction curve: accuracy as a function of how much is abstained on.

    Pixels are ranked by confidence and progressively dropped from the least confident
    upward. ``coverage`` is the fraction retained; ``accuracy`` is measured on exactly
    that retained set.

    A confidence score that carries no information produces a flat curve -- dropping the
    least confident pixels would not raise accuracy on the rest. The area under this curve
    is therefore a direct test of whether the number means anything, independent of
    whether it is calibrated to the right scale.

    Args:
        confidence: Per-pixel confidence.
        correct: Per-pixel correctness.
        targets: Accuracy levels to report achievable coverage for.

    Returns:
        The curve, plus the coverage still available at each target accuracy
        (``nan`` where the target is unreachable at any threshold).
    """
    order = np.argsort(-confidence)  # most confident first
    ranked = correct[order].astype(np.float64)
    # Accuracy over the top-k most confident pixels, for every k.
    running = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    coverage = np.arange(1, ranked.size + 1) / ranked.size

    out: dict[str, Any] = {
        "auc": float(np.trapezoid(running, coverage)),
        "accuracy_at_full_coverage": float(running[-1]),
    }
    for target in targets:
        reachable = np.flatnonzero(running >= target)
        # The largest coverage whose retained set still meets the target.
        out[f"coverage_at_acc_{target}"] = (
            float(coverage[reachable[-1]]) if reachable.size else float("nan")
        )
        out[f"threshold_at_acc_{target}"] = (
            float(confidence[order][reachable[-1]]) if reachable.size else float("nan")
        )
    # Sample the curve sparsely for plotting rather than storing millions of points.
    picks = np.linspace(0, ranked.size - 1, 200).astype(int)
    out["curve_coverage"] = [float(v) for v in coverage[picks]]
    out["curve_accuracy"] = [float(v) for v in running[picks]]
    return out


def per_class_calibration(
    logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    class_names: list[str],
    n_bins: int = 15,
) -> list[dict[str, Any]]:
    """Calibration error broken out by *predicted* class.

    Grouped by prediction rather than by ground truth, because that is the only grouping
    available at inference time -- a deployed system knows what it predicted, not what was
    true. Reveals whether the classes the model is worst at are also the ones it is most
    overconfident about, which is the combination that matters for a safety case.
    """
    probs = (logits / temperature).softmax(dim=1)
    confidence, prediction = probs.max(dim=1)
    correct = (prediction == targets).numpy()
    confidence = confidence.numpy()
    prediction = prediction.numpy()

    rows: list[dict[str, Any]] = []
    for index, name in enumerate(class_names):
        selected = prediction == index
        if selected.sum() < n_bins * 10:  # too few pixels for a stable bin estimate
            continue
        ece, _ = expected_calibration_error(confidence[selected], correct[selected], n_bins)
        rows.append(
            {
                "class": name,
                "predicted_pixels": int(selected.sum()),
                "accuracy": float(correct[selected].mean()),
                "mean_confidence": float(confidence[selected].mean()),
                "ece": ece,
                # Positive => the model claims more certainty than it earns.
                "overconfidence": float(confidence[selected].mean() - correct[selected].mean()),
            }
        )
    return rows


@torch.no_grad()
def _collect(
    model: nn.Module, loader: Any, device: torch.device, max_pixels: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return collect_predictions(model, loader, device, max_pixels=max_pixels, seed=seed)


def run(
    checkpoint: str | Path,
    data_root: str | Path,
    corruptions: tuple[str, ...] = WEATHER_LIKE,
    severities: tuple[int, ...] = (1, 3),
    device: str = "auto",
    batch_size: int = 8,
    max_pixels: int = 2_000_000,
    seed: int = 0,
    adapt: bool = True,
) -> dict[str, Any]:
    """Measure calibration on clean and corrupted data, with and without BN adaptation."""
    from evaluation.eval import load_checkpoint
    from modeling.config import resolve_device

    device_t = resolve_device(device)
    model, payload = load_checkpoint(checkpoint, device_t)
    class_names = payload["class_names"]
    imgsz = tuple(payload.get("imgsz", (224, 320)))

    # 1. Fit the deployment temperature on clean data, exactly as the server does.
    clean_loader = build_corrupted_loader(data_root, imgsz, None, 1, batch_size)
    clean_logits, clean_targets = _collect(model, clean_loader, device_t, max_pixels, seed)

    half = clean_targets.numel() // 2
    order = torch.randperm(clean_targets.numel(), generator=torch.Generator().manual_seed(seed))
    fit_idx, eval_idx = order[:half], order[half:]
    scaler = TemperatureScaler()
    deployed_t = scaler.fit(clean_logits[fit_idx], clean_targets[fit_idx])
    print(f"deployment temperature (fitted on clean) T = {deployed_t:.4f}", flush=True)

    rows: list[dict[str, Any]] = []

    def measure(name: str, severity: int, logits: torch.Tensor, targets: torch.Tensor,
                adapted: bool) -> dict[str, Any]:
        confidence, correct = _confidence_and_correctness(logits, targets, deployed_t)
        ece_deployed, _ = expected_calibration_error(confidence, correct)
        # The ceiling: what calibration would look like if we could refit on this shift.
        oracle = TemperatureScaler()
        oracle_t = oracle.fit(logits, targets)
        oracle_conf, oracle_correct = _confidence_and_correctness(logits, targets, oracle_t)
        ece_oracle, _ = expected_calibration_error(oracle_conf, oracle_correct)
        coverage = risk_coverage(confidence, correct)
        return {
            "corruption": name,
            "severity": severity,
            "bn_adapted": adapted,
            "accuracy": float(correct.mean()),
            "mean_confidence": float(confidence.mean()),
            "overconfidence": float(confidence.mean() - correct.mean()),
            "ece_deployed_t": ece_deployed,
            "ece_refit_t": ece_oracle,
            "refit_temperature": oracle_t,
            "calibration_gap": ece_deployed - ece_oracle,
            "coverage_at_acc_0.95": coverage["coverage_at_acc_0.95"],
            "risk_coverage_auc": coverage["auc"],
            "_curve": coverage,
        }

    clean_row = measure("clean", 0, clean_logits[eval_idx], clean_targets[eval_idx], False)
    rows.append(clean_row)
    print(f"  clean            ECE={clean_row['ece_deployed_t']:.4f}  "
          f"acc={clean_row['accuracy']:.4f}", flush=True)

    per_class = per_class_calibration(
        clean_logits[eval_idx], clean_targets[eval_idx], deployed_t, class_names
    )

    for name in corruptions:
        for severity in severities:
            loader = build_corrupted_loader(data_root, imgsz, name, severity, batch_size)
            logits, targets = _collect(model, loader, device_t, max_pixels, seed)
            row = measure(name, severity, logits, targets, False)
            rows.append(row)
            print(f"  {name:16s} s{severity}  ECE={row['ece_deployed_t']:.4f} "
                  f"(refit {row['ece_refit_t']:.4f})  acc={row['accuracy']:.4f}", flush=True)

            if adapt:
                adapted_model = adapt_batchnorm(model, loader, device_t)
                a_logits, a_targets = _collect(adapted_model, loader, device_t, max_pixels, seed)
                a_row = measure(name, severity, a_logits, a_targets, True)
                rows.append(a_row)
                print(f"  {name:16s} s{severity}  +BN ECE={a_row['ece_deployed_t']:.4f}  "
                      f"acc={a_row['accuracy']:.4f}", flush=True)
                del adapted_model

    return {
        "checkpoint": str(checkpoint),
        "deployed_temperature": deployed_t,
        "class_names": class_names,
        "rows": rows,
        "per_class_clean": per_class,
    }


def _write_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "corruption", "severity", "bn_adapted", "accuracy", "mean_confidence",
        "overconfidence", "ece_deployed_t", "ece_refit_t", "refit_temperature",
        "calibration_gap", "coverage_at_acc_0.95", "risk_coverage_auc",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_shift_calibration(report: dict[str, Any], out_path: str | Path) -> Path:
    """Two panels: how calibration degrades with shift, and the risk-coverage curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in report["rows"] if not r["bn_adapted"]]
    adapted = {(r["corruption"], r["severity"]): r for r in report["rows"] if r["bn_adapted"]}

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5))

    labels = [f"{r['corruption']}\ns{r['severity']}" if r["severity"] else "clean" for r in rows]
    x = np.arange(len(rows))
    left.bar(x - 0.2, [r["ece_deployed_t"] for r in rows], 0.4,
             label="clean-fitted T (deployed)", color="#c1121f")
    left.bar(x + 0.2, [r["ece_refit_t"] for r in rows], 0.4,
             label="refit on shift (ceiling)", color="#457b9d")
    have_adapted = [adapted.get((r["corruption"], r["severity"])) for r in rows]
    if any(have_adapted):
        left.plot(
            [i for i, a in enumerate(have_adapted) if a],
            [a["ece_deployed_t"] for a in have_adapted if a],
            "o--", color="#2a9d8f", label="after BN adaptation",
        )
    left.set_xticks(x)
    left.set_xticklabels(labels, fontsize=7)
    left.set_ylabel("expected calibration error")
    left.set_title("Confidence was calibrated on clean data.\nDoes it survive the shift?")
    left.legend(fontsize=8)
    left.grid(axis="y", alpha=0.3)

    for row in rows:
        curve = row["_curve"]
        label = "clean" if not row["severity"] else f"{row['corruption']} s{row['severity']}"
        right.plot(curve["curve_coverage"], curve["curve_accuracy"], label=label, linewidth=1.2)
    right.axhline(0.95, color="grey", linestyle=":", linewidth=1)
    right.set_xlabel("coverage (fraction of pixels the model commits to)")
    right.set_ylabel("accuracy on retained pixels")
    right.set_title("Selective prediction:\nhow much can it commit to, and be right?")
    right.legend(fontsize=7)
    right.grid(alpha=0.3)

    figure.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed/level1_official")
    parser.add_argument("--corruptions", nargs="+", default=list(WEATHER_LIKE))
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=2_000_000)
    parser.add_argument("--no-adapt", action="store_true")
    parser.add_argument("--out", default="results/shift_calibration.csv")
    parser.add_argument("--json-out", default="results/shift_calibration.json")
    parser.add_argument("--figure", default="results/figures/shift_calibration.png")
    args = parser.parse_args()

    report = run(
        args.checkpoint, args.data, tuple(args.corruptions), tuple(args.severities),
        args.device, args.batch_size, args.max_pixels, adapt=not args.no_adapt,
    )
    csv_path = _write_csv(report["rows"], args.out)
    figure = plot_shift_calibration(report, args.figure)

    serialisable = {
        **{k: v for k, v in report.items() if k != "rows"},
        "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in report["rows"]],
        "risk_coverage_clean": {
            k: v for k, v in report["rows"][0]["_curve"].items() if not k.startswith("curve_")
        },
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(serialisable, indent=2))

    clean = report["rows"][0]
    worst = max(report["rows"], key=lambda r: r["ece_deployed_t"])
    print(f"\nclean ECE {clean['ece_deployed_t']:.4f} -> worst "
          f"{worst['corruption']} s{worst['severity']} ECE {worst['ece_deployed_t']:.4f} "
          f"({worst['ece_deployed_t'] / max(clean['ece_deployed_t'], 1e-9):.1f}x)")
    print(f"-> {csv_path}\n-> {figure}")


if __name__ == "__main__":
    main()
