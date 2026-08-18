"""Accuracy / latency / size Pareto across model variants and precisions.

Takes a set of trained checkpoints — typically {teacher, standalone student, distilled
student} — exports each to ONNX, quantizes each to INT8, and benchmarks every combination
under identical conditions. The output answers the question a deployment actually asks:
for a given latency budget, what is the most accurate model available?

All timing is CPU with a pinned thread count. Apple silicon has no INT8 inference path
through MPS, so a GPU measurement would silently run the quantized graph in FP32 and
report a speedup that does not exist.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def evaluate_variant(
    checkpoint: str | Path,
    data_root: str | Path,
    label: str,
    out_dir: str | Path,
    threads: int = 4,
    n_calibration: int = 64,
) -> list[dict[str, Any]]:
    """Export, quantize and benchmark one checkpoint at both precisions."""
    from compression.export import check_parity, export_onnx
    from compression.quantize import benchmark_latency, evaluate_onnx, quantize_model

    out_dir = Path(out_dir)
    fp32 = export_onnx(checkpoint, out_dir / f"{label}_fp32.onnx")
    meta = json.loads(fp32.with_suffix(".json").read_text())
    imgsz = tuple(meta["imgsz"])
    n_classes = len(meta["class_names"])

    parity = check_parity(checkpoint, fp32, data_root)
    if not parity["passed"]:
        print(f"  WARNING: {label} failed ONNX parity "
              f"(label disagreement {parity['label_disagreement_fraction']:.2e})", flush=True)

    int8 = quantize_model(fp32, out_dir / f"{label}_int8.onnx", data_root, imgsz, n_calibration)

    rows: list[dict[str, Any]] = []
    for precision, path in (("fp32", fp32), ("int8", int8)):
        miou = evaluate_onnx(path, data_root, imgsz, n_classes, threads=threads)
        latency = benchmark_latency(path, imgsz, threads=threads)
        rows.append({
            "variant": label,
            "precision": precision,
            "miou": miou,
            "size_mb": path.stat().st_size / 1e6,
            "parity_passed": parity["passed"],
            **latency,
        })
        print(f"  {label:>20s} {precision:>5s}  mIoU={miou:.4f}  "
              f"{latency['latency_ms_mean']:6.1f} ms  {rows[-1]['size_mb']:6.1f} MB", flush=True)
    return rows


def pareto_frontier(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the rows that are not dominated on both latency and accuracy.

    A model is dominated when some other model is at least as fast *and* at least as
    accurate. Only the survivors represent a real deployment choice; everything else is
    strictly worse than an available alternative.
    """
    frontier = []
    for row in rows:
        dominated = any(
            other is not row
            and other["latency_ms_mean"] <= row["latency_ms_mean"]
            and other["miou"] >= row["miou"]
            and (other["latency_ms_mean"] < row["latency_ms_mean"] or other["miou"] > row["miou"])
            for other in rows
        )
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda r: r["latency_ms_mean"])


def plot_pareto(rows: Sequence[dict[str, Any]], out_path: str | Path) -> Path:
    """Scatter latency against mIoU, sized by model size, with the frontier traced."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    for index, variant in enumerate(variants):
        for precision, marker in (("fp32", "o"), ("int8", "s")):
            subset = [r for r in rows if r["variant"] == variant and r["precision"] == precision]
            if not subset:
                continue
            ax.scatter(
                [r["latency_ms_mean"] for r in subset],
                [r["miou"] for r in subset],
                s=[max(40, r["size_mb"] * 6) for r in subset],
                marker=marker,
                color=colours[index % len(colours)],
                alpha=0.85,
                edgecolors="white",
                linewidths=1.2,
                label=f"{variant} · {precision}",
                zorder=3,
            )

    frontier = pareto_frontier(rows)
    if len(frontier) > 1:
        ax.plot(
            [r["latency_ms_mean"] for r in frontier],
            [r["miou"] for r in frontier],
            "--", color="#333", lw=1.2, zorder=2, label="Pareto frontier",
        )

    for row in rows:
        ax.annotate(
            f"{row['variant']}/{row['precision']}",
            (row["latency_ms_mean"], row["miou"]),
            textcoords="offset points", xytext=(7, -3), fontsize=7, color="#444",
        )

    ax.set_xlabel("CPU latency (ms, batch 1, 4 threads)")
    ax.set_ylabel("validation mIoU")
    ax.set_title("Accuracy / latency / size trade-off   (marker area ∝ model size)")
    ax.grid(alpha=0.3, zorder=0)
    ax.legend(fontsize=8, loc="lower right")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="One or more 'label=path/to.pt' pairs, or bare paths",
    )
    parser.add_argument("--data", default="data/processed/level1_official")
    parser.add_argument("--out-dir", default="checkpoints/pareto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default="results/compression_pareto.csv")
    parser.add_argument("--figure", default="results/figures/compression_pareto.png")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for entry in args.checkpoints:
        label, _, path = entry.partition("=")
        if not path:
            label, path = Path(label).stem, label
        print(f"\n=== {label} ===", flush=True)
        rows += evaluate_variant(path, args.data, label, args.out_dir, args.threads)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["variant", "precision", "miou", "latency_ms_mean", "latency_ms_p50",
              "latency_ms_p95", "throughput_img_s", "size_mb", "parity_passed",
              "threads", "batch_size"]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    figure = plot_pareto(rows, args.figure)
    frontier = pareto_frontier(rows)
    print(f"\nPareto-optimal configurations ({len(frontier)} of {len(rows)}):")
    for row in frontier:
        print(f"  {row['variant']:>20s} {row['precision']:>5s}  "
              f"mIoU={row['miou']:.4f}  {row['latency_ms_mean']:.1f} ms  {row['size_mb']:.1f} MB")
    print(f"\n-> {out}\n-> {figure}")


if __name__ == "__main__":
    main()
