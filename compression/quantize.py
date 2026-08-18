"""INT8 post-training quantization and CPU latency benchmarking.

Quantization is done with ONNX Runtime rather than torch.ao because Apple silicon has no
INT8 inference path through MPS -- the GPU would silently run the model in FP32 and the
"quantized" latency would be meaningless. Running both precisions on **CPU** keeps the
comparison honest: same backend, same threads, same input, only the precision differs.

Static (calibrated) quantization is used rather than dynamic. Dynamic quantization mainly
accelerates MatMul-heavy graphs; a convolutional segmentation network needs calibrated
activation ranges to quantize its convolutions at all.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np


class ImageCalibrationReader:
    """Feeds real validation frames to the quantizer to estimate activation ranges.

    Calibrating on random noise produces ranges that no real image occupies, which is a
    common cause of large accuracy drops that get blamed on INT8 itself.
    """

    def __init__(self, data_root: str | Path, imgsz: tuple[int, int], n_samples: int = 64):
        from modeling.dataset import IDDSegmentation

        dataset = IDDSegmentation(data_root, "val", imgsz, train=False)
        count = min(n_samples, len(dataset))
        # Spread the calibration set across the split rather than taking a contiguous
        # prefix, which would over-represent a handful of drive sequences.
        stride = max(1, len(dataset) // count)
        self.samples = [dataset[i][0].numpy()[None] for i in range(0, stride * count, stride)][:count]
        self._iterator: Iterator[dict[str, np.ndarray]] | None = None

    def get_next(self) -> dict[str, np.ndarray] | None:
        """ONNX Runtime calibration protocol."""
        if self._iterator is None:
            self._iterator = iter({"input": s} for s in self.samples)
        return next(self._iterator, None)

    def rewind(self) -> None:
        self._iterator = None


def quantize_model(
    onnx_path: str | Path,
    out_path: str | Path,
    data_root: str | Path,
    imgsz: tuple[int, int],
    n_calibration: int = 64,
) -> Path:
    """Statically quantize an ONNX model to INT8.

    Args:
        onnx_path: FP32 source model.
        out_path: Destination for the INT8 model.
        data_root: Prepared variant used for calibration.
        imgsz: Model input resolution.
        n_calibration: Frames used to estimate activation ranges.

    Returns:
        The written path.
    """
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    onnx_path, out_path = Path(onnx_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prepared = out_path.with_name(out_path.stem + "_prepared.onnx")

    # Shape inference and graph cleanup first; quantizing an un-preprocessed graph
    # silently skips nodes whose shapes are unknown.
    quant_pre_process(str(onnx_path), str(prepared), skip_symbolic_shape=False)

    quantize_static(
        str(prepared),
        str(out_path),
        ImageCalibrationReader(data_root, imgsz, n_calibration),
        quant_format=QuantFormat.QDQ,
        # Per-channel weight scales matter for convolutions, whose output channels have
        # very different dynamic ranges; per-tensor scaling costs real accuracy here.
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    prepared.unlink(missing_ok=True)
    return out_path


def benchmark_latency(
    onnx_path: str | Path,
    imgsz: tuple[int, int],
    batch_size: int = 1,
    warmup: int = 5,
    iterations: int = 30,
    threads: int = 4,
) -> dict[str, float]:
    """Measure single-stream CPU latency.

    Thread count is pinned so FP32 and INT8 are measured under identical conditions --
    ORT otherwise picks different defaults per model and the comparison drifts.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path), options, providers=["CPUExecutionProvider"]
    )

    rng = np.random.default_rng(0)
    batch = rng.standard_normal((batch_size, 3, *imgsz)).astype(np.float32)

    for _ in range(warmup):
        session.run(None, {"input": batch})

    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(None, {"input": batch})
        timings.append((time.perf_counter() - start) * 1000)

    timings_array = np.array(timings)
    return {
        "latency_ms_mean": float(timings_array.mean()),
        "latency_ms_p50": float(np.percentile(timings_array, 50)),
        "latency_ms_p95": float(np.percentile(timings_array, 95)),
        "throughput_img_s": float(batch_size * 1000 / timings_array.mean()),
        "threads": threads,
        "batch_size": batch_size,
    }


def evaluate_onnx(
    onnx_path: str | Path,
    data_root: str | Path,
    imgsz: tuple[int, int],
    n_classes: int,
    batch_size: int = 8,
    threads: int = 4,
) -> float:
    """mIoU of an ONNX model over the validation split."""
    import onnxruntime as ort
    import torch
    from torch.utils.data import DataLoader

    from evaluation.metrics import ConfusionMatrix
    from modeling.dataset import IDDSegmentation

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])

    dataset = IDDSegmentation(data_root, "val", imgsz, train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    confusion = ConfusionMatrix(n_classes)

    for images, targets in loader:
        logits = session.run(None, {"input": images.numpy()})[0]
        confusion.update(torch.from_numpy(logits), targets)
    return confusion.miou()


def run(
    checkpoint: str | Path,
    data_root: str | Path,
    out_dir: str | Path = "checkpoints",
    n_calibration: int = 64,
    threads: int = 4,
) -> list[dict[str, Any]]:
    """Export FP32, quantize to INT8, and compare accuracy, latency and size."""
    from compression.export import check_parity, export_onnx

    out_dir = Path(out_dir)

    fp32 = export_onnx(checkpoint, out_dir / "model_fp32.onnx")
    meta = json.loads(fp32.with_suffix(".json").read_text())
    imgsz = tuple(meta["imgsz"])
    n_classes = len(meta["class_names"])

    parity = check_parity(checkpoint, fp32, data_root)
    print(f"ONNX parity: {'PASS' if parity['passed'] else 'FAIL'} "
          f"(label disagreement {parity['label_disagreement_fraction']:.2e})", flush=True)

    int8 = quantize_model(fp32, out_dir / "model_int8.onnx", data_root, imgsz, n_calibration)

    rows: list[dict[str, Any]] = []
    for name, path in (("fp32", fp32), ("int8", int8)):
        miou = evaluate_onnx(path, data_root, imgsz, n_classes, threads=threads)
        latency = benchmark_latency(path, imgsz, threads=threads)
        rows.append(
            {
                "precision": name,
                "miou": miou,
                "size_mb": path.stat().st_size / 1e6,
                **latency,
            }
        )
        print(f"{name}: mIoU={miou:.4f}  {latency['latency_ms_mean']:.1f}ms  "
              f"{path.stat().st_size / 1e6:.1f}MB", flush=True)

    if len(rows) == 2:
        rows[1]["miou_delta"] = rows[1]["miou"] - rows[0]["miou"]
        rows[1]["speedup"] = rows[0]["latency_ms_mean"] / rows[1]["latency_ms_mean"]
        rows[1]["size_reduction"] = rows[0]["size_mb"] / rows[1]["size_mb"]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed/level1_official")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--calibration", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default="results/compression.csv")
    args = parser.parse_args()

    rows = run(args.checkpoint, args.data, args.out_dir, args.calibration, args.threads)

    import csv

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["precision"] + [f for f in fields if f != "precision"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
