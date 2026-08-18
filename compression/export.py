"""ONNX export with a numerical parity check against PyTorch.

Exporting and assuming it worked is a standard way to ship a subtly broken model: an
unsupported op silently falls back, a dynamic axis is baked to a constant, or a
normalisation constant is dropped, and the served model quietly differs from the one that
was evaluated. :func:`check_parity` compares the two on real inputs and reports both the
logit error and, more importantly, the fraction of pixels whose predicted class changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def export_onnx(
    checkpoint: str | Path,
    out_path: str | Path,
    imgsz: tuple[int, int] | None = None,
    opset: int = 18,
    dynamic_batch: bool = True,
    standalone: bool = True,
) -> Path:
    """Export a trained checkpoint to ONNX.

    Args:
        checkpoint: Path to a checkpoint from :func:`modeling.train.train`.
        out_path: Destination ``.onnx`` file.
        imgsz: Override the checkpoint's training resolution.
        opset: ONNX opset version. Defaults to 18, which is what torch's dynamo exporter
            emits natively for this graph; requesting a lower opset makes onnxscript
            attempt a version downgrade that fails on this model's Squeeze/Unsqueeze
            nodes and falls back anyway, producing an alarming traceback for no benefit.
        dynamic_batch: Mark the batch dimension dynamic so the server can batch requests.
        standalone: Consolidate weights into the ``.onnx`` file itself. torch.onnx.export
            writes tensors to a sibling ``.onnx.data`` file by default, which is easy to
            lose when copying a model to a server; a single file cannot be half-deployed.

    Returns:
        The written path.
    """
    from evaluation.eval import load_checkpoint

    model, payload = load_checkpoint(checkpoint, "cpu")
    model.eval()
    height, width = imgsz or tuple(payload.get("imgsz", (224, 320)))
    dummy = torch.randn(1, 3, height, width)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}} if dynamic_batch else None,
        do_constant_folding=True,
    )

    if standalone:
        _inline_external_data(out_path)
    # Persist the metadata a server needs but ONNX does not carry.
    out_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "class_names": payload["class_names"],
                "imgsz": [height, width],
                "architecture": payload["architecture"],
                "encoder": payload["encoder"],
                "source_checkpoint": str(checkpoint),
                "val_miou": payload.get("miou"),
            },
            indent=2,
        )
    )
    return out_path


def _inline_external_data(path: Path) -> None:
    """Fold a sibling ``.onnx.data`` file back into the model, leaving one file.

    Skipped above the 2 GB protobuf limit, where external data is mandatory rather than
    merely inconvenient.
    """
    import onnx

    sidecar = path.with_suffix(path.suffix + ".data")
    if not sidecar.exists():
        return
    if sidecar.stat().st_size > 1_800_000_000:
        return
    model = onnx.load(str(path))  # resolves external data relative to the file
    onnx.save_model(model, str(path), save_as_external_data=False)
    sidecar.unlink(missing_ok=True)


def check_parity(
    checkpoint: str | Path,
    onnx_path: str | Path,
    data_root: str | Path | None = None,
    n_samples: int = 16,
    logit_tolerance: float = 1e-3,
    label_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Compare PyTorch and ONNX Runtime outputs on real inputs.

    Random noise is a weak test -- it exercises none of the activation statistics a real
    image produces -- so real validation frames are used when available.

    Args:
        checkpoint: The PyTorch checkpoint.
        onnx_path: The exported ONNX model.
        data_root: Prepared variant to draw frames from; falls back to random input.
        n_samples: How many frames to compare.
        logit_tolerance: Max acceptable mean absolute logit difference.
        label_tolerance: Max acceptable fraction of pixels whose argmax changes.

    Returns:
        Parity metrics plus a boolean ``passed``.
    """
    import onnxruntime as ort

    from evaluation.eval import load_checkpoint

    model, payload = load_checkpoint(checkpoint, "cpu")
    height, width = tuple(payload.get("imgsz", (224, 320)))

    if data_root is not None and Path(data_root).exists():
        from modeling.dataset import IDDSegmentation

        dataset = IDDSegmentation(data_root, "val", (height, width), train=False)
        batch = torch.stack([dataset[i][0] for i in range(min(n_samples, len(dataset)))])
        source = "real validation frames"
    else:
        batch = torch.randn(n_samples, 3, height, width)
        source = "random input (no dataset found)"

    with torch.no_grad():
        torch_logits = model(batch).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(["logits"], {"input": batch.numpy()})[0]

    absolute = np.abs(torch_logits - onnx_logits)
    torch_labels = torch_logits.argmax(axis=1)
    onnx_labels = onnx_logits.argmax(axis=1)
    disagreement = float((torch_labels != onnx_labels).mean())

    result = {
        "source": source,
        "n_samples": int(batch.shape[0]),
        "mean_abs_logit_diff": float(absolute.mean()),
        "max_abs_logit_diff": float(absolute.max()),
        "label_disagreement_fraction": disagreement,
        "logit_tolerance": logit_tolerance,
        "label_tolerance": label_tolerance,
    }
    result["passed"] = bool(
        result["mean_abs_logit_diff"] < logit_tolerance and disagreement < label_tolerance
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="checkpoints/model.onnx")
    parser.add_argument("--data", default="data/processed/level1_official")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    path = export_onnx(args.checkpoint, args.out, opset=args.opset)
    size_mb = path.stat().st_size / 1e6
    print(f"exported -> {path}  ({size_mb:.1f} MB)")

    parity = check_parity(args.checkpoint, path, args.data)
    print(json.dumps(parity, indent=2))
    if not parity["passed"]:
        raise SystemExit("ONNX parity check FAILED - the exported model differs from PyTorch")
    print("parity OK")


if __name__ == "__main__":
    main()
