"""Does the drivable-path caveat fire when the score really is optimistic?

The serving layer's drivable-path score counts every corridor pixel predicted drivable.
This model absorbs ~20% of true non-drivable into drivable, so wherever that absorption
lands inside the corridor the score is optimistic -- and ``road_edge_caveat`` exists to
say so. That is a claim about the flag, so it is measured against ground truth here:

* an **absorption event** is a frame in which true non-drivable pixels inside the
  corridor are predicted drivable;
* **recall** is the share of those frames, and of those absorbed pixels, the flag fires
  on;
* the **firing rate** is how often it fires at all -- a flag that always fires is noise.

It runs the served artefact -- the ONNX model, the deployed temperature, the API's own
preprocessing and :func:`serving.drivability.drivability` itself -- at each frame's native
resolution, so it measures exactly what the endpoint returns. The rule it replaced, which
fired only when the corridor held *predicted* non-drivable, is scored alongside, together
with the margin sweep the shipped margin was chosen from.

Validation is both where the margin was chosen and where recall is reported, so the
recall figure is optimistic by construction; the sweep is published so that is visible.

Run::

    python -m evaluation.road_edge_caveat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

#: Road-edge margins, in pixels, swept to choose the shipped one.
SWEEP_PX = (0, 1, 2, 3, 4, 6)


def _previous_rule(prediction: np.ndarray, region: np.ndarray) -> bool:
    """The rule this replaced: predicted drivable and predicted non-drivable both in the
    corridor. Total absorption leaves no predicted non-drivable, so it cannot fire."""
    from serving.drivability import DRIVABLE, NON_DRIVABLE

    in_corridor = prediction[region]
    return bool((in_corridor == DRIVABLE).any() and (in_corridor == NON_DRIVABLE).any())


def frame_records(
    onnx_path: str | Path, data_root: str | Path, split: str, temperature: float
) -> list[dict[str, Any]]:
    """Score every frame of a split exactly as ``POST /predict`` would."""
    import cv2
    import onnxruntime as ort

    from serving.api import _preprocess, _softmax
    from serving.drivability import (
        DRIVABLE,
        NON_DRIVABLE,
        corridor_mask,
        drivability,
        road_edge_near_corridor,
    )

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    meta_path = Path(onnx_path).with_name("model_fp32.json")
    imgsz = tuple(json.loads(meta_path.read_text())["imgsz"]) if meta_path.exists() else (224, 320)

    root = Path(data_root)
    records: list[dict[str, Any]] = []
    for image_path in sorted((root / "images" / split).glob("*.jpg")):
        rgb = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        truth = cv2.imread(str(root / "masks" / split / f"{image_path.stem}.png"),
                           cv2.IMREAD_UNCHANGED)
        height, width = rgb.shape[:2]

        logits = session.run(None, {"input": _preprocess(rgb, imgsz)})[0]
        probabilities = _softmax(logits / temperature)[0]
        prediction = cv2.resize(probabilities.argmax(axis=0).astype(np.uint8), (width, height),
                                interpolation=cv2.INTER_NEAREST)
        confidence = cv2.resize(probabilities.max(axis=0), (width, height),
                                interpolation=cv2.INTER_LINEAR)

        region = corridor_mask(height, width)
        absorbed = int((region & (truth == NON_DRIVABLE) & (prediction == DRIVABLE)).sum())
        records.append({
            "frame": image_path.stem,
            "corridor_pixels": int(region.sum()),
            "absorbed_pixels": absorbed,
            "absorbed_fraction": absorbed / max(1, int(region.sum())),
            "shipped": drivability(prediction, confidence)["road_edge_caveat"],
            "previous": _previous_rule(prediction, region),
            **{f"margin_{m}px": road_edge_near_corridor(prediction, region, m) for m in SWEEP_PX},
        })
    return records


def score_rule(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Recall over absorption events, absorbed-pixel recall, and firing rate."""
    fires = np.array([r[key] for r in records], dtype=bool)
    absorbed = np.array([r["absorbed_pixels"] for r in records])
    fraction = np.array([r["absorbed_fraction"] for r in records])
    any_event, big_event = absorbed > 0, fraction >= 0.01
    return {
        "frames_caught_any": int((fires & any_event).sum()),
        "frames_caught_1pct": int((fires & big_event).sum()),
        "absorbed_pixel_recall": float(absorbed[fires].sum() / absorbed.sum())
        if absorbed.sum() else None,
        "firing_rate": float(fires.mean()),
        "frames_fired": int(fires.sum()),
    }


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    absorbed = np.array([r["absorbed_pixels"] for r in records])
    fraction = np.array([r["absorbed_fraction"] for r in records])
    return {
        "n_frames": len(records),
        "events_any": int((absorbed > 0).sum()),
        "events_1pct": int((fraction >= 0.01).sum()),
        "rules": {key: score_rule(records, key) for key in ("shipped", "previous")},
        "margin_sweep": [
            {"margin_px": m, **score_rule(records, f"margin_{m}px")} for m in SWEEP_PX
        ],
    }


def main() -> None:
    from serving.drivability import ROAD_EDGE_MARGIN

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", default="checkpoints/model_int8.onnx")
    parser.add_argument("--data", default="data/processed/level1_official")
    parser.add_argument("--split", default="val")
    parser.add_argument("--calibration", default="results/calibration_report.json")
    parser.add_argument("--out", default="results/road_edge_caveat.json")
    args = parser.parse_args()

    calibration = Path(args.calibration)
    temperature = (
        float(json.loads(calibration.read_text())["temperature"]) if calibration.exists() else 1.0
    )
    records = frame_records(args.onnx, args.data, args.split, temperature)
    report = {
        "onnx": args.onnx,
        "split": args.split,
        "temperature": temperature,
        "shipped_margin_fraction_of_width": ROAD_EDGE_MARGIN,
        **summarise(records),
        "frames": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    print(f"{report['n_frames']} frames; absorption events: {report['events_any']} "
          f"(>=1% of corridor: {report['events_1pct']})")
    for name, rule in report["rules"].items():
        print(f"  {name:8s} caught {rule['frames_caught_any']}/{report['events_any']} "
              f"({rule['frames_caught_1pct']}/{report['events_1pct']} at >=1%), "
              f"absorbed-pixel recall {rule['absorbed_pixel_recall']:.3f}, "
              f"fires on {100 * rule['firing_rate']:.0f}% of frames")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
