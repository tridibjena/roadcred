"""FastAPI service exposing the trained segmentation model and the results tables.

Endpoints:

* ``GET  /health``   -- liveness plus what model is loaded
* ``POST /predict``  -- image in, colourised mask + calibrated confidence out
* ``GET  /results``  -- every ``results/*.csv`` as JSON, for the frontend's charts
* ``GET  /results/{name}`` -- a single table
* ``GET  /figures/{name}`` -- a generated PNG figure

Inference runs through ONNX Runtime rather than PyTorch: it is the artefact the
compression work actually produces, so the served model is the one that was benchmarked.

Confidence is temperature-scaled using the factor fitted in
``results/calibration_report.json`` when present. Temperature scaling is monotone, so it
never changes the predicted class -- only the confidence attached to it.
"""

from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from serving.schemas import ClassShare, HealthResponse, ModelInfo, PredictResponse

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODEL_CANDIDATES = ("model_int8.onnx", "model_fp32.onnx")

#: Same palette the offline error analysis uses, so figures and the UI agree.
CLASS_COLOURS = np.array(
    [
        [128, 64, 128], [244, 35, 232], [220, 20, 60], [0, 0, 230],
        [190, 153, 153], [107, 142, 35], [70, 130, 180],
    ],
    dtype=np.uint8,
)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LOW_CONFIDENCE_THRESHOLD = 0.6

app = FastAPI(
    title="RoadSense API",
    description="7-class semantic segmentation of unstructured Indian road scenes (IDD).",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    # Local Vite dev server. A deployed instance would pin its real origin here.
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelBundle:
    """Lazily loaded ONNX session plus its metadata and calibration temperature."""

    def __init__(self) -> None:
        self.session: Any = None
        self.meta: dict[str, Any] = {}
        self.temperature: float = 1.0
        self.calibrated: bool = False
        self.path: Path | None = None

    def load(self) -> bool:
        """Find and load a model. Returns whether one was available."""
        import onnxruntime as ort

        for name in MODEL_CANDIDATES:
            candidate = REPO_ROOT / "checkpoints" / name
            if candidate.exists():
                self.path = candidate
                break
        else:
            return False

        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            str(self.path), options, providers=["CPUExecutionProvider"]
        )

        sidecar = (REPO_ROOT / "checkpoints" / "model_fp32.json")
        self.meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}

        report = RESULTS_DIR / "calibration_report.json"
        if report.exists():
            payload = json.loads(report.read_text())
            self.temperature = float(payload.get("temperature", 1.0))
            self.calibrated = True
        return True

    @property
    def class_names(self) -> list[str]:
        return self.meta.get("class_names", [f"class{i}" for i in range(7)])

    @property
    def imgsz(self) -> tuple[int, int]:
        return tuple(self.meta.get("imgsz", (224, 320)))  # type: ignore[return-value]

    def info(self) -> ModelInfo:
        return ModelInfo(
            loaded=self.session is not None,
            architecture=self.meta.get("architecture"),
            encoder=self.meta.get("encoder"),
            class_names=self.class_names if self.session else [],
            imgsz=list(self.imgsz) if self.session else [],
            precision="int8" if self.path and "int8" in self.path.name else "fp32",
            val_miou=self.meta.get("val_miou"),
            temperature=self.temperature,
            model_path=self.path.name if self.path else None,
        )


bundle = ModelBundle()


@app.on_event("startup")
def _startup() -> None:
    if not bundle.load():
        print("No ONNX model found. Run: python -m compression.quantize --checkpoint <ckpt>")


def _encode_png(array: np.ndarray) -> str:
    """Encode an RGB or grayscale array as a base64 PNG data payload."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _preprocess(image: np.ndarray, imgsz: tuple[int, int]) -> np.ndarray:
    """Resize to the model's resolution and apply ImageNet normalisation."""
    import cv2

    resized = cv2.resize(image, (imgsz[1], imgsz[0]), interpolation=cv2.INTER_LINEAR)
    normalised = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return normalised.transpose(2, 0, 1)[None]


def _softmax(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=axis, keepdims=True)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check and a description of the loaded model."""
    return HealthResponse(status="ok" if bundle.session else "no_model", model=bundle.info())


@app.post("/predict", response_model=PredictResponse)
async def predict(image: UploadFile = File(...)) -> PredictResponse:
    """Segment an uploaded image.

    Returns the colourised mask, an overlay, a confidence heatmap, and per-class shares.
    Confidence is temperature-calibrated when a calibration report is available.
    """
    import cv2

    if bundle.session is None and not bundle.load():
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Run: python -m compression.quantize --checkpoint <ckpt>",
        )

    payload = await image.read()
    decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]

    started = time.perf_counter()
    logits = bundle.session.run(None, {"input": _preprocess(rgb, bundle.imgsz)})[0]
    inference_ms = (time.perf_counter() - started) * 1000

    probabilities = _softmax(logits / bundle.temperature)[0]
    prediction = probabilities.argmax(axis=0).astype(np.uint8)
    confidence = probabilities.max(axis=0)

    # Report at the caller's resolution, not the model's.
    prediction = cv2.resize(prediction, (width, height), interpolation=cv2.INTER_NEAREST)
    confidence = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_LINEAR)

    names = bundle.class_names
    colour_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for index in range(len(names)):
        colour_mask[prediction == index] = CLASS_COLOURS[index % len(CLASS_COLOURS)]

    overlay = (rgb.astype(np.float32) * 0.55 + colour_mask.astype(np.float32) * 0.45).astype(np.uint8)
    heatmap = cv2.applyColorMap((confidence * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    total = prediction.size
    classes = [
        ClassShare(
            name=name,
            index=index,
            pixel_fraction=float((prediction == index).sum() / total),
            mean_confidence=float(confidence[prediction == index].mean())
            if (prediction == index).any()
            else 0.0,
            colour=CLASS_COLOURS[index % len(CLASS_COLOURS)].tolist(),
        )
        for index, name in enumerate(names)
    ]

    return PredictResponse(
        width=width,
        height=height,
        class_names=names,
        classes=classes,
        mean_confidence=float(confidence.mean()),
        low_confidence_fraction=float((confidence < LOW_CONFIDENCE_THRESHOLD).mean()),
        temperature=bundle.temperature,
        calibrated=bundle.calibrated,
        inference_ms=inference_ms,
        mask_png=_encode_png(colour_mask),
        overlay_png=_encode_png(overlay),
        confidence_png=_encode_png(heatmap),
    )


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """Read a results CSV, coercing numeric strings so charts get numbers not strings."""
    import csv

    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if value in (None, ""):
                    row[key] = None
                    continue
                try:
                    row[key] = float(value) if "." in value or "e" in value.lower() else int(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
    return rows


@app.get("/results")
def results() -> dict[str, Any]:
    """Every results table as JSON, keyed by filename stem."""
    if not RESULTS_DIR.exists():
        return {"tables": {}, "figures": []}
    tables = {path.stem: _read_csv(path) for path in sorted(RESULTS_DIR.glob("*.csv"))}
    reports = {
        path.stem: json.loads(path.read_text()) for path in sorted(RESULTS_DIR.glob("*.json"))
    }
    figures = [p.name for p in sorted(FIGURES_DIR.glob("*.png"))] if FIGURES_DIR.exists() else []
    return {"tables": tables, "reports": reports, "figures": figures}


@app.get("/results/{name}")
def result_table(name: str) -> list[dict[str, Any]]:
    """A single results table by filename stem."""
    path = RESULTS_DIR / f"{name}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No results table named {name!r}")
    return _read_csv(path)


@app.get("/figures/{name}")
def figure(name: str) -> FileResponse:
    """Serve a generated figure by filename."""
    path = (FIGURES_DIR / name).resolve()
    # Prevent path traversal out of the figures directory.
    if not str(path).startswith(str(FIGURES_DIR.resolve())) or not path.exists():
        raise HTTPException(status_code=404, detail=f"No figure named {name!r}")
    return FileResponse(path, media_type="image/png")
