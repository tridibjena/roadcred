"""Pydantic response models for the RoadSense API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassShare(BaseModel):
    """How much of a frame one class occupies."""

    name: str = Field(..., description="Class name")
    index: int = Field(..., description="Class index in the model's output")
    pixel_fraction: float = Field(..., description="Share of predicted pixels, 0-1")
    mean_confidence: float = Field(..., description="Mean calibrated confidence over those pixels")
    colour: list[int] = Field(..., description="RGB colour used in the overlay")


class PredictResponse(BaseModel):
    """Result of segmenting one uploaded image."""

    width: int
    height: int
    class_names: list[str]
    classes: list[ClassShare]
    mean_confidence: float = Field(..., description="Calibrated mean confidence over all pixels")
    low_confidence_fraction: float = Field(
        ..., description="Share of pixels below the confidence threshold"
    )
    temperature: float = Field(..., description="Calibration temperature applied to logits")
    calibrated: bool = Field(..., description="False when no calibration report was found")
    inference_ms: float
    mask_png: str = Field(..., description="Base64 PNG of the colourised class mask")
    overlay_png: str = Field(..., description="Base64 PNG of the mask blended over the input")
    confidence_png: str = Field(..., description="Base64 PNG heatmap of per-pixel confidence")


class ModelInfo(BaseModel):
    """What the server currently has loaded."""

    loaded: bool
    architecture: str | None = None
    encoder: str | None = None
    class_names: list[str] = []
    imgsz: list[int] = []
    precision: str | None = None
    val_miou: float | None = None
    temperature: float | None = None
    model_path: str | None = None


class HealthResponse(BaseModel):
    status: str
    model: ModelInfo
