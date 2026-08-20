"""Pydantic response models for the RoadCred API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassShare(BaseModel):
    """How much of a frame one class occupies."""

    name: str = Field(..., description="Class name")
    index: int = Field(..., description="Class index in the model's output")
    pixel_fraction: float = Field(..., description="Share of predicted pixels, 0-1")
    mean_confidence: float = Field(..., description="Mean calibrated confidence over those pixels")
    colour: list[int] = Field(..., description="RGB colour used in the overlay")


class Drivability(BaseModel):
    """Forward-corridor summary for one frame.

    A description of the segmentation, not a safety assessment. Components are exposed
    alongside the score so a low value can be attributed: a corridor blocked by a vehicle
    and one the model is simply unsure about are different situations with the same score.
    """

    score: float = Field(..., description="coverage x mean_confidence x (1 - obstruction)")
    coverage: float = Field(..., description="Share of the corridor predicted drivable")
    mean_confidence: float = Field(..., description="Mean confidence over those drivable pixels")
    obstruction: float = Field(..., description="Share of corridor occupied by vehicles or people")
    boundary_fraction: float = Field(
        ..., description="Share of corridor adjacent to a class border, where the model is least reliable"
    )
    low_confidence: float = Field(..., description="Share of corridor below the confidence threshold")
    corridor_pixels: int
    road_edge_caveat: bool = Field(
        ...,
        description="Corridor spans a drivable/non-drivable border; the model absorbs "
        "17-20% of true non-drivable into drivable, so the score is optimistic here",
    )


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
    drivability: Drivability
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
