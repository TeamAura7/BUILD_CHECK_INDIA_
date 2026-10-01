from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


MeasurementSource = Literal[
    "NATIVE_TEXT", "OCR", "VECTOR_GEOMETRY", "DERIVED", "VISION_DXF_RENDER", "VECTOR_GEOMETRY_RECONSTRUCTED"
]


class IndependentMeasurement(BaseModel):
    field: str
    # Length dimensions keep the historical value_m field.  Area/coverage/FAR
    # validation uses the generic numeric value + unit pair below.
    value_m: Optional[float] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    source: Optional[MeasurementSource] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    note: Optional[str] = None
    page: Optional[int] = None
    geometry_bbox_pts: Optional[list[float]] = None


class IndependentCVResult(BaseModel):
    document_id: str
    pages_analyzed: list[int] = Field(default_factory=list)
    measurements: list[IndependentMeasurement] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    site_plan_page: Optional[int] = None
    site_plan_bbox_pts: Optional[list[float]] = None
    scale_points_per_metre: Optional[float] = None
    scale_confidence: Optional[float] = None
    # Explicit outcome of the independent CV/native/OCR site-plan pipeline
    # itself (not of any individual field within it). "OK" means the
    # pipeline ran to completion (it may still have zero measurements on a
    # sheet with no resolvable site plan). "FAILED" means it raised and
    # produced no measurements at all -- this must never be silently
    # conflated with "OK, but nothing found", since the final-fusion layer
    # needs to know the difference to decide whether to trust a lone
    # pre-fusion Vision reading. "NOT_RUN" is for callers that never
    # attempted independent CV validation at all (e.g. a Vision-only tool).
    status: Literal["OK", "FAILED", "NOT_RUN"] = "OK"
    error: Optional[str] = None
