from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class VisionRegion(BaseModel):
    id: str
    type: str
    bbox: Optional[list[float]] = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    label: Optional[str] = None
    evidence: Optional[str] = None


class VisionDimension(BaseModel):
    value: Optional[float] = None
    unit: Optional[str] = None
    type: str = "UNKNOWN"
    region_id: Optional[str] = None
    bbox: Optional[list[float]] = None
    evidence: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    # Set by BaseArchitecturalPlanExtractor._is_item_grounded when grounding
    # runs: True if this value's numeric text was actually found in the
    # PDF's native text layer, False if grounding ran but there was no
    # native text layer at all to check against (e.g. a scanned/raster
    # page), None if grounding did not run (e.g. ground_against_native_text
    # =False, or a value-less item). This is independent of the model's own
    # self-reported `confidence` -- it exists so a confidence-laundering
    # check downstream can tell "the model says it's sure" apart from "this
    # was actually confirmed against evidence the model didn't produce".
    grounded: Optional[bool] = None


class VisionArea(BaseModel):
    value: Optional[float] = None
    unit: Optional[str] = None
    type: str = "UNKNOWN"
    region_id: Optional[str] = None
    evidence: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    # See VisionDimension.grounded above -- identical meaning/provenance.
    grounded: Optional[bool] = None


class VisionPageResult(BaseModel):
    page_number: int
    units: Optional[str] = None
    scale: Optional[str] = None
    regions: list[VisionRegion] = Field(default_factory=list)
    dimensions: list[VisionDimension] = Field(default_factory=list)
    areas: list[VisionArea] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    raw_response: Optional[str] = None
    # The actual PDF page size (in points, rotated-display space -- same
    # frame as `page.get_drawings()`/CV geometry) that this page's images
    # and bboxes were derived from. Populated from the rendered PNG's own
    # pixel dimensions (reliable for scanned pages with no text layer too),
    # NOT reconstructed from vision_render_dpi at consumption time -- this
    # is what lets `vision_extraction.spatial.vision_bbox_to_page_points`
    # correctly detect whether a model's bbox is in raw image pixels or on
    # a normalized 0-1000 grid (see that function's docstring). 0.0 means
    # unknown (e.g. old cached vision_result.json predating this field);
    # consumers must treat 0.0 as "can't convert precisely" rather than
    # dividing by zero.
    page_width_pts: float = 0.0
    page_height_pts: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class VisionDocumentResult(BaseModel):
    pages: list[VisionPageResult] = Field(default_factory=list)
    model_name: str
    enabled: bool = True
    warnings: list[str] = Field(default_factory=list)
