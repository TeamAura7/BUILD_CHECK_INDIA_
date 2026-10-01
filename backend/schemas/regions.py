"""
Semantic drawing-region detection contract.

Additive to the schema surface -- nothing here is consumed by
`NormalizedPlan`/`ComplianceResult` or the RuleEngine. `DetectedRegion` is
an intermediate artifact used only to ground *where on the page* a Vision
focus-pass crop or a region-scoped OpenCV pass should look; it never carries
a measurement value itself.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from backend.schemas.geometry import BoundingBox

RegionType = Literal[
    "SITE_PLAN",
    "AREA_STATEMENT",
    "GROUND_FLOOR_PLAN",
    "FIRST_FLOOR_PLAN",
    "SECOND_FLOOR_PLAN",
    "THIRD_FLOOR_PLAN",
    "OTHER_FLOOR_PLAN",
    "ELEVATION",
    "SECTION",
]

RegionSource = Literal["caption_anchor", "vision"]


class DetectedRegion(BaseModel):
    """A page-space bounding box believed to contain one semantic drawing region.

    `source` records how the region was found: `caption_anchor` means it
    came from a deterministic search for the region's own printed caption
    ("SITE PLAN", "AREA STATEMENT", "ELEVATION", ...) plus geometric
    windowing around it -- independent of any VLM call, and the same
    technique already proven on real sheets by
    `backend.cv_extraction.site_plan`'s SITE_PLAN anchor search. `vision`
    means it came from a VLM's own first-pass region guess, which is not
    geometrically grounded and can be wrong (see
    `backend.vision_extraction.base`'s cross-check against this).
    """

    type: RegionType
    bbox_pts: BoundingBox
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: Optional[str] = None
    source: RegionSource
    page: int = 0


__all__ = ["RegionType", "RegionSource", "DetectedRegion"]
