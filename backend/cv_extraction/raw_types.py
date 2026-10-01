"""
Raw extraction intermediate types — Phase 2 internal representation.

These are deliberately NOT part of the Phase 1 contract
(`backend/schemas/`). Phase 1's `ExtractionResult` is the flat, resolved
shape that spatial reasoning consumes; it does not have room for
per-primitive vector lines, raw OpenCV contours, or word-level OCR boxes.

`RawExtractionBundle` is the low-level, page-accurate record of
*everything* the hybrid pipeline found (PDF-native, OCR, OpenCV) before
it gets folded into an `ExtractionResult`. It exists so that:

  - debug overlays can show exactly what was extracted, per source
  - tests can assert on individual primitives instead of only the final
    folded output
  - the folding step (`cv_extraction/pdf_extractor.py`) is a pure,
    inspectable function of this bundle

Every item here is traceable back to a page and, where relevant, a
source location (`SourceKind` + PAGE_POINTS coordinates).

Coordinate space: everything in this module is in PAGE_POINTS
(the source PDF page's native point space, origin top-left, X right,
Y down — see `backend/schemas/geometry.py`), regardless of whether it
came from vector extraction, PDF text, OCR, or an OpenCV raster pass.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from backend.schemas.geometry import BoundingBox, Line, Point, Polygon
from backend.schemas.vision import VisionPageResult


class SourceKind(str, Enum):
    """Which extraction stage produced a given raw primitive."""

    VECTOR_PDF = "VECTOR_PDF"       # PyMuPDF drawing/path extraction
    PDF_TEXT = "PDF_TEXT"           # PyMuPDF native text extraction
    OCR = "OCR"                     # Tesseract/PaddleOCR fallback
    OPENCV_RASTER = "OPENCV_RASTER"  # OpenCV line/contour detection on a rasterized page


class PageMetadata(BaseModel):
    """Per-page facts needed to interpret every other raw record on that page."""

    page_number: int = Field(..., ge=0, description="0-indexed page number")
    width_pts: float
    height_pts: float
    rotation_degrees: int = Field(default=0, description="Page /Rotate entry, normalized to 0/90/180/270")
    has_native_text: bool = Field(
        default=False, description="True if PyMuPDF found any selectable text on this page"
    )
    is_scanned: bool = Field(
        default=False,
        description="True if this page had to fall back to OCR (no/insufficient native text)",
    )
    vector_object_count: int = Field(default=0, description="Number of vector paths found on the page")


class RawTextItem(BaseModel):
    """A single text span/word, from either native PDF text or OCR."""

    text: str
    bounding_box: BoundingBox
    page: int
    source: SourceKind
    font_size: Optional[float] = None
    orientation_degrees: Optional[float] = None
    ocr_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    # True only for a synthetic OCR item that merges several individual
    # word-level detections into one same-line phrase (see
    # `ocr_fallback.ocr_page`). Tesseract's `image_to_data` is word-granular
    # -- "SITE" and "PLAN" arrive as two separate items -- which silently
    # breaks every multi-word regex in `site_plan.py` (the "SITE PLAN"
    # anchor, "WIDE ROAD", ID-label exclusions, area/height table labels)
    # on any scanned/OCR-only page, even when OCR read every word correctly.
    # Line-grouped items exist purely so phrase matching can see the whole
    # line; numeric-candidate extraction should skip them (the tighter
    # per-word bounding box is what position-sensitive picking needs) and
    # use this flag to do so.
    is_line_group: bool = False


class RawLine(BaseModel):
    """A single straight-line primitive, vector or raster-detected."""

    line: Line
    page: int
    source: SourceKind
    stroke_width: Optional[float] = None


class RawRectangle(BaseModel):
    """A single axis-aligned (or near-axis-aligned) rectangle primitive."""

    bounding_box: BoundingBox
    page: int
    source: SourceKind


class RawPolygon(BaseModel):
    """A single closed/near-closed polygon primitive (outline, contour, wall loop)."""

    polygon: Polygon
    page: int
    source: SourceKind
    is_closed: bool = True
    area_pts2: Optional[float] = None


class CoordinateTransformRecord(BaseModel):
    """
    Records exactly how raster-space (OCR/OpenCV) coordinates on a page were
    mapped back into that page's native PAGE_POINTS space, and what scale
    factor is provisionally attached for a later, real page->metric
    conversion (spatial reasoning owns doing that conversion for real,
    e.g. from a detected scale bar or dimension cross-check).
    """

    page: int
    raster_dpi: float
    points_per_pixel: float = Field(..., gt=0, description="page points = pixel * points_per_pixel")
    rotation_degrees: int
    page_width_pts: float
    page_height_pts: float
    provisional_points_per_metre: float = Field(
        ..., gt=0, description="Fallback page->metric scale; NOT authoritative, spatial reasoning may override"
    )


class DimensionCandidate(BaseModel):
    """
    A candidate numerical annotation that MIGHT be a dimension.

    Deliberately unresolved: no FRONT_SETBACK / PLOT_WIDTH / etc. label.
    That semantic assignment belongs to spatial reasoning / Phase 3.
    """

    raw_text: str
    numeric_value: float
    unit_hint: Optional[str] = Field(
        default=None, description="Unit token as detected in text, e.g. 'm', 'ft', 'sq.ft' — unvalidated"
    )
    bounding_box: BoundingBox
    page: int
    source: SourceKind
    nearby_geometry_ids: list[int] = Field(
        default_factory=list, description="Indices into RawExtractionBundle.lines this candidate sits near"
    )
    orientation_degrees: Optional[float] = None
    confidence: float = Field(..., ge=0, le=1)


class RawExtractionBundle(BaseModel):
    """
    Full raw output of the hybrid extraction pipeline for one document,
    BEFORE folding into the Phase 1 `ExtractionResult` contract.
    """

    document_id: str
    pages: list[PageMetadata] = Field(default_factory=list)

    text_evidence: list[RawTextItem] = Field(default_factory=list)
    geometry_evidence: list[RawLine | RawRectangle | RawPolygon] = Field(default_factory=list)

    lines: list[RawLine] = Field(default_factory=list)
    rectangles: list[RawRectangle] = Field(default_factory=list)
    polygons: list[RawPolygon] = Field(default_factory=list)

    dimensions_candidates: list[DimensionCandidate] = Field(default_factory=list)
    ocr_evidence: list[RawTextItem] = Field(default_factory=list)
    page_metadata: list[PageMetadata] = Field(default_factory=list)
    coordinate_transform: list[CoordinateTransformRecord] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)
    vision_pages: list[VisionPageResult] = Field(default_factory=list)

    def transform_for_page(self, page: int) -> Optional[CoordinateTransformRecord]:
        for t in self.coordinate_transform:
            if t.page == page:
                return t
        return None


__all__ = [
    "SourceKind",
    "PageMetadata",
    "RawTextItem",
    "RawLine",
    "RawRectangle",
    "RawPolygon",
    "CoordinateTransformRecord",
    "DimensionCandidate",
    "RawExtractionBundle",
]
