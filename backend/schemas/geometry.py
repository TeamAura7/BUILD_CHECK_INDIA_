"""
Core geometric primitives shared by every pipeline stage.

Coordinate contract
--------------------
* All Point / BoundingBox / Line / Polygon coordinates are in a single
  per-document LOCAL coordinate space: page points at extraction time,
  metres after normalization. Which space a geometry is in is tracked by
  `coordinate_space` on NormalizedGeometry — geometry never silently
  changes space.
* Origin is top-left of the source page, X right, Y down (matches PDF/
  image page conventions). Normalization to a metric, Y-up plan space is
  a NormalizedGeometry concern, not something baked into these primitives.
* Rotation, scale and DPI/points-per-metre factors used to go from page
  space to metric space MUST be recorded (see NormalizedGeometry) so the
  transform is reproducible and auditable.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from backend.schemas.enums import SpatialRelationType


class CoordinateSpace(str, Enum):
    PAGE_POINTS = "PAGE_POINTS"   # raw extraction space, unit = pdf/image points/px
    METRIC_PLAN = "METRIC_PLAN"   # normalized space, unit = metres


class Point(BaseModel):
    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


class BoundingBox(BaseModel):
    """Axis-aligned bounding box, min/max corners."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @field_validator("max_x")
    @classmethod
    def _x_ordered(cls, v: float, info) -> float:
        min_x = info.data.get("min_x")
        if min_x is not None and v < min_x:
            raise ValueError("max_x must be >= min_x")
        return v

    @field_validator("max_y")
    @classmethod
    def _y_ordered(cls, v: float, info) -> float:
        min_y = info.data.get("min_y")
        if min_y is not None and v < min_y:
            raise ValueError("max_y must be >= min_y")
        return v

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def center(self) -> Point:
        return Point(x=(self.min_x + self.max_x) / 2, y=(self.min_y + self.max_y) / 2)

    def intersects(self, other: "BoundingBox") -> bool:
        return not (
            self.max_x < other.min_x
            or other.max_x < self.min_x
            or self.max_y < other.min_y
            or other.max_y < self.min_y
        )


class Line(BaseModel):
    start: Point
    end: Point

    @property
    def length(self) -> float:
        return ((self.end.x - self.start.x) ** 2 + (self.end.y - self.start.y) ** 2) ** 0.5


class Polygon(BaseModel):
    """Simple polygon as an ordered ring of points (not necessarily closed explicitly)."""

    points: list[Point] = Field(..., min_length=3)

    @property
    def bounding_box(self) -> BoundingBox:
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return BoundingBox(min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys))

    @property
    def area(self) -> float:
        """Shoelace formula. Positive/negative sign is winding order, magnitude matters."""
        pts = self.points
        n = len(pts)
        total = 0.0
        for i in range(n):
            j = (i + 1) % n
            total += pts[i].x * pts[j].y - pts[j].x * pts[i].y
        return abs(total) / 2.0


class Dimension(BaseModel):
    """A labeled measurement read off a plan (a dimension line, a text label, etc.)."""

    label: Optional[str] = None
    magnitude: float
    unit: str  # LengthUnit / AreaUnit value as string; validated at higher layer
    geometry: Optional[Line] = None  # the dimension line this came from, if any
    text_bounding_box: Optional[BoundingBox] = None  # exact bbox of the printed dimension text
    page: Optional[int] = None  # 0-based source page for semantic/vision association
    # Raw 0-1 score carried over from cv_extraction.raw_types.DimensionCandidate.
    # confidence (same convention as VisionDimension.confidence /
    # IndependentMeasurement.confidence) -- previously computed and then
    # discarded when a DimensionCandidate was folded into a Dimension here,
    # leaving every dimension in ExtractionResult.dimensions
    # confidence-blind. None means no candidate-level score is available
    # (e.g. a Dimension built by something other than
    # PDFHybridExtractor._fold). Optional and additive: existing callers
    # that never read this field are unaffected.
    confidence: Optional[float] = Field(default=None, ge=0, le=1)


class NormalizedGeometry(BaseModel):
    """
    A geometry that has been converted from page space into metric plan
    space, with the transform used recorded for auditability.

    This is the ONLY geometry shape the RuleEngine and downstream
    consumers should reason about — never raw page-space geometry.
    """

    coordinate_space: CoordinateSpace = CoordinateSpace.METRIC_PLAN
    polygon: Optional[Polygon] = None
    bounding_box: Optional[BoundingBox] = None
    points_per_metre: float = Field(
        ..., gt=0, description="Scale factor used to convert page points -> metres"
    )
    rotation_degrees: float = 0.0
    source_page: Optional[int] = None


class SpatialRelation(BaseModel):
    """Qualitative relation between two entities, referenced by id."""

    subject_id: str
    object_id: str
    relation: SpatialRelationType
    distance_m: Optional[float] = None
    notes: Optional[str] = None
