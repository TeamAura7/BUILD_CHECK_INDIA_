"""
Synthetic geometry builders for Phase 3 (spatial reasoning) unit/integration
tests — analogous to `pdf_builders.py` but operating directly on the
Phase 1/2 schema objects (`ExtractionResult`, `PlotCandidate`, ...) instead
of real PDFs, so tests can construct precise, known geometry without
depending on PDF rendering/OCR fidelity.

Coordinate convention matches `schemas/geometry.py`: PAGE_POINTS space,
origin top-left, X right, Y down. All builders here use
`points_per_metre=40.0` unless overridden, so a 200x200 pt square plot is
exactly 5m x 5m.
"""

from __future__ import annotations

from typing import Optional

from backend.schemas.candidates import BuildingCandidate, PlotCandidate, RoadCandidate
from backend.schemas.enums import DocumentType
from backend.schemas.evidence import TextEvidence, ValueField
from backend.schemas.extraction import ExtractionResult
from backend.schemas.geometry import (
    BoundingBox,
    CoordinateSpace,
    Dimension,
    Line,
    NormalizedGeometry,
    Point,
    Polygon,
)

DEFAULT_PPM = 40.0


def rect_polygon(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    return Polygon(
        points=[Point(x=x0, y=y0), Point(x=x1, y=y0), Point(x=x1, y=y1), Point(x=x0, y=y1)]
    )


def geom(polygon: Polygon, ppm: float = DEFAULT_PPM, page: int = 0) -> NormalizedGeometry:
    return NormalizedGeometry(
        coordinate_space=CoordinateSpace.PAGE_POINTS,
        polygon=polygon,
        bounding_box=polygon.bounding_box,
        points_per_metre=ppm,
        source_page=page,
    )


def bbox_geom(bbox: BoundingBox, ppm: float = DEFAULT_PPM, page: int = 0) -> NormalizedGeometry:
    return NormalizedGeometry(
        coordinate_space=CoordinateSpace.PAGE_POINTS,
        bounding_box=bbox,
        points_per_metre=ppm,
        source_page=page,
    )


def plot_candidate(polygon: Polygon, cid: str = "plot-1", ppm: float = DEFAULT_PPM, page: int = 0) -> PlotCandidate:
    return PlotCandidate(
        id=cid,
        geometry=geom(polygon, ppm, page),
        width=ValueField.missing(),
        depth=ValueField.missing(),
        area=ValueField.missing(),
    )


def building_candidate(
    polygon: Polygon, cid: str = "building-1", ppm: float = DEFAULT_PPM, page: int = 0
) -> BuildingCandidate:
    return BuildingCandidate(
        id=cid,
        geometry=geom(polygon, ppm, page),
        width=ValueField.missing(),
        depth=ValueField.missing(),
        footprint_area=ValueField.missing(),
    )


def road_candidate(
    bbox: BoundingBox, cid: str = "road-1", label: Optional[str] = None, ppm: float = DEFAULT_PPM, page: int = 0
) -> RoadCandidate:
    return RoadCandidate(
        id=cid,
        geometry=bbox_geom(bbox, ppm, page),
        width=ValueField.missing(),
        name_or_label=label,
    )


def dim(
    magnitude: float,
    unit: str = "m",
    label: Optional[str] = None,
    line: Optional[Line] = None,
) -> Dimension:
    return Dimension(label=label or f"{magnitude} {unit}", magnitude=magnitude, unit=unit, geometry=line)


def text(raw_text: str, page: int = 0, bbox: Optional[BoundingBox] = None) -> TextEvidence:
    return TextEvidence(raw_text=raw_text, page=page, bounding_box=bbox)


def make_extraction(
    document_id: str = "doc-1",
    plot_candidates: Optional[list[PlotCandidate]] = None,
    building_candidates: Optional[list[BuildingCandidate]] = None,
    road_candidates: Optional[list[RoadCandidate]] = None,
    dimensions: Optional[list[Dimension]] = None,
    text_evidence: Optional[list[TextEvidence]] = None,
    page_count: int = 1,
    document_type: DocumentType = DocumentType.VECTOR_PDF,
) -> ExtractionResult:
    return ExtractionResult(
        document_id=document_id,
        document_type=document_type,
        page_count=page_count,
        plot_candidates=plot_candidates or [],
        building_candidates=building_candidates or [],
        road_candidates=road_candidates or [],
        dimensions=dimensions or [],
        text_evidence=text_evidence or [],
    )


def standard_rectangular_plan(
    plot_w_pts: float = 400.0,
    plot_h_pts: float = 480.0,
    margin_pts: float = 60.0,
    ppm: float = DEFAULT_PPM,
) -> ExtractionResult:
    """
    A clean rectangular plot with a centered rectangular building, a road
    strip below the plot (south edge = front), a FRONT-label-free layout
    (front resolved via ROAD candidate), and matching dimension labels
    with geometry lines for width/depth. Building is inset from every
    edge by `margin_pts`, so setbacks are exactly margin_pts/ppm metres
    on every side.
    """
    plot_poly = rect_polygon(0, 0, plot_w_pts, plot_h_pts)
    building_poly = rect_polygon(margin_pts, margin_pts, plot_w_pts - margin_pts, plot_h_pts - margin_pts)
    road_bbox = BoundingBox(min_x=0, min_y=plot_h_pts + 10, max_x=plot_w_pts, max_y=plot_h_pts + 40)

    width_dim = dim(
        plot_w_pts / ppm, "m", label="PLOT WIDTH", line=Line(start=Point(x=0, y=-10), end=Point(x=plot_w_pts, y=-10))
    )
    depth_dim = dim(
        plot_h_pts / ppm,
        "m",
        label="PLOT DEPTH",
        line=Line(start=Point(x=-10, y=0), end=Point(x=-10, y=plot_h_pts)),
    )

    return make_extraction(
        document_id="standard-rect",
        plot_candidates=[plot_candidate(plot_poly, ppm=ppm)],
        building_candidates=[building_candidate(building_poly, ppm=ppm)],
        road_candidates=[road_candidate(road_bbox, label="9M ROAD", ppm=ppm)],
        dimensions=[width_dim, depth_dim],
        text_evidence=[
            text("PLOT", bbox=BoundingBox(min_x=10, min_y=10, max_x=60, max_y=25)),
            text(
                "9.0 M WIDE ROAD",
                bbox=BoundingBox(min_x=0, min_y=plot_h_pts + 45, max_x=100, max_y=plot_h_pts + 60),
            ),
        ],
    )


__all__ = [
    "DEFAULT_PPM",
    "rect_polygon",
    "geom",
    "bbox_geom",
    "plot_candidate",
    "building_candidate",
    "road_candidate",
    "dim",
    "text",
    "make_extraction",
    "standard_rectangular_plan",
]
