"""
First-pass geometric candidate detection.

Per ARCHITECTURE.md: "Extraction (Phase 2) produces ExtractionResult per
document. It may emit multiple competing PlotCandidate / BuildingCandidate
/ RoadCandidate for the same region — resolution happens next [in spatial
reasoning]."

Everything produced here is a ROUGH, LOW/MEDIUM-confidence geometric
guess in un-normalized PAGE_POINTS space — never a resolved,
authoritative measurement. `value` is deliberately left unset
(only `raw_value` + geometry + evidence are populated) because real
page->metric conversion is a spatial-reasoning concern once a scale
reference is established. This module does NOT decide FRONT_SETBACK /
PLOT_WIDTH / BUILDING_WIDTH or any other final semantic field — it only
proposes "this closed shape is plausibly a plot/building/road outline".

Heuristics used (all overridable/replaceable by spatial reasoning):
  - The largest closed polygon on a page is the PLOT candidate.
  - Closed polygons substantially contained within the plot candidate's
    bounding box are BUILDING candidates.
  - Long, thin rectangles/polygons whose nearby text mentions "road" are
    ROAD candidates — never fabricated without textual support.
"""

from __future__ import annotations

from backend.config import get_settings
from backend.cv_extraction.raw_types import (
    CoordinateTransformRecord,
    DimensionCandidate,
    RawPolygon,
    RawRectangle,
    RawTextItem,
)
from backend.schemas.candidates import BuildingCandidate, PlotCandidate, RoadCandidate
from backend.schemas.enums import ConfidenceLevel
from backend.schemas.evidence import Confidence, GeometryEvidence, TextEvidence, ValueField
from backend.schemas.geometry import BoundingBox, CoordinateSpace, NormalizedGeometry, Polygon
from backend.spatial_reasoning import geometry_utils as geo

# A polygon must be at least this large a fraction of the page area to
# even be considered a plot/building candidate (filters out small
# furniture/annotation glyphs picked up by OpenCV contour detection).
MIN_AREA_FRACTION_OF_PAGE = 0.01

# Aspect-ratio threshold (long side / short side) above which a shape is
# "road-shaped" — a plausible long, thin strip.
ROAD_ASPECT_RATIO_THRESHOLD = 4.0

_UNRESOLVED = "raw geometric candidate — un-normalized page-space bounding box, awaiting scale resolution"


def _bbox_area(bbox: BoundingBox) -> float:
    return max(0.0, bbox.width) * max(0.0, bbox.height)


def _unresolved_value_field(
    magnitude_pts: float,
    unit: str,
    evidence: list[TextEvidence | GeometryEvidence],
    note: str,
) -> ValueField[float]:
    from backend.schemas.units import UnitValue

    return ValueField[float](
        value=None,
        raw_value=UnitValue(magnitude=round(magnitude_pts, 3), unit=unit),
        confidence=Confidence(level=ConfidenceLevel.LOW, reason=note),
        source="cv_extraction geometric candidate heuristic",
        evidence=evidence,
    )


def _geometry_for(
    polygon: Polygon,
    bbox: BoundingBox,
    page: int,
    transform: CoordinateTransformRecord | None,
) -> NormalizedGeometry:
    return NormalizedGeometry(
        coordinate_space=CoordinateSpace.PAGE_POINTS,
        polygon=polygon,
        bounding_box=bbox,
        points_per_metre=(transform.provisional_points_per_metre if transform else 1.0),
        rotation_degrees=(transform.rotation_degrees if transform else 0.0),
        source_page=page,
    )


def _polygon_bbox(rp: RawPolygon) -> BoundingBox:
    return rp.polygon.bounding_box


def build_plot_and_building_candidates(
    document_id: str,
    polygons: list[RawPolygon],
    page_areas_pts2: dict[int, float],
    transforms: dict[int, CoordinateTransformRecord],
    page_dims_pts: dict[int, tuple[float, float]] | None = None,
) -> tuple[list[PlotCandidate], list[BuildingCandidate]]:
    """
    Produce MULTIPLE plausible plot candidates (never just "the largest
    closed polygon"), and let downstream spatial reasoning
    (`plot_resolution.score_plot_candidates`) rank them.

    Per FIX #1 / FIX #2 (phase3.1): before a geometry candidate is
    eligible, it must survive:

      1. degenerate/invalid-polygon rejection (`geo.is_valid_polygon`)
      2. duplicate-geometry removal (`geo.dedupe_bboxes`)
      3. page-frame / drawing-sheet-border rejection (`geo.is_page_frame_like`)

    Every polygon that survives is emitted as BOTH a PlotCandidate and a
    BuildingCandidate — this is intentional, not a bug: which
    interpretation is correct for a given region is a resolution
    decision, not a generation-time decision. `plot_resolution.py` scores
    plot-likeness (rectangularity, boundary-dimension coverage, plot/site
    label proximity, road adjacency, nesting); `building_resolution.py`
    separately filters building-likeness against the *resolved* plot
    (rejecting anything ~= the plot's own footprint as a duplicate/
    compound-wall candidate). Neither module assumes it received exactly
    one candidate.
    """
    settings = get_settings()
    page_dims_pts = page_dims_pts or {}
    plot_candidates: list[PlotCandidate] = []
    building_candidates: list[BuildingCandidate] = []

    by_page: dict[int, list[RawPolygon]] = {}
    for rp in polygons:
        if not rp.is_closed:
            continue
        by_page.setdefault(rp.page, []).append(rp)

    for page, page_polygons in by_page.items():
        page_area = page_areas_pts2.get(page, 0.0)
        page_width, page_height = page_dims_pts.get(page, (0.0, 0.0))
        min_area = page_area * MIN_AREA_FRACTION_OF_PAGE if page_area else 0.0

        sized = [
            (rp, rp.area_pts2 if rp.area_pts2 is not None else rp.polygon.area)
            for rp in page_polygons
        ]
        sized = [(rp, area) for rp, area in sized if area >= min_area]

        # 1. Reject degenerate/invalid polygons.
        valid = [
            (rp, area)
            for rp, area in sized
            if geo.is_valid_polygon(
                rp.polygon,
                min_area=settings.min_valid_polygon_area_pts2,
                max_perimeter_area_ratio=settings.max_valid_perimeter_area_ratio,
            )
        ]

        # 2. Remove duplicate geometry (near-identical bounding boxes).
        deduped_pairs = geo.dedupe_bboxes(
            [(pair, _polygon_bbox(pair[0])) for pair in valid],
            tolerance=settings.candidate_dedupe_tolerance_pts,
        )
        deduped = [pair for pair, _bbox in deduped_pairs]

        # 3. Reject page-frame / drawing-sheet-border geometry.
        surviving: list[tuple[RawPolygon, float]] = []
        frame_rejected = 0
        for rp, area in deduped:
            bbox = _polygon_bbox(rp)
            if page_width and page_height and geo.is_page_frame_like(
                bbox,
                page_width,
                page_height,
                touch_tolerance=settings.page_frame_touch_tolerance_pts,
                area_fraction_threshold=settings.page_frame_area_fraction_threshold,
            ):
                frame_rejected += 1
                continue
            surviving.append((rp, area))

        if not surviving:
            continue
        surviving.sort(key=lambda pair: pair[1], reverse=True)

        transform = transforms.get(page)
        total = len(surviving)
        for idx, (rp, area) in enumerate(surviving):
            bbox = _polygon_bbox(rp)
            geom_evidence = GeometryEvidence(
                page=page,
                bounding_box=bbox,
                description=(
                    f"closed polygon candidate, area rank {idx + 1}/{total} on page "
                    f"({frame_rejected} page-frame-like candidate(s) rejected)"
                ),
            )
            note = (
                f"Closed polygon candidate (area rank {idx + 1}/{total} on page), classified as "
                "a candidate PLOT/BUILDING interpretation by geometry heuristic. Unresolved — "
                "competing interpretations not yet ruled out; not automatically the largest."
            )
            plot_candidates.append(
                PlotCandidate(
                    id=f"{document_id}-plot-p{page}-{idx}",
                    geometry=_geometry_for(rp.polygon, bbox, page, transform),
                    confidence_note=note,
                    width=_unresolved_value_field(bbox.width, "pt", [geom_evidence], _UNRESOLVED),
                    depth=_unresolved_value_field(bbox.height, "pt", [geom_evidence], _UNRESOLVED),
                    area=_unresolved_value_field(area, "pt2", [geom_evidence], _UNRESOLVED),
                )
            )
            building_candidates.append(
                BuildingCandidate(
                    id=f"{document_id}-building-p{page}-{idx}",
                    geometry=_geometry_for(rp.polygon, bbox, page, transform),
                    confidence_note=note,
                    width=_unresolved_value_field(bbox.width, "pt", [geom_evidence], _UNRESOLVED),
                    depth=_unresolved_value_field(bbox.height, "pt", [geom_evidence], _UNRESOLVED),
                    footprint_area=_unresolved_value_field(area, "pt2", [geom_evidence], _UNRESOLVED),
                )
            )

    return plot_candidates, building_candidates


def build_road_candidates(
    document_id: str,
    polygons: list[RawPolygon],
    rectangles: list[RawRectangle],
    text_items: list[RawTextItem],
    transforms: dict[int, CoordinateTransformRecord],
    proximity_pts: float = 60.0,
) -> list[RoadCandidate]:
    """
    Only proposes a ROAD candidate for a long/thin shape that has nearby
    text mentioning "road" — never fabricated from geometry alone.
    """
    road_texts = [t for t in text_items if "road" in t.text.lower()]
    if not road_texts:
        return []

    shapes: list[tuple[int, BoundingBox, object]] = []
    for rp in polygons:
        shapes.append((rp.page, _polygon_bbox(rp), rp))
    for rr in rectangles:
        shapes.append((rr.page, rr.bounding_box, rr))

    candidates: list[RoadCandidate] = []
    seen_bboxes: set[tuple] = set()
    idx = 0
    for page, bbox, source_obj in shapes:
        width, height = bbox.width, bbox.height
        if min(width, height) <= 0:
            continue
        aspect = max(width, height) / min(width, height)
        if aspect < ROAD_ASPECT_RATIO_THRESHOLD:
            continue
        near_road_text = any(
            t.page == page and _bbox_distance(bbox, t.bounding_box) <= proximity_pts
            for t in road_texts
        )
        if not near_road_text:
            continue
        key = (page, round(bbox.min_x, 1), round(bbox.min_y, 1), round(bbox.max_x, 1), round(bbox.max_y, 1))
        if key in seen_bboxes:
            continue
        seen_bboxes.add(key)

        transform = transforms.get(page)
        geom_evidence = GeometryEvidence(
            page=page, bounding_box=bbox, description="long/thin shape near 'road' text label"
        )
        label = next((t.text for t in road_texts if t.page == page), None)
        candidates.append(
            RoadCandidate(
                id=f"{document_id}-road-p{page}-{idx}",
                geometry=NormalizedGeometry(
                    coordinate_space=CoordinateSpace.PAGE_POINTS,
                    bounding_box=bbox,
                    points_per_metre=(transform.provisional_points_per_metre if transform else 1.0),
                    rotation_degrees=(transform.rotation_degrees if transform else 0.0),
                    source_page=page,
                ),
                confidence_note=(
                    "Long/thin shape adjacent to text mentioning 'road', classified as a ROAD "
                    "candidate by proximity heuristic. Unresolved."
                ),
                width=_unresolved_value_field(
                    min(width, height), "pt", [geom_evidence], _UNRESOLVED
                ),
                name_or_label=label,
            )
        )
        idx += 1

    return candidates


def _bbox_distance(a: BoundingBox, b: BoundingBox) -> float:
    if a.intersects(b):
        return 0.0
    dx = max(a.min_x - b.max_x, b.min_x - a.max_x, 0.0)
    dy = max(a.min_y - b.max_y, b.min_y - a.max_y, 0.0)
    return (dx * dx + dy * dy) ** 0.5


__all__ = [
    "build_plot_and_building_candidates",
    "build_road_candidates",
    "MIN_AREA_FRACTION_OF_PAGE",
    "ROAD_ASPECT_RATIO_THRESHOLD",
]
