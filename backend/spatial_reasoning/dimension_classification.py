"""
Semantic classification of raw `Dimension` candidates (Phase 2 output)
into the semantic types listed in phase3.md:

    PLOT_WIDTH, PLOT_DEPTH, BUILDING_WIDTH, BUILDING_DEPTH,
    FRONT_SETBACK, REAR_SETBACK, LEFT_SETBACK, RIGHT_SETBACK,
    ROAD_WIDTH, ROOM_DIMENSION, OTHER

Deliberately NOT "find number -> classify based on nearby word". The
primary signal is geometric: where the dimension's line sits relative to
the plot polygon, the building polygon(s), and the front/rear/left/right
edge groups resolved by `front_side.py`. Text keywords are a secondary,
confidence-adjusting signal, never the sole basis for a decision — a
dimension with no associated geometry line at all can still be
classified from keywords, but only at LOW/MEDIUM confidence.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from backend.config import get_settings
from backend.schemas.enums import ConfidenceLevel
from backend.schemas.geometry import Dimension, Line, Point, Polygon
from backend.spatial_reasoning import geometry_utils as geo
from backend.vision_extraction.spatial import bbox_gap, vision_bbox_to_page_points

# Imported lazily-safe (scale.py has no dependency back on this module) --
# reused here so a synthetic vision-only Dimension's unit gets converted
# to metres with the exact same logic/unit table as every other Dimension
# in the pipeline, rather than a second, possibly-drifting implementation.
from backend.spatial_reasoning.scale import dimension_length_metres


class DimensionSemanticType(str, Enum):
    PLOT_WIDTH = "PLOT_WIDTH"
    PLOT_DEPTH = "PLOT_DEPTH"
    BUILDING_WIDTH = "BUILDING_WIDTH"
    BUILDING_DEPTH = "BUILDING_DEPTH"
    FRONT_SETBACK = "FRONT_SETBACK"
    REAR_SETBACK = "REAR_SETBACK"
    LEFT_SETBACK = "LEFT_SETBACK"
    RIGHT_SETBACK = "RIGHT_SETBACK"
    ROAD_WIDTH = "ROAD_WIDTH"
    ROOM_DIMENSION = "ROOM_DIMENSION"
    OTHER = "OTHER"


_SETBACK_BY_SIDE = {
    "front": DimensionSemanticType.FRONT_SETBACK,
    "rear": DimensionSemanticType.REAR_SETBACK,
    "left": DimensionSemanticType.LEFT_SETBACK,
    "right": DimensionSemanticType.RIGHT_SETBACK,
}

_KEYWORD_PATTERNS: dict[DimensionSemanticType, re.Pattern] = {
    DimensionSemanticType.PLOT_WIDTH: re.compile(r"\b(plot|site|property)\b.*\b(width|frontage)\b", re.I),
    DimensionSemanticType.PLOT_DEPTH: re.compile(r"\b(plot|site|property)\b.*\b(depth|length)\b", re.I),
    DimensionSemanticType.BUILDING_WIDTH: re.compile(r"\b(building|house|block)\b.*\b(width)\b", re.I),
    DimensionSemanticType.BUILDING_DEPTH: re.compile(r"\b(building|house|block)\b.*\b(depth|length)\b", re.I),
    DimensionSemanticType.ROAD_WIDTH: re.compile(r"\broad\b", re.I),
}
_SETBACK_KEYWORD = re.compile(r"\bset\s*-?back\b", re.I)
_SIDE_KEYWORD = {
    "front": re.compile(r"\bfront\b", re.I),
    "rear": re.compile(r"\b(rear|back)\b", re.I),
    "left": re.compile(r"\bleft\b", re.I),
    "right": re.compile(r"\bright\b", re.I),
}
_ROOM_KEYWORD = re.compile(
    r"\b(bed\s*room|bedroom|kitchen|hall|toilet|bath(room)?|w\.?c\.?|living|dining|"
    r"store|balcony|passage|room|verandah|veranda|puja|study)\b",
    re.I,
)

_NEAR_FRACTION = 0.03  # "on the boundary" tolerance, as a fraction of the plot bbox diagonal


@dataclass
class ClassifiedDimension:
    dimension: Dimension
    semantic_type: DimensionSemanticType
    confidence: ConfidenceLevel
    reasoning: str
    value_metres: Optional[float] = None
    score_breakdown: dict[str, float] = field(default_factory=dict)


def _keyword_hits(label: str) -> dict[DimensionSemanticType, float]:
    scores: dict[DimensionSemanticType, float] = {}
    if not label:
        return scores
    if _SETBACK_KEYWORD.search(label):
        for side, pattern in _SIDE_KEYWORD.items():
            if pattern.search(label):
                scores[_SETBACK_BY_SIDE[side]] = 1.0
        if not scores:
            # "SETBACK" with no side named — weak signal spread across all four
            for t in _SETBACK_BY_SIDE.values():
                scores[t] = 0.25
    if _ROOM_KEYWORD.search(label):
        scores[DimensionSemanticType.ROOM_DIMENSION] = scores.get(DimensionSemanticType.ROOM_DIMENSION, 0) + 0.6
    for sem_type, pattern in _KEYWORD_PATTERNS.items():
        if pattern.search(label):
            scores[sem_type] = scores.get(sem_type, 0) + 0.6
    return scores


def _min_distance_to_edges(line_mid: Point, edges: list[geo.Edge]) -> float:
    if not edges:
        return math.inf
    return min(geo.point_segment_distance(line_mid, e) for e in edges)


def _plot_axis_edges(
    front_edges: list[geo.Edge], rear_edges: list[geo.Edge], left_edges: list[geo.Edge], right_edges: list[geo.Edge]
) -> tuple[float, float]:
    """Representative orientation (deg, mod 180) of the frontage axis and the depth axis."""
    frontage_edges = front_edges + rear_edges
    depth_edges = left_edges + right_edges
    frontage_deg = geo.line_orientation_degrees(frontage_edges[0]) if frontage_edges else 0.0
    depth_deg = geo.line_orientation_degrees(depth_edges[0]) if depth_edges else 90.0
    return frontage_deg, depth_deg


def _vision_semantic_hint(dim: Dimension, vision_pages) -> tuple[Optional[DimensionSemanticType], float, str]:
    """Return the strongest VLM semantic label for a matching dimension."""
    if dim.page is None or not vision_pages:
        return None, 0.0, ""
    dim_value = float(dim.magnitude)
    dim_mid = geo.edge_midpoint(dim.geometry) if dim.geometry is not None else None
    best_type: Optional[DimensionSemanticType] = None
    best_score = 0.0
    best_reason = ""

    for vp in vision_pages:
        if vp.page_number != dim.page + 1:
            continue
        for vd in vp.dimensions:
            if vd.value is None or vd.confidence <= 0:
                continue
            if abs(float(vd.value) - dim_value) > max(0.08, abs(dim_value) * 0.02):
                continue
            try:
                semantic = DimensionSemanticType(vd.type)
            except ValueError:
                continue

            spatial = 0.35
            if dim_mid is not None and vd.bbox is not None:
                vb = vision_bbox_to_page_points(vd.bbox, vp.page_width_pts, vp.page_height_pts)
                if vb is not None:
                    distance = math.hypot(dim_mid.x - vb.center.x, dim_mid.y - vb.center.y)
                    spatial = max(0.25, 1.0 - distance / max(80.0, vb.width * 3, vb.height * 3))
            score = vd.confidence * spatial
            if score > best_score:
                best_type = semantic
                best_score = score
                best_reason = (
                    f"Vision model associated value {vd.value} with {vd.type} "
                    f"(confidence {vd.confidence:.2f})."
                )
    return best_type, best_score, best_reason


def classify_dimensions(
    dimensions: list[Dimension],
    plot_polygon: Optional[Polygon],
    building_polygons: list[Polygon],
    front_edges: list[geo.Edge],
    rear_edges: list[geo.Edge],
    left_edges: list[geo.Edge],
    right_edges: list[geo.Edge],
    road_bboxes: list,
    value_metres_lookup,
    vision_pages=None,
) -> list[ClassifiedDimension]:
    """
    Classify every raw Dimension. `value_metres_lookup(dimension) -> float | None`
    resolves a dimension's real-world length in metres (via unit conversion,
    or via the scale estimate when the unit itself is unresolved).
    """
    results: list[ClassifiedDimension] = []
    plot_bbox = plot_polygon.bounding_box if plot_polygon else None
    plot_diag = math.hypot(plot_bbox.width, plot_bbox.height) if plot_bbox else 0.0
    near_tol = max(plot_diag * _NEAR_FRACTION, 1e-6) if plot_diag else 1e-6
    frontage_deg, depth_deg = (
        _plot_axis_edges(front_edges, rear_edges, left_edges, right_edges) if plot_polygon else (0.0, 90.0)
    )

    for dim in dimensions:
        label = dim.label or ""
        kw_scores = _keyword_hits(label)
        value_m = value_metres_lookup(dim)
        vision_type, vision_score, vision_reason = _vision_semantic_hint(dim, vision_pages)

        if dim.geometry is None:
            # Vision can supply semantics even when the native PDF has no dimension line.
            if vision_type is not None and vision_score >= 0.45:
                results.append(
                    ClassifiedDimension(
                        dimension=dim,
                        semantic_type=vision_type,
                        confidence=ConfidenceLevel.HIGH if vision_score >= 0.75 else ConfidenceLevel.MEDIUM,
                        reasoning=vision_reason,
                        value_metres=value_m,
                        score_breakdown={"vision": vision_score},
                    )
                )
                continue
            # No spatial anchor at all — keywords are all we have.
            if kw_scores:
                best_type = max(kw_scores, key=kw_scores.get)
                results.append(
                    ClassifiedDimension(
                        dimension=dim,
                        semantic_type=best_type,
                        confidence=ConfidenceLevel.LOW,
                        reasoning=f"No associated geometry line; classified from label text alone ('{label}').",
                        value_metres=value_m,
                        score_breakdown=kw_scores,
                    )
                )
            else:
                results.append(
                    ClassifiedDimension(
                        dimension=dim,
                        semantic_type=DimensionSemanticType.OTHER,
                        confidence=ConfidenceLevel.LOW,
                        reasoning="No associated geometry line and no recognizable label keyword.",
                        value_metres=value_m,
                    )
                )
            continue

        mid = geo.edge_midpoint(dim.geometry)
        line_deg = geo.line_orientation_degrees(dim.geometry)
        geo_scores: dict[DimensionSemanticType, float] = {}

        if plot_polygon is not None:
            dist_to_plot_boundary = geo.point_to_polygon_boundary_distance(mid, plot_polygon)
            on_plot_boundary = dist_to_plot_boundary <= near_tol
        else:
            dist_to_plot_boundary = math.inf
            on_plot_boundary = False

        nearest_building = None
        dist_to_building_boundary = math.inf
        inside_building = False
        for bp in building_polygons:
            d = geo.point_to_polygon_boundary_distance(mid, bp)
            if d < dist_to_building_boundary:
                dist_to_building_boundary = d
                nearest_building = bp
            if geo.point_in_polygon(mid, bp):
                inside_building = True
        on_building_boundary = dist_to_building_boundary <= near_tol

        dist_to_road = min((geo.point_segment_distance(mid, e) for bbox in road_bboxes for e in geo.polygon_edges(geo.bbox_to_polygon(bbox))), default=math.inf)

        # 1) Lying ON the plot boundary, parallel to it -> PLOT_WIDTH/DEPTH
        if on_plot_boundary:
            align_frontage = geo.orientation_alignment(line_deg, frontage_deg)
            align_depth = geo.orientation_alignment(line_deg, depth_deg)
            if align_frontage >= align_depth:
                geo_scores[DimensionSemanticType.PLOT_WIDTH] = 0.9 * align_frontage
            else:
                geo_scores[DimensionSemanticType.PLOT_DEPTH] = 0.9 * align_depth

        # 2) Lying ON a building boundary, parallel to it -> BUILDING_WIDTH/DEPTH
        if on_building_boundary and not on_plot_boundary:
            align_frontage = geo.orientation_alignment(line_deg, frontage_deg)
            align_depth = geo.orientation_alignment(line_deg, depth_deg)
            if align_frontage >= align_depth:
                geo_scores[DimensionSemanticType.BUILDING_WIDTH] = 0.85 * align_frontage
            else:
                geo_scores[DimensionSemanticType.BUILDING_DEPTH] = 0.85 * align_depth

        # 3) Spans the gap between a building edge and the matching plot edge,
        #    roughly PERPENDICULAR to that side -> a setback.
        if plot_polygon is not None and nearest_building is not None and not on_plot_boundary:
            for side_name, edges in (
                ("front", front_edges),
                ("rear", rear_edges),
                ("left", left_edges),
                ("right", right_edges),
            ):
                if not edges:
                    continue
                d_side = _min_distance_to_edges(mid, edges)
                if d_side <= near_tol * 4 and dist_to_building_boundary <= near_tol * 4:
                    side_deg = geo.line_orientation_degrees(edges[0])
                    perp_align = 1.0 - geo.orientation_alignment(line_deg, side_deg)
                    if perp_align > 0.4:
                        geo_scores[_SETBACK_BY_SIDE[side_name]] = geo_scores.get(
                            _SETBACK_BY_SIDE[side_name], 0
                        ) + 0.8 * perp_align

        # 4) Fully inside a building's interior, away from its own boundary -> ROOM_DIMENSION
        if inside_building and not on_building_boundary:
            geo_scores[DimensionSemanticType.ROOM_DIMENSION] = geo_scores.get(
                DimensionSemanticType.ROOM_DIMENSION, 0
            ) + 0.6

        # 5) Outside the plot altogether, near a road candidate -> ROAD_WIDTH
        if not on_plot_boundary and dist_to_plot_boundary > near_tol and dist_to_road <= near_tol * 4:
            geo_scores[DimensionSemanticType.ROAD_WIDTH] = geo_scores.get(DimensionSemanticType.ROAD_WIDTH, 0) + 0.7

        combined: dict[DimensionSemanticType, float] = dict(geo_scores)
        for t, s in kw_scores.items():
            combined[t] = combined.get(t, 0) + s
        if vision_type is not None and vision_score >= 0.35:
            # Semantic vision evidence is a strong additional signal, but it
            # cannot override a clear geometric contradiction by itself.
            combined[vision_type] = combined.get(vision_type, 0) + 1.0 * vision_score

        if not combined:
            results.append(
                ClassifiedDimension(
                    dimension=dim,
                    semantic_type=DimensionSemanticType.OTHER,
                    confidence=ConfidenceLevel.LOW,
                    reasoning="Dimension line did not align with any plot/building boundary, "
                    "setback gap, road candidate, or recognizable label.",
                    value_metres=value_m,
                )
            )
            continue

        best_type = max(combined, key=combined.get)
        best_score = combined[best_type]
        # Confidence: strong when geometry AND keyword agree; moderate for
        # geometry-only or strong-keyword-only; low otherwise.
        geo_agrees = geo_scores.get(best_type, 0) > 0
        kw_agrees = kw_scores.get(best_type, 0) > 0
        vision_agrees = vision_type is best_type and vision_score >= 0.35
        if vision_agrees and vision_reason:
            level = ConfidenceLevel.HIGH if vision_score >= 0.75 else ConfidenceLevel.MEDIUM
            reason = vision_reason
        elif geo_agrees and kw_agrees:
            level = ConfidenceLevel.HIGH
            reason = f"Geometry (proximity/orientation) and label text both indicate {best_type.value}."
        elif geo_agrees:
            level = ConfidenceLevel.MEDIUM if best_score >= 0.7 else ConfidenceLevel.LOW
            reason = f"Geometric proximity/orientation indicates {best_type.value} (label did not confirm)."
        else:
            level = ConfidenceLevel.MEDIUM if best_score >= 0.9 else ConfidenceLevel.LOW
            reason = f"Label text indicates {best_type.value}; no confirming geometry."

        results.append(
            ClassifiedDimension(
                dimension=dim,
                semantic_type=best_type,
                confidence=level,
                reasoning=reason,
                value_metres=value_m,
                score_breakdown=combined,
            )
        )

    return results


def vision_only_dimensions(
    vision_pages,
    existing_dimensions: list[Dimension],
    page: Optional[int],
    min_confidence: Optional[float] = None,
) -> list[ClassifiedDimension]:
    """
    The actual CV<->vision fusion for dimensions the deterministic
    pipeline never produced a `Dimension` candidate for at all -- as
    opposed to `_vision_semantic_hint` (used inside `classify_dimensions`
    above), which only ever *relabels* a `Dimension` the native-text/OCR
    extractor already found.

    A common real-world failure mode (see PHASE3_1_NOTES.md) is that
    native/OCR extraction misses a printed number outright: a page-level
    "ALL DIMENSIONS ARE IN METRE" convention the per-candidate regex
    can't see, garbled OCR on a scanned sheet, an unusual label format,
    etc. When that happens the vision model may still have read the
    number correctly (and it has already passed
    `base.py::_ground_against_native_text`'s hallucination check, or the
    page had no native text layer to ground against in the first place).
    Without this function that grounded reading was simply discarded --
    vision was "an evidence source, not the final authority" only in the
    sense that it could never contribute a *new* value, only vote on
    values the deterministic side already found.

    Every result here is capped at LOW confidence (it has no dimension-
    line geometry backing it) and is just one more `EvidenceCandidate`
    once merged into `pipeline.py`'s candidate lists -- normal
    evidence-reconciliation rules still apply, so a vision-only reading
    that contradicts strong geometry still surfaces as a genuine conflict
    rather than silently winning.
    """
    if page is None or not vision_pages:
        return []
    settings = get_settings()
    threshold = min_confidence if min_confidence is not None else settings.vision_only_dimension_min_confidence

    existing_values_on_page: set[float] = set()
    for dim in existing_dimensions:
        if dim.page is not None and dim.page != page:
            continue
        val = dimension_length_metres(dim)
        if val is not None:
            existing_values_on_page.add(round(val, 2))

    out: list[ClassifiedDimension] = []
    seen_this_pass: set[tuple[str, float]] = set()
    for vp in vision_pages:
        if vp.page_number != page + 1:
            continue
        for vd in vp.dimensions:
            if vd.value is None or vd.confidence < threshold:
                continue
            try:
                semantic = DimensionSemanticType(vd.type)
            except ValueError:
                continue
            if semantic in (DimensionSemanticType.OTHER, DimensionSemanticType.ROOM_DIMENSION):
                continue  # not useful for plot/building/setback/road fields

            synthetic = Dimension(
                label=f"[vision] {vd.evidence or vd.type}",
                magnitude=float(vd.value),
                unit=vd.unit or "m",
                geometry=None,
                page=page,
            )
            value_m = dimension_length_metres(synthetic)
            if value_m is None:
                continue
            rounded = round(value_m, 2)
            if rounded in existing_values_on_page:
                # Already represented by a native/OCR Dimension -- that
                # one was already fused via `_vision_semantic_hint` above;
                # adding it again here would double-count the same
                # physical reading as two independent evidence sources.
                continue
            key = (semantic.value, rounded)
            if key in seen_this_pass:
                continue
            seen_this_pass.add(key)

            out.append(
                ClassifiedDimension(
                    dimension=synthetic,
                    semantic_type=semantic,
                    confidence=ConfidenceLevel.LOW,
                    reasoning=(
                        f"Vision-only evidence: model read '{vd.evidence}' as {vd.value}"
                        f"{vd.unit or ''} classified {vd.type} (confidence {vd.confidence:.2f}). "
                        "No matching native/OCR dimension was found on this page, so this is not "
                        "corroborated by dimension-line geometry -- treated as a single additional "
                        "evidence source, still subject to reconciliation against everything else."
                    ),
                    value_metres=value_m,
                    score_breakdown={"vision_only": vd.confidence},
                )
            )
    return out


__all__ = [
    "DimensionSemanticType",
    "ClassifiedDimension",
    "classify_dimensions",
    "vision_only_dimensions",
]
