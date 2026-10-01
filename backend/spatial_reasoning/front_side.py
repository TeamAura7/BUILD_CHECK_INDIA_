"""
Front-side reasoning.

NEVER assumes PDF-page-top = front. Front is resolved from evidence, in
this priority order (per phase3.md):

    1. explicit "FRONT" text label
    2. a ROAD candidate (from Phase 2 candidate geometry)
    3. "STREET" text
    4. "ACCESS" text
    5. "MAIN ENTRY" / "ENTRANCE" text
    6. "GATE" text
    7. (orientation info — no cardinal-direction source is available from
       Phase 2 today, so this level is a no-op placeholder for when one is)
    8. spatial relationships — deterministic, clearly-flagged LOW-confidence
       fallback when nothing above matched anything

Once a front edge is chosen: rear = the plot-boundary edge farthest from
it; left/right are assigned from the *viewer's* perspective standing at
the front edge and facing into the plot (i.e. as if arriving from the
road) — see `_assign_left_right` for the exact convention, which is
recorded in the returned reasoning so it's auditable rather than assumed.

    plot width = frontage (front/rear edge length)
    plot depth = perpendicular dimension (left/right edge length)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from backend.schemas.enums import ConfidenceLevel
from backend.schemas.geometry import Point, Polygon
from backend.spatial_reasoning import geometry_utils as geo
from backend.spatial_reasoning.road_access import AccessEvidence


@dataclass
class FrontSideResolution:
    front_edges: list[geo.Edge] = field(default_factory=list)
    rear_edges: list[geo.Edge] = field(default_factory=list)
    left_edges: list[geo.Edge] = field(default_factory=list)
    right_edges: list[geo.Edge] = field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MISSING
    reasoning: str = ""
    evidence_level: Optional[str] = None  # which priority tier resolved it


_PRIORITY_TO_CONFIDENCE = {
    "FRONT label": ConfidenceLevel.HIGH,
    "ROAD candidate": ConfidenceLevel.HIGH,
    "STREET text": ConfidenceLevel.MEDIUM,
    "ACCESS text": ConfidenceLevel.MEDIUM,
    "MAIN ENTRY text": ConfidenceLevel.MEDIUM,
    "GATE text": ConfidenceLevel.MEDIUM,
}


def _nearest_edge_to_point(edges: list[geo.Edge], p: Point) -> geo.Edge:
    return min(edges, key=lambda e: geo.point_segment_distance(p, e))


def _nearest_edge_to_bbox(edges: list[geo.Edge], bbox) -> geo.Edge:
    center = bbox.center
    return min(edges, key=lambda e: geo.point_segment_distance(center, e))


def _assign_left_right(
    edges: list[geo.Edge], front_edge: geo.Edge, rear_edge: geo.Edge
) -> tuple[list[geo.Edge], list[geo.Edge]]:
    """
    Convention: stand on the front edge, face the rear edge (i.e. walk
    from the road into the plot). "Right" is the side your right hand
    points to. In this PAGE_POINTS space (X right, Y down, matching
    screen conventions) that is the forward vector rotated +90 degrees
    mathematically (fx, fy) -> (-fy, fx).
    """
    front_mid = geo.edge_midpoint(front_edge)
    rear_mid = geo.edge_midpoint(rear_edge)
    fx, fy = rear_mid.x - front_mid.x, rear_mid.y - front_mid.y
    norm = math.hypot(fx, fy) or 1.0
    fx, fy = fx / norm, fy / norm
    right_vec = (-fy, fx)

    left_edges, right_edges = [], []
    for e in edges:
        if e is front_edge or e is rear_edge:
            continue
        mid = geo.edge_midpoint(e)
        rel = (mid.x - front_mid.x, mid.y - front_mid.y)
        proj = rel[0] * right_vec[0] + rel[1] * right_vec[1]
        (right_edges if proj >= 0 else left_edges).append(e)
    return left_edges, right_edges


def resolve_front_side(
    plot_polygon: Polygon,
    road_bbox: Optional[object],
    access_evidence: list[AccessEvidence],
) -> FrontSideResolution:
    edges = geo.polygon_edges(plot_polygon)
    if len(edges) < 3:
        return FrontSideResolution(
            confidence=ConfidenceLevel.MISSING, reasoning="Plot polygon is degenerate (fewer than 3 edges)."
        )

    front_edge: Optional[geo.Edge] = None
    evidence_level: Optional[str] = None

    front_labels = [a for a in access_evidence if a.kind == "front" and a.text_evidence.bounding_box]
    if front_labels:
        pts = [t.text_evidence.bounding_box.center for t in front_labels]
        # nearest edge across all FRONT-labelled text occurrences
        front_edge = min(edges, key=lambda e: min(geo.point_segment_distance(p, e) for p in pts))
        evidence_level = "FRONT label"

    if front_edge is None and road_bbox is not None:
        front_edge = _nearest_edge_to_bbox(edges, road_bbox)
        evidence_level = "ROAD candidate"

    if front_edge is None:
        for kind, level_name in (
            ("street", "STREET text"),
            ("access", "ACCESS text"),
            ("main_entry", "MAIN ENTRY text"),
            ("gate", "GATE text"),
        ):
            matches = [a for a in access_evidence if a.kind == kind and a.text_evidence.bounding_box]
            if matches:
                pts = [t.text_evidence.bounding_box.center for t in matches]
                front_edge = min(edges, key=lambda e: min(geo.point_segment_distance(p, e) for p in pts))
                evidence_level = level_name
                break

    if front_edge is None:
        # Last resort: deterministic, clearly-flagged fallback. This is
        # NOT "top of page = front" — it is the polygon's first edge in
        # ring order, chosen only because no directional evidence of any
        # kind exists, and it is surfaced at LOW confidence rather than
        # silently trusted.
        front_edge = edges[0]
        evidence_level = None

    rear_edge = max(edges, key=lambda e: geo.point_segment_distance(geo.edge_midpoint(front_edge), e))
    if rear_edge is front_edge and len(edges) > 1:
        rear_edge = max(
            (e for e in edges if e is not front_edge),
            key=lambda e: geo.point_segment_distance(geo.edge_midpoint(front_edge), e),
        )

    left_edges, right_edges = _assign_left_right(edges, front_edge, rear_edge)

    if evidence_level is not None:
        confidence = _PRIORITY_TO_CONFIDENCE[evidence_level]
        reasoning = (
            f"Front edge resolved from {evidence_level}. Rear = farthest plot edge from the front "
            "edge. Left/right assigned as if standing at the front edge facing into the plot "
            "(arriving from the road)."
        )
    else:
        confidence = ConfidenceLevel.LOW
        reasoning = (
            "No FRONT label, ROAD candidate, STREET/ACCESS/MAIN-ENTRY/GATE text was found on this "
            "page. Front edge defaulted deterministically to the polygon's first boundary edge — "
            "this is a fallback, not an assumption that any particular page direction is the front. "
            "Treat front/rear/left/right with LOW confidence until orientation evidence is available."
        )

    return FrontSideResolution(
        front_edges=[front_edge],
        rear_edges=[rear_edge],
        left_edges=left_edges,
        right_edges=right_edges,
        confidence=confidence,
        reasoning=reasoning,
        evidence_level=evidence_level,
    )


__all__ = ["FrontSideResolution", "resolve_front_side"]
