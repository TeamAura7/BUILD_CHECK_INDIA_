"""
Road + access evidence resolution.

Picks the best road candidate for the plot's page (if any) and collects
text evidence for gate/main-entry/access mentions, which `front_side.py`
uses as fallback signals when there's no explicit road candidate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from backend.schemas.candidates import RoadCandidate
from backend.schemas.evidence import TextEvidence, ValueField
from backend.schemas.geometry import BoundingBox, CoordinateSpace, NormalizedGeometry, Polygon
from backend.schemas.extraction import ExtractionResult
from backend.spatial_reasoning import geometry_utils as geo

_ACCESS_PATTERNS = {
    "road": re.compile(r"\broad\b", re.I),
    "street": re.compile(r"\bstreet\b", re.I),
    "access": re.compile(r"\baccess\b", re.I),
    "main_entry": re.compile(r"\bmain\s*entry\b|\bmain\s*entrance\b|\bentry\b|\bentrance\b", re.I),
    "gate": re.compile(r"\bgate\b", re.I),
    "front": re.compile(r"\bfront\b", re.I),
}


@dataclass
class AccessEvidence:
    kind: str  # matches keys of _ACCESS_PATTERNS
    text_evidence: TextEvidence


def _bbox_nested(inner: BoundingBox, outer: BoundingBox, tol: float) -> bool:
    return (
        inner.min_x >= outer.min_x - tol
        and inner.max_x <= outer.max_x + tol
        and inner.min_y >= outer.min_y - tol
        and inner.max_y <= outer.max_y + tol
    )


def infer_road_polygon_by_adjacency(
    plot_polygon: Polygon,
    candidates: list[Polygon],
    max_gap_ratio: float = 0.12,
) -> Optional[Polygon]:
    """
    Identify which candidate polygon is the road using PURE GEOMETRY --
    adjacency to the plot boundary plus an elongated, strip-like shape --
    instead of a layer name or a "ROAD" text label.

    This matters most for DXF: the vector geometry is exact, so leaning on
    how a particular drafter happened to name a layer (or whether they
    bothered to write "ROAD" as text at all) throws away the one advantage
    DXF has over a scanned/rasterized PDF. A road, structurally, is: (a) a
    polygon that sits just outside the plot boundary (not nested inside
    it -- that's a room or the building), and (b) long and thin, running
    alongside one edge of the plot rather than being some other shape
    entirely.

    Both conditions have to be checked together: adjacency alone would
    also match a neighbouring plot's compound wall, and elongation alone
    would also match a long interior corridor. Scored, not just
    thresholded, so the single best-fitting candidate wins when several
    pass both checks (e.g. a corner plot with polygons adjacent on two
    sides).

    `max_gap_ratio` is relative to the plot's own bounding-box span (not
    an absolute distance), so this works unscaled -- it never needs a
    points-per-metre factor, which is one of the two page-space quantities
    (the identity of which polygon is the road) that should hold regardless
    of scale; only the OTHER one (an actual metric road WIDTH) needs scale
    and is deliberately not attempted here (see best_road_candidate's
    text-anchor fallback docstring: identity and width are different
    claims with different evidence bars).

    Returns the winning candidate Polygon, or None if nothing plausible
    (by shape+adjacency) is adjacent to the plot at all -- callers should
    then fall through to text-based evidence rather than guess.
    """
    plot_bbox = plot_polygon.bounding_box
    plot_edges = geo.polygon_edges(plot_polygon)
    plot_span = max(plot_bbox.width, plot_bbox.height)
    if plot_span <= 0:
        return None
    max_gap = plot_span * max_gap_ratio

    best: Optional[Polygon] = None
    best_score = 0.0
    for cand in candidates:
        if cand is plot_polygon:
            continue
        cand_bbox = cand.bounding_box
        # A road sits outside/along the plot boundary -- a polygon mostly
        # inside the plot's own bounding box is a room/building/courtyard,
        # not a road, regardless of its shape.
        if _bbox_nested(cand_bbox, plot_bbox, tol=0.02 * plot_span):
            continue
        # Reject anything that itself fully contains the plot (a sheet
        # border, title block frame, or page outline) -- these are not
        # roads either, however elongated their bounding box may be.
        if _bbox_nested(plot_bbox, cand_bbox, tol=0.02 * plot_span):
            continue

        gap = geo.polygon_to_edges_distance(cand, plot_edges)
        if gap > max_gap:
            continue

        elongation = geo.aspect_ratio(cand_bbox)  # 1.0 = square, higher = strip-like
        if elongation == float("inf"):
            continue
        # Score rewards being both close (gap near 0) and strip-shaped;
        # neither alone is sufficient (see docstring).
        proximity = 1.0 - (gap / max_gap)  # in (0, 1]
        score = proximity * min(elongation, 8.0)  # cap so a sliver line doesn't dominate on shape alone
        if score > best_score:
            best_score = score
            best = cand

    return best


def collect_access_evidence(text_evidence: list[TextEvidence], page: Optional[int]) -> list[AccessEvidence]:
    found: list[AccessEvidence] = []
    for t in text_evidence:
        if page is not None and t.page != page:
            continue
        for kind, pattern in _ACCESS_PATTERNS.items():
            if pattern.search(t.raw_text or ""):
                found.append(AccessEvidence(kind=kind, text_evidence=t))
    return found


def best_road_candidate(
    extraction: ExtractionResult, page: Optional[int]
) -> Optional[RoadCandidate]:
    candidates = [
        r for r in extraction.road_candidates if r.geometry and r.geometry.bounding_box is not None
    ]
    if page is not None:
        candidates = [r for r in candidates if r.geometry.source_page == page]
    if not candidates:
        # Fallback for common site-plan drawings where ROAD is printed inside
        # a large road area but OpenCV does not classify that area as a
        # long/thin rectangle. The text anchor is sufficient for front-side
        # orientation; it is deliberately NOT used to estimate road width.
        road_texts = [t for t in extraction.text_evidence if t.page == page and re.search(r"\broad\b", t.raw_text or "", re.I) and t.bounding_box]
        if road_texts:
            t = min(road_texts, key=lambda x: x.bounding_box.width * x.bounding_box.height)
            b = t.bounding_box
            pad = max(b.width, b.height) * 1.5
            bbox = BoundingBox(
                min_x=b.min_x - pad, min_y=b.min_y - pad,
                max_x=b.max_x + pad, max_y=b.max_y + pad,
            )
            return RoadCandidate(
                id=f"text-road-p{page}",
                geometry=NormalizedGeometry(
                    coordinate_space=CoordinateSpace.PAGE_POINTS,
                    bounding_box=bbox,
                    points_per_metre=1.0,
                    source_page=page,
                ),
                confidence_note="ROAD text anchor used only for front-side orientation; no road width inferred.",
                width=ValueField[float].missing("Road width is not explicitly dimensioned; ROAD text is used only for front-side orientation."),
                name_or_label=t.raw_text,
            )
        return None
    # Prefer the widest labeled candidate; road width is only ever a
    # rough page-space proxy here (real conversion happens once scale is known).
    candidates.sort(
        key=lambda r: (r.name_or_label is not None, r.geometry.bounding_box.width * r.geometry.bounding_box.height),
        reverse=True,
    )
    return candidates[0]


__all__ = ["AccessEvidence", "collect_access_evidence", "best_road_candidate"]
