"""
Fragmented-geometry building reconstruction for DXF files.

Many real-world DXFs never capture a building footprint as a single
closed polygon: walls are exported as disconnected `LINE` entities, or
as an open (unclosed) `LWPOLYLINE`/`POLYLINE`, because of how the source
CAD tool exported the drawing or because the file was itself produced by
vectorizing/tracing a scanned sheet. `dxf_extractor.py`'s primary
building-detection path only ever considers genuinely CLOSED polygons
(`_pick_building_polygon`), so a DXF like this previously fell straight
through to the Vision-render fallback (`_vision_fallback_regions`) even
though the real wall line-work is sitting right there in the vector data
-- Vision only had to be asked because nothing tried to stitch it back
together first.

This module is that missing deterministic step, tried BEFORE the Vision
fallback: it never rasterizes and never asks a model anything, it only
reasons about the DXF's own exact coordinates -- strictly more trustworthy
than a Vision bounding-box guess, and it still works when Vision is
disabled entirely.

Pipeline (never invents geometry that isn't backed by a real segment in
the file; returns None -- meaning "stay MISSING" -- rather than guess):

  1. strip segments on annotation-ish layers (dimension lines, hatching,
     text leaders, centerlines, grids, MEP/furniture/symbol layers) --
     these are not wall/boundary line-work and stitching them in would
     fabricate a fake outline.
  2. snap nearby segment endpoints together (handles the sub-drawing-unit
     gaps typical of tracing/export noise, without inventing a corner
     that isn't approximately where the file's own points already are).
  3. merge consecutive collinear segments (a wall exported as several
     collinear pieces is still one wall, not several corners).
  4. build a connected-components graph over the snapped endpoints.
  5. detect closed loops via a cycle-space basis (spanning tree +
     one fundamental cycle per non-tree edge) -- the standard technique
     for enumerating a planar graph's independent cycles without an
     external graph library.
  6. rank closed loops by plot-containment, area plausibility and aspect
     ratio, and return the best one -- or None if nothing plausible
     survives, which callers must treat as "reconstruction found
     nothing", never as an invented footprint.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from backend.schemas.geometry import BoundingBox, Point, Polygon
from backend.spatial_reasoning import geometry_utils as geo

# Layers that carry annotation/reference/services line-work rather than
# real wall or boundary geometry. A segment on one of these must never be
# stitched into a reconstructed outline -- a dimension line or a hatch
# stroke is not a wall, and letting one contribute a "corner" would
# fabricate geometry the drawing never actually contained.
ANNOTATION_LAYER_RE = re.compile(
    r"DIM(ENSION)?S?\b|\bTEXT\b|ANNOT|HATCH|CENTER\s*-?\s*LINE|CENTRELINE|\bC[-_ ]?LINE\b|"
    r"\bGRID\b|\bAXIS\b|FURNITURE|\bSYMBOL\b|TITLE\s*BLOCK|\bBORDER\b|\bNOTE\b|LEADER|"
    r"\bELEC(TRICAL)?\b|\bPLUMB(ING)?\b|\bHVAC\b|HATCHING",
    re.I,
)

DEFAULT_SNAP_TOLERANCE = 0.05
_COLLINEAR_ANGLE_TOL_DEG = 3.0
_MAX_COMPONENT_EDGES = 500  # bail out of a pathologically large component rather than risk a hang
_MIN_PLOT_AREA_FRACTION = 0.03
_MAX_PLOT_AREA_FRACTION = 0.92
_MAX_ASPECT_RATIO = 10.0

Segment = tuple[tuple[float, float], tuple[float, float]]
LayeredSegment = tuple[str, tuple[float, float], tuple[float, float]]


@dataclass
class _Node:
    id: int
    x: float
    y: float


@dataclass
class ReconstructionResult:
    polygon: Polygon
    score: float
    reason: str
    segments_used: int
    candidates_considered: int


def _snap_endpoints(segments: list[Segment], tol: float) -> tuple[list[_Node], list[tuple[int, int]]]:
    """Cluster segment endpoints within `tol` of each other into shared
    graph nodes, via grid bucketing (not a full O(n^2) nearest-point
    search, so this stays tractable for a few thousand fragments)."""
    nodes: list[_Node] = []
    buckets: dict[tuple[int, int], list[int]] = {}
    cell = max(tol, 1e-9)

    def _bucket_key(x: float, y: float) -> tuple[int, int]:
        return (int(math.floor(x / cell)), int(math.floor(y / cell)))

    def _find_or_create(x: float, y: float) -> int:
        bx, by = _bucket_key(x, y)
        best_id: Optional[int] = None
        best_d = tol
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for nid in buckets.get((bx + dx, by + dy), ()):
                    n = nodes[nid]
                    d = math.hypot(n.x - x, n.y - y)
                    if d <= best_d:
                        best_d = d
                        best_id = nid
        if best_id is not None:
            return best_id
        nid = len(nodes)
        nodes.append(_Node(id=nid, x=x, y=y))
        buckets.setdefault((bx, by), []).append(nid)
        return nid

    edges: set[tuple[int, int]] = set()
    for (x1, y1), (x2, y2) in segments:
        if math.hypot(x2 - x1, y2 - y1) < 1e-9:
            continue
        a = _find_or_create(x1, y1)
        b = _find_or_create(x2, y2)
        if a == b:
            continue
        edges.add((min(a, b), max(a, b)))
    return nodes, sorted(edges)


def _merge_collinear(
    nodes: list[_Node], edges: list[tuple[int, int]], angle_tol_deg: float
) -> list[tuple[int, int]]:
    """Contract degree-2 nodes whose two incident edges are collinear:
    A-B-C becomes a single A-C edge when B is not a real corner, just a
    point where one straight wall happened to be split into two segments
    during export/tracing. Bounded iteration count -- if convergence
    hasn't happened by then the remaining graph is used as-is rather than
    risking unbounded work on a pathological input."""
    adj: dict[int, set[int]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    max_iterations = max(len(nodes), 1) * 2
    for _ in range(max_iterations):
        contracted = False
        for nid, neighbors in list(adj.items()):
            if len(neighbors) != 2:
                continue
            a, c = tuple(neighbors)
            if a not in adj or c not in adj:
                continue
            na, nb, nc = nodes[a], nodes[nid], nodes[c]
            d1 = math.degrees(math.atan2(nb.y - na.y, nb.x - na.x))
            d2 = math.degrees(math.atan2(nc.y - nb.y, nc.x - nb.x))
            diff = abs((d1 - d2 + 180.0) % 360.0 - 180.0)
            if diff > angle_tol_deg and abs(diff - 180.0) > angle_tol_deg:
                continue
            adj[a].discard(nid)
            adj[c].discard(nid)
            if a != c:
                adj[a].add(c)
                adj[c].add(a)
            del adj[nid]
            contracted = True
        if not contracted:
            break

    merged: set[tuple[int, int]] = set()
    for nid, neighbors in adj.items():
        for other in neighbors:
            merged.add((min(nid, other), max(nid, other)))
    return sorted(merged)


def _connected_components(edges: list[tuple[int, int]]) -> list[set[int]]:
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen: set[int] = set()
    components: list[set[int]] = []
    for start in adj:
        if start in seen:
            continue
        stack = [start]
        comp: set[int] = set()
        while stack:
            cur = stack.pop()
            if cur in comp:
                continue
            comp.add(cur)
            seen.add(cur)
            for nb in adj.get(cur, ()):
                if nb not in comp:
                    stack.append(nb)
        components.append(comp)
    return components


def _fundamental_cycles(edges: list[tuple[int, int]], component: set[int]) -> list[list[int]]:
    """Spanning-tree + one fundamental cycle per non-tree edge: the
    standard cycle-space-basis technique for enumerating a graph's
    independent cycles without an external graph library. For a roughly
    planar architectural line drawing, these fundamental cycles are
    exactly the closed loops (room outlines, the building outline, the
    plot boundary) the reconstruction is looking for."""
    comp_edges = [(a, b) for a, b in edges if a in component and b in component]
    if len(comp_edges) > _MAX_COMPONENT_EDGES:
        return []

    adj: dict[int, list[int]] = {nid: [] for nid in component}
    for a, b in comp_edges:
        adj[a].append(b)
        adj[b].append(a)

    root = next(iter(component))
    parent: dict[int, Optional[int]] = {root: None}
    tree_edges: set[tuple[int, int]] = set()
    queue = [root]
    while queue:
        cur = queue.pop(0)
        for nb in adj[cur]:
            if nb not in parent:
                parent[nb] = cur
                tree_edges.add((min(cur, nb), max(cur, nb)))
                queue.append(nb)

    def _path_to_root(n: int) -> list[int]:
        path = [n]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        return path

    cycles: list[list[int]] = []
    for a, b in comp_edges:
        key = (min(a, b), max(a, b))
        if key in tree_edges:
            continue
        pa, pb = _path_to_root(a), _path_to_root(b)
        set_pb = set(pb)
        lca = next((n for n in pa if n in set_pb), None)
        if lca is None:
            continue
        cycle = []
        for n in pa:
            cycle.append(n)
            if n == lca:
                break
        idx = pb.index(lca)
        cycle.extend(reversed(pb[:idx]))
        if len(cycle) >= 3:
            cycles.append(cycle)
    return cycles


def _make_polygon(points_xy: list[tuple[float, float]]) -> Optional[Polygon]:
    cleaned: list[tuple[float, float]] = []
    for x, y in points_xy:
        if cleaned and math.hypot(x - cleaned[-1][0], y - cleaned[-1][1]) < 1e-9:
            continue
        cleaned.append((x, y))
    if len(cleaned) >= 2 and math.hypot(cleaned[0][0] - cleaned[-1][0], cleaned[0][1] - cleaned[-1][1]) < 1e-9:
        cleaned.pop()
    if len(cleaned) < 3:
        return None
    poly = Polygon(points=[Point(x=x, y=y) for x, y in cleaned])
    if poly.area < 1e-6:
        return None
    return poly


def _is_simple_polygon(poly: Polygon) -> bool:
    """Reject a topologically closed but self-intersecting ring (a
    figure-eight/bowtie) before it is scored at all.

    Cycle detection in `_fundamental_cycles` only guarantees a closed
    sequence of nodes -- it says nothing about whether that sequence
    crosses itself. A bowtie can still produce a nonzero shoelace area
    (`Polygon.area` has no notion of self-intersection), which would
    otherwise let a geometrically invalid "building" survive purely
    because a number came out of the area formula. Reuses
    `geometry_utils.is_valid_polygon` (already the codebase's one
    self-intersection check, used the same way in
    `candidate_geometry.py`) rather than a second implementation;
    vertex-count/area/perimeter-ratio floors are left at their permissive
    defaults since `_make_polygon` and this function's own checks already
    cover those.
    """
    return geo.is_valid_polygon(poly)


def _score_candidate(poly: Polygon, plot_polygon: Optional[Polygon]) -> Optional[float]:
    if not _is_simple_polygon(poly):
        return None
    bbox = poly.bounding_box
    ar = geo.aspect_ratio(bbox)
    if not math.isfinite(ar) or ar > _MAX_ASPECT_RATIO:
        return None
    score = 1.0 / (1.0 + ar)

    if plot_polygon is not None and plot_polygon.area > 0:
        frac = poly.area / plot_polygon.area
        if frac < _MIN_PLOT_AREA_FRACTION or frac > _MAX_PLOT_AREA_FRACTION:
            return None
        centroid = bbox.center
        if not geo.point_in_polygon(centroid, plot_polygon):
            return None
        # A mild extra preference for plausible mid-range coverage -- not
        # a hard requirement, since real coverage norms vary widely.
        score += 1.0 - abs(frac - 0.5)
    return score


def _cap_segments_for_reconstruction(structural: list[Segment]) -> list[Segment]:
    """Bound the segment count reaching `_snap_endpoints`/`_merge_collinear`.

    `_snap_endpoints` is grid-bucketed and stays roughly linear in segment
    count, but `_merge_collinear` iteratively contracts degree-2 nodes and
    is worst-case O(nodes^2) -- the same unbounded-graph-stage bug class
    already found and fixed for `site_graph.build_site_graph`
    (`backend.cv_extraction.dxf_extractor._build_role_inference_graph`,
    which caps to `settings.max_dxf_role_inference_polygons` largest-area
    polygons before that O(n^2) build runs). `_fundamental_cycles` already
    caps itself via `_MAX_COMPONENT_EDGES`, but only after these two earlier
    stages have already paid the full uncapped cost.

    Mirrors that fix here: cap to the longest `settings.
    max_dxf_reconstruction_segments` segments. A real wall or boundary run
    is always among the longer segments on a sheet; short segments are
    disproportionately tracing noise or duplicate fragments, so this loses
    essentially nothing relevant while making reconstruction bounded.
    """
    from backend.config import get_settings

    limit = get_settings().max_dxf_reconstruction_segments
    if len(structural) <= limit:
        return structural
    return sorted(
        structural,
        key=lambda seg: math.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]),
        reverse=True,
    )[:limit]


def reconstruct_building_polygon(
    segments: list[LayeredSegment],
    plot_polygon: Optional[Polygon],
    snap_tolerance: float = DEFAULT_SNAP_TOLERANCE,
) -> Optional[ReconstructionResult]:
    """Attempt to reconstruct a single closed building-footprint polygon
    from fragmented open line-work (`segments`: (layer, start, end)
    triples, drawing units already scaled to metres by the caller).

    Returns None when nothing plausible was found -- a caller must treat
    that as "stay MISSING", never fall back to inventing a footprint.
    """
    structural: list[Segment] = [
        (start, end) for layer, start, end in segments if not ANNOTATION_LAYER_RE.search(layer or "")
    ]
    if not structural:
        return None
    structural = _cap_segments_for_reconstruction(structural)

    nodes, raw_edges = _snap_endpoints(structural, snap_tolerance)
    if not raw_edges:
        return None
    edges = _merge_collinear(nodes, raw_edges, _COLLINEAR_ANGLE_TOL_DEG)
    if not edges:
        return None

    candidates: list[tuple[Polygon, float]] = []
    for component in _connected_components(edges):
        if len(component) < 3:
            continue
        for cycle in _fundamental_cycles(edges, component):
            pts = [(nodes[n].x, nodes[n].y) for n in cycle]
            poly = _make_polygon(pts)
            if poly is None:
                continue
            score = _score_candidate(poly, plot_polygon)
            if score is not None:
                candidates.append((poly, score))

    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[1], reverse=True)
    best_poly, best_score = candidates[0]
    return ReconstructionResult(
        polygon=best_poly,
        score=best_score,
        reason=(
            f"Reconstructed from {len(structural)} fragmented open line-work segment(s) on non-annotation "
            f"layers (endpoints snapped at {snap_tolerance:g} drawing-unit tolerance, collinear runs merged, "
            f"best of {len(candidates)} closed loop(s) found via cycle detection; score={best_score:.3f})."
        ),
        segments_used=len(structural),
        candidates_considered=len(candidates),
    )


__all__ = [
    "ANNOTATION_LAYER_RE",
    "DEFAULT_SNAP_TOLERANCE",
    "ReconstructionResult",
    "reconstruct_building_polygon",
]
