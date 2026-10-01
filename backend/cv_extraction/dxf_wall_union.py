"""
Area-based wall-union building reconstruction for DXF files.

`dxf_reconstruction.py`'s closed-loop/cycle-detection approach requires the
wall line-work to form an actual topologically CLOSED loop once endpoints
are snapped together -- it is exact and trustworthy when that holds, but
several very common real-drawing patterns break it outright rather than
just degrading it:

  - a door/window/opening gap in a wall centerline trace is far larger than
    any reasonable endpoint-snap tolerance, so the graph never closes
    across it at all -- the fundamental-cycle search then either finds no
    cycle through that stretch of wall, or "closes" through some unrelated,
    much smaller interior loop instead;
  - walls drawn as double parallel lines (inner/outer wall face) with
    mitred or unclosed corners can leave the true outer envelope
    disconnected from itself even though every individual wall is present;
  - overlapping/duplicate line-work (a wall re-traced twice, hatching leaking
    into the "structural" layer) creates extra graph edges and many small
    spurious candidate cycles for the scoring heuristic to be fooled by.

This module is an ADDITIONAL fallback, tried only when
`reconstruct_building_polygon` already found nothing plausible -- it never
replaces or retunes that path. Instead of reasoning about the wall network
as a 1-D graph that must close, it reasons about it as a 2-D AREA:

  1. estimate wall thickness from THIS document's own geometry (parallel
     wall-face line pairs, when present -- never a hardcoded constant);
  2. find document-relative door/window-opening-sized gaps between
     mutually-nearest dangling wall endpoints, and add an explicit
     bridging segment for each one;
  3. buffer every wall segment (real and bridging) outward by half the
     estimated thickness into a thin rectangle;
  4. union all of those thin areas together (this is what "closes" a
     corner, a double-traced wall, or a bridged opening -- two
     overlapping/touching thin areas merge into one regardless of
     whether their source endpoints snap exactly);
  5. take the resulting area's outer boundary as the building footprint.

Never invents geometry unsupported by the file's own segments: every
number used (thickness, closing distance) is derived from this document's
own geometry, with a small, clearly-labelled drawing-scale-relative
fallback only when no such evidence exists at all. Returns None -- meaning
"this method also found nothing plausible" -- rather than guessing,
exactly like `dxf_reconstruction.reconstruct_building_polygon`.
"""

from __future__ import annotations

import math
from typing import Optional

from shapely import geometry as shp_geom
from shapely.ops import unary_union

from backend.cv_extraction.dxf_reconstruction import (
    ANNOTATION_LAYER_RE,
    DEFAULT_SNAP_TOLERANCE,
    ReconstructionResult,
    Segment,
    _snap_endpoints,
)
from backend.schemas.geometry import Point, Polygon
from backend.spatial_reasoning import geometry_utils as geo

# Two segments are considered a candidate "double wall face" pair only when
# their orientations agree within this tolerance -- generous enough for
# tracing/export noise, tight enough not to pair unrelated segments.
_PARALLEL_ANGLE_TOL_DEG = 5.0

# A thickness estimate is only trusted once this many independent parallel-
# pair samples agree (the median of a handful of coincidental pairings
# drawn from a large pool of unrelated segments is not reliable) -- below
# that, fall back to a drawing-scale-relative default.
_MIN_THICKNESS_SAMPLES = 3

# A parallel pair only counts as evidence of wall thickness when the two
# segments actually run alongside each other for a meaningful fraction of
# their own length -- otherwise two unrelated near-parallel segments that
# merely pass close to one another somewhere would masquerade as a wall's
# two faces.
_MIN_OVERLAP_FRACTION = 0.3

# A "wall" thicker than this fraction of the whole structural drawing's own
# diagonal is not a wall -- guards the parallel-pair search against locking
# onto two edges of a large room or the plot boundary itself.
_MAX_THICKNESS_FRACTION_OF_SCALE = 0.05

# When no parallel-wall-face evidence exists at all (single-centerline
# wall drawings, common in simplified/schematic DXFs), fall back to a
# thickness this small a fraction of the drawing's own scale -- just
# enough for the buffer/union machinery to produce valid polygons, not a
# claim about the real wall's true thickness.
_FALLBACK_THICKNESS_FRACTION_OF_SCALE = 0.002

# A door/window opening is expected to be a small fraction of the overall
# wall network's own scale -- a "gap" wider than this is more likely two
# genuinely separate structures than one opening, and is left unclosed.
_MAX_CLOSING_FRACTION_OF_SCALE = 0.25

_MIN_PLOT_AREA_FRACTION = 0.03
_MAX_PLOT_AREA_FRACTION = 0.92
_MAX_ASPECT_RATIO = 10.0

# Bail out of a pathologically large segment population rather than risk a
# timeout -- mirrors `dxf_reconstruction._MAX_COMPONENT_EDGES`'s own
# reasoning. The thickness/gap searches below are angle-bucketed (not
# naively O(n^2) over the whole population), but a real fragmented-trace
# DXF can still put thousands of segments in the very same angle bucket
# (e.g. every dash of an axis-aligned wall), and Shapely's buffer+union
# cost also grows with segment count -- this is the fallback path tried
# only when cycle-detection already failed, so bailing out here still
# leaves the coarser bounding-envelope fallback available.
_MAX_SEGMENTS_FOR_WALL_UNION = 400


def _line_through_segment(seg: Segment) -> tuple[tuple[float, float], tuple[float, float]]:
    (x1, y1), (x2, y2) = seg
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1e-9:
        return (x1, y1), (1.0, 0.0)
    return (x1, y1), ((x2 - x1) / length, (y2 - y1) / length)


def _perp_distance(pt: tuple[float, float], anchor: tuple[float, float], direction: tuple[float, float]) -> float:
    dx, dy = pt[0] - anchor[0], pt[1] - anchor[1]
    nx, ny = -direction[1], direction[0]
    return abs(dx * nx + dy * ny)


def _project(pt: tuple[float, float], anchor: tuple[float, float], direction: tuple[float, float]) -> float:
    dx, dy = pt[0] - anchor[0], pt[1] - anchor[1]
    return dx * direction[0] + dy * direction[1]


def _segment_angle_deg(seg: Segment) -> float:
    (x1, y1), (x2, y2) = seg
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def _overlap_fraction(
    si: Segment, sj: Segment, anchor: tuple[float, float], direction: tuple[float, float]
) -> float:
    pa = [_project(p, anchor, direction) for p in si]
    pb = [_project(p, anchor, direction) for p in sj]
    a0, a1 = min(pa), max(pa)
    b0, b1 = min(pb), max(pb)
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    shorter = min(a1 - a0, b1 - b0)
    if shorter <= 1e-9:
        return 0.0
    return overlap / shorter


def _bbox_diagonal(segments: list[Segment]) -> float:
    xs = [p[0] for seg in segments for p in seg]
    ys = [p[1] for seg in segments for p in seg]
    if not xs:
        return 0.0
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def estimate_wall_thickness(segments: list[Segment], scale: float) -> float:
    """Infer wall thickness from THIS document's own geometry: real
    architectural line-work very often draws a wall as two parallel lines
    (inner and outer face) a constant distance apart -- that distance IS
    the wall thickness. Finds near-parallel, meaningfully-overlapping
    segment pairs and takes the MEDIAN of their perpendicular separation
    (robust to a few unrelated near-parallel segments elsewhere in the
    drawing). Falls back to a small drawing-scale-relative default only
    when no such pairing exists at all -- never a fixed absolute value.
    """
    thickness, _paired = _estimate_wall_thickness_with_pairs(segments, scale)
    return thickness


def _estimate_wall_thickness_with_pairs(
    segments: list[Segment], scale: float
) -> tuple[float, set[int]]:
    """As `estimate_wall_thickness`, but also returns the indices of
    segments that were actually paired as a double wall face -- so the
    caller can avoid buffering those by the FULL estimated thickness on
    top of the real gap already between them (which would double-count
    the wall's own thickness: the two face lines already ARE its
    thickness, unlike a lone centerline segment which needs the buffer to
    represent the wall's thickness at all)."""
    if scale <= 0:
        return 0.0, set()
    max_thickness = scale * _MAX_THICKNESS_FRACTION_OF_SCALE
    angles = [_segment_angle_deg(s) for s in segments]
    buckets: dict[int, list[int]] = {}
    for i, a in enumerate(angles):
        buckets.setdefault(round(a / _PARALLEL_ANGLE_TOL_DEG), []).append(i)

    samples: list[float] = []
    pairs: list[tuple[int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for key, idxs in buckets.items():
        for neighbor_key in (key - 1, key, key + 1):
            for j in buckets.get(neighbor_key, ()):
                for i in idxs:
                    if i >= j:
                        continue
                    pair = (i, j)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    diff = abs(angles[i] - angles[j])
                    diff = min(diff, 180.0 - diff)
                    if diff > _PARALLEL_ANGLE_TOL_DEG:
                        continue
                    anchor, direction = _line_through_segment(segments[i])
                    d = _perp_distance(segments[j][0], anchor, direction)
                    if d <= 1e-9 or d > max_thickness:
                        continue
                    if _overlap_fraction(segments[i], segments[j], anchor, direction) < _MIN_OVERLAP_FRACTION:
                        continue
                    samples.append(d)
                    pairs.append(pair)

    if len(samples) >= _MIN_THICKNESS_SAMPLES:
        thickness = sorted(samples)[len(samples) // 2]
        # Only the pairs whose OWN separation agrees with the accepted
        # (median) thickness are treated as real double-face pairs -- a
        # pair whose gap was merely close enough to be sampled but is
        # actually a different, unrelated pair of nearby parallel lines
        # should still get the normal centerline buffering.
        agreeing = {
            idx
            for (i, j), d in zip(pairs, samples)
            if abs(d - thickness) <= max(1e-6, thickness * 0.25)
            for idx in (i, j)
        }
        return thickness, agreeing
    return max(scale * _FALLBACK_THICKNESS_FRACTION_OF_SCALE, 1e-6), set()


def _dangling_node_ids(nodes: list, edges: list[tuple[int, int]]) -> dict[int, int]:
    degree: dict[int, int] = {}
    for a, b in edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    return {nid: deg for nid, deg in degree.items() if deg == 1}


def _dangling_edge_direction(nid: int, edges: list[tuple[int, int]], nodes: list) -> tuple[float, float]:
    """Direction the wall EXTRAPOLATES past its dangling end `nid` (i.e.
    from the wall's other endpoint, through `nid`, and onward into the
    gap) -- not the direction back into the existing wall."""
    for a, b in edges:
        if a == nid or b == nid:
            other = b if a == nid else a
            return _line_through_segment(((nodes[other].x, nodes[other].y), (nodes[nid].x, nodes[nid].y)))[1]
    return (1.0, 0.0)


def find_gap_bridge_segments(
    segments: list[Segment], snap_tolerance: float, scale: float
) -> list[Segment]:
    """Find door/window/opening-sized gaps from THIS document's own
    dangling wall endpoints, and return an explicit bridging segment for
    each one, to be buffered and unioned alongside the real wall segments.

    A real gap leaves two wall endpoints that don't snap together but
    plausibly continue each other's direction (roughly ahead of, and not
    far to the side of, the dangling end's own wall direction) -- this
    pairs each dangling endpoint with its single best (nearest, most
    plausible) partner and only keeps a pair once BOTH ends agree the
    other is their best match, which is what makes this safe to apply to
    every gap found rather than picking just one: two genuinely separate,
    unrelated dangling ends elsewhere in the drawing essentially never
    happen to be mutually-nearest to each other with correctly opposed
    directions.

    Deliberately inserts real bridging LINE geometry rather than
    morphologically dilating-then-eroding the whole unioned wall area:
    the latter also dilates the building's own INTERIOR (the hollow space
    the walls enclose), and eroding back afterward can eat away much more
    than just the gap when that interior is larger than the closing
    distance -- inserting a small bridge at the specific identified gap
    has no such side effect on the rest of the shape.
    """
    if scale <= 0:
        return []
    nodes, edges = _snap_endpoints(segments, snap_tolerance)
    dangling = _dangling_node_ids(nodes, edges)
    if len(dangling) < 2:
        return []
    max_gap = scale * _MAX_CLOSING_FRACTION_OF_SCALE
    dangling_ids = list(dangling.keys())
    directions = {nid: _dangling_edge_direction(nid, edges, nodes) for nid in dangling_ids}

    def _best_partner(a: int) -> Optional[int]:
        na = nodes[a]
        da = directions[a]
        best_id: Optional[int] = None
        best_gap: Optional[float] = None
        for b in dangling_ids:
            if a == b:
                continue
            nb = nodes[b]
            gap = math.hypot(nb.x - na.x, nb.y - na.y)
            if gap <= snap_tolerance or gap > max_gap:
                continue
            forward = _project((nb.x, nb.y), (na.x, na.y), da)
            if forward <= 0:
                continue
            lateral = _perp_distance((nb.x, nb.y), (na.x, na.y), da)
            if lateral > gap * 0.5:
                continue
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_id = b
        return best_id

    best_partner = {nid: _best_partner(nid) for nid in dangling_ids}
    bridges: list[Segment] = []
    seen: set[tuple[int, int]] = set()
    for a, b in best_partner.items():
        if b is None or best_partner.get(b) != a:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        bridges.append(((nodes[a].x, nodes[a].y), (nodes[b].x, nodes[b].y)))
    return bridges


def _score_candidate(poly: Polygon, plot_polygon: Optional[Polygon]) -> Optional[float]:
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
        score += 1.0 - abs(frac - 0.5)
    return score


def _point_touches_another_segment_span(
    pt: tuple[float, float], segments: list[Segment], tol: float
) -> bool:
    """True when `pt` lies on (within `tol` of) some segment's own span --
    a T-junction touch point, e.g. an interior partition wall's end
    meeting the MIDDLE of an exterior wall rather than that wall's own
    endpoint. `_snap_endpoints` never creates a shared graph node for
    this case (it only clusters endpoint-to-endpoint), so such a point
    would otherwise look exactly like an unresolved dangling gap."""
    for seg in segments:
        (x1, y1), (x2, y2) = seg
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1e-9:
            continue
        anchor, direction = (x1, y1), ((x2 - x1) / length, (y2 - y1) / length)
        if _perp_distance(pt, anchor, direction) > tol:
            continue
        proj = _project(pt, anchor, direction)
        if tol <= proj <= length - tol:
            return True
    return False


def _largest_polygon(geom) -> Optional[shp_geom.base.BaseGeometry]:
    if geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)
    return None


def reconstruct_building_via_wall_union(
    segments: list[tuple[str, tuple[float, float], tuple[float, float]]],
    plot_polygon: Optional[Polygon],
    snap_tolerance: float = DEFAULT_SNAP_TOLERANCE,
) -> Optional[ReconstructionResult]:
    """Reconstruct a building footprint by buffering wall segments into
    thin areas, unioning them, and taking the union's outer boundary --
    see this module's own docstring for the full rationale. Intended as
    an ADDITIONAL fallback tried only after `dxf_reconstruction.
    reconstruct_building_polygon` already returned None.
    """
    structural: list[Segment] = [
        (start, end) for layer, start, end in segments if not ANNOTATION_LAYER_RE.search(layer or "")
    ]
    structural = [s for s in structural if math.hypot(s[1][0] - s[0][0], s[1][1] - s[0][1]) > 1e-9]
    if len(structural) < 3:
        return None
    if len(structural) > _MAX_SEGMENTS_FOR_WALL_UNION:
        return None

    scale = _bbox_diagonal(structural)
    if scale <= 0:
        return None

    thickness, paired_indices = _estimate_wall_thickness_with_pairs(structural, scale)
    bridges = find_gap_bridge_segments(structural, snap_tolerance, scale)

    # A wall network is only trustworthy to buffer-and-union into a
    # CLOSED footprint once every dangling end ON ITS OWN OUTER ENVELOPE
    # has either already been part of a closed loop or been bridged above
    # -- buffering a network whose outer perimeter is still open (e.g. one
    # whole side of the building missing, with no plausible opening-sized
    # gap to bridge) still produces SOME polygon (the thin band the walls
    # themselves trace), but that shape does not represent a real enclosed
    # footprint at all. Per this project's "wrong is worse than missing"
    # principle, refuse rather than report that thin-band shape as a
    # building. A dangling end INTERIOR to the network (e.g. a partition
    # wall that doesn't reach all the way to the wall it abuts -- an
    # ordinary, harmless architectural detail) is exempt: it never
    # affects the outer boundary regardless of whether it closes.
    nodes, edges = _snap_endpoints(structural, snap_tolerance)
    dangling_ids = set(_dangling_node_ids(nodes, edges).keys())
    if dangling_ids:
        hull = geo.convex_hull([Point(x=n.x, y=n.y) for n in nodes])
        hull_poly = Polygon(points=hull) if len(hull) >= 3 else None
        on_hull_tolerance = max(scale * 0.02, snap_tolerance * 2.0)
        bridged_points = {pt for seg in bridges for pt in seg}
        for nid in dangling_ids:
            pt = (nodes[nid].x, nodes[nid].y)
            if pt in bridged_points:
                continue
            if hull_poly is not None:
                on_outer_envelope = geo.point_to_polygon_boundary_distance(Point(x=pt[0], y=pt[1]), hull_poly) <= on_hull_tolerance
            else:
                on_outer_envelope = True
            if not on_outer_envelope:
                continue
            # A T-junction where this dangling end meets the MIDDLE of
            # another wall (not that wall's own endpoint) is not a gap at
            # all -- `_snap_endpoints` only clusters endpoint-to-endpoint,
            # so the touch point never became a shared graph node, but the
            # wall it touches is still there and still whole.
            if _point_touches_another_segment_span(pt, structural, snap_tolerance):
                continue
            return None

    # A segment identified as one face of a double-line wall already HAS
    # its thickness represented by the gap to its paired face -- buffering
    # it by the full estimated thickness on top of that would double-count
    # the wall's own thickness. Give it just enough buffer to union
    # cleanly (the same small, scale-relative amount used when no
    # thickness evidence exists at all); a lone centerline segment (the
    # common case) still gets the full estimated thickness.
    nominal = max(scale * _FALLBACK_THICKNESS_FRACTION_OF_SCALE, 1e-6)
    buffered = []
    for idx, seg in enumerate(structural):
        line = shp_geom.LineString(seg)
        if line.length <= 1e-9:
            continue
        half = (nominal if idx in paired_indices else thickness) / 2.0
        buffered.append(line.buffer(half, cap_style="flat", join_style="mitre"))
    for seg in bridges:
        line = shp_geom.LineString(seg)
        if line.length > 1e-9:
            buffered.append(line.buffer(thickness / 2.0, cap_style="flat", join_style="mitre"))
    if not buffered:
        return None
    unioned = unary_union(buffered)

    footprint = _largest_polygon(unioned)
    if footprint is None or footprint.is_empty or footprint.area <= 0:
        return None

    exterior_coords = list(footprint.exterior.coords)
    points = [Point(x=x, y=y) for x, y in exterior_coords[:-1]] if len(exterior_coords) > 1 else []
    if len(points) < 3:
        return None
    poly = Polygon(points=points)

    score = _score_candidate(poly, plot_polygon)
    if score is None:
        return None

    return ReconstructionResult(
        polygon=poly,
        score=score,
        reason=(
            f"Reconstructed by buffering {len(structural)} fragmented wall segment(s) into thin areas "
            f"(estimated wall thickness {thickness:.3f} drawing-unit(s), inferred from this document's own "
            f"parallel wall-face line-work where available) plus {len(bridges)} inferred opening-bridge(s) "
            f"(from this document's own dangling wall endpoints, e.g. door/window gaps), unioning them all, "
            f"and taking the union's outer boundary; score={score:.3f}."
        ),
        segments_used=len(structural),
        candidates_considered=1,
    )


__all__ = [
    "estimate_wall_thickness",
    "find_gap_bridge_segments",
    "reconstruct_building_via_wall_union",
]
