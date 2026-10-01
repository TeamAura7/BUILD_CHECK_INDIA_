"""
Generic polygon/segment geometry helpers used throughout spatial reasoning.

Deliberately dependency-free (no shapely) to match the rest of the repo.
Everything here is a pure function of `backend.schemas.geometry` primitives
and works in whatever coordinate space it is given (PAGE_POINTS or
METRIC_PLAN) — callers are responsible for converting to metric plan
space before treating a distance as a real-world metre value.
"""

from __future__ import annotations

import math
from typing import Sequence

from backend.schemas.geometry import BoundingBox, Line, Point, Polygon

Edge = Line


def polygon_edges(polygon: Polygon) -> list[Edge]:
    """Ordered ring edges of a polygon, including the closing edge."""
    pts = polygon.points
    n = len(pts)
    return [Line(start=pts[i], end=pts[(i + 1) % n]) for i in range(n)]


def polygon_perimeter(polygon: Polygon) -> float:
    return sum(e.length for e in polygon_edges(polygon))


def polygon_centroid(polygon: Polygon) -> Point:
    """Area-weighted (shoelace) centroid; falls back to vertex average for degenerate polygons."""
    pts = polygon.points
    n = len(pts)
    a_sum = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        j = (i + 1) % n
        cross = pts[i].x * pts[j].y - pts[j].x * pts[i].y
        a_sum += cross
        cx += (pts[i].x + pts[j].x) * cross
        cy += (pts[i].y + pts[j].y) * cross
    a_sum /= 2.0
    if abs(a_sum) < 1e-9:
        return Point(x=sum(p.x for p in pts) / n, y=sum(p.y for p in pts) / n)
    cx /= 6.0 * a_sum
    cy /= 6.0 * a_sum
    return Point(x=cx, y=cy)


def is_valid_polygon(
    polygon: Polygon,
    min_vertices: int = 3,
    min_area: float = 1e-6,
    max_perimeter_area_ratio: float = 1e9,
) -> bool:
    """
    General polygon validity check — deliberately NOT just
    ``polygon.area / bbox.area`` (rectangularity), which says nothing
    about degenerate/duplicate vertices or self-intersection.

    Rejects:
      - fewer than `min_vertices` distinct points
      - near-zero area (collapsed/degenerate)
      - extreme perimeter^2/area ratio (slivers, spikes, near-duplicate
        back-and-forth vertices)
      - self-intersecting rings (a simple, non-crossing ring is required)
    """
    pts = polygon.points
    if len(pts) < min_vertices:
        return False

    # Duplicate-point collapse: distinct points (within a tiny epsilon).
    distinct: list[Point] = []
    for p in pts:
        if not any(math.hypot(p.x - q.x, p.y - q.y) < 1e-6 for q in distinct):
            distinct.append(p)
    if len(distinct) < min_vertices:
        return False

    area = abs(polygon.area)
    if area < min_area:
        return False

    perimeter = polygon_perimeter(polygon)
    if area > 0:
        ratio = (perimeter * perimeter) / area
        if ratio > max_perimeter_area_ratio:
            return False

    if _polygon_self_intersects(polygon):
        return False

    return True


def _polygon_self_intersects(polygon: Polygon) -> bool:
    edges = polygon_edges(polygon)
    n = len(edges)
    if n < 4:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            # Adjacent edges (including the wrap-around pair) share a
            # vertex by construction — that's not a self-intersection.
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if _segments_intersect(edges[i], edges[j]):
                return True
    return False


def is_page_frame_like(
    bbox: BoundingBox,
    page_width: float,
    page_height: float,
    touch_tolerance: float = 4.0,
    area_fraction_threshold: float = 0.92,
) -> bool:
    """
    True if `bbox` looks like the drawing-sheet/page border rather than a
    real plot/site boundary: it both (a) touches or nearly touches the
    page edge on at least two sides, and (b) covers a very large fraction
    of the total page area. Neither condition alone is sufficient — a
    genuinely large plot that happens to touch one edge should not be
    rejected, and a small candidate pinned to a corner should not be
    rejected either.
    """
    if page_width <= 0 or page_height <= 0:
        return False

    page_area = page_width * page_height
    frac = (bbox.width * bbox.height) / page_area if page_area else 0.0
    if frac < area_fraction_threshold:
        return False

    touches = 0
    if bbox.min_x <= touch_tolerance:
        touches += 1
    if bbox.min_y <= touch_tolerance:
        touches += 1
    if bbox.max_x >= page_width - touch_tolerance:
        touches += 1
    if bbox.max_y >= page_height - touch_tolerance:
        touches += 1

    return touches >= 2


def dedupe_bboxes(
    items: Sequence[tuple[object, BoundingBox]], tolerance: float = 2.0
) -> list[tuple[object, BoundingBox]]:
    """
    Remove near-duplicate geometry (same item type paired with its
    bounding box) — keeps the first occurrence of each cluster of
    bounding boxes within `tolerance` page-points of each other on all
    four sides.
    """
    kept: list[tuple[object, BoundingBox]] = []
    for item, bbox in items:
        is_dup = False
        for _, kept_bbox in kept:
            if (
                abs(bbox.min_x - kept_bbox.min_x) <= tolerance
                and abs(bbox.min_y - kept_bbox.min_y) <= tolerance
                and abs(bbox.max_x - kept_bbox.max_x) <= tolerance
                and abs(bbox.max_y - kept_bbox.max_y) <= tolerance
            ):
                is_dup = True
                break
        if not is_dup:
            kept.append((item, bbox))
    return kept


def convex_hull(points: Sequence[Point]) -> list[Point]:
    """Andrew's monotone-chain convex hull. Returns hull vertices in
    counter-clockwise order, deduplicated. O(n log n), no external deps."""
    pts = sorted({(p.x, p.y) for p in points})
    if len(pts) <= 2:
        return [Point(x=x, y=y) for x, y in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return [Point(x=x, y=y) for x, y in hull]


def min_area_bounding_rectangle(points: Sequence[Point]) -> Optional[Polygon]:
    """Smallest-area rectangle (at any rotation) enclosing `points`.

    Standard "rotating calipers" technique: the minimum-area enclosing
    rectangle of a point set always has one side flush with an edge of the
    set's convex hull, so it suffices to try the orientation of each hull
    edge and keep the best. Generic computational-geometry utility -- no
    coordinates, thresholds, or assumptions tied to any specific drawing.
    Returns None for fewer than 3 distinct points (degenerate).
    """
    hull = convex_hull(points)
    if len(hull) < 3:
        return None

    best_area = math.inf
    best_rect: Optional[list[tuple[float, float]]] = None
    n = len(hull)
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        edge_dx, edge_dy = b.x - a.x, b.y - a.y
        edge_len = math.hypot(edge_dx, edge_dy)
        if edge_len < 1e-12:
            continue
        ux, uy = edge_dx / edge_len, edge_dy / edge_len  # unit vector along the edge
        vx, vy = -uy, ux  # perpendicular unit vector

        us = [(p.x - a.x) * ux + (p.y - a.y) * uy for p in hull]
        vs = [(p.x - a.x) * vx + (p.y - a.y) * vy for p in hull]
        u_min, u_max = min(us), max(us)
        v_min, v_max = min(vs), max(vs)
        area = (u_max - u_min) * (v_max - v_min)
        if area < best_area:
            best_area = area
            best_rect = [
                (a.x + u_min * ux + v_min * vx, a.y + u_min * uy + v_min * vy),
                (a.x + u_max * ux + v_min * vx, a.y + u_max * uy + v_min * vy),
                (a.x + u_max * ux + v_max * vx, a.y + u_max * uy + v_max * vy),
                (a.x + u_min * ux + v_max * vx, a.y + u_min * uy + v_max * vy),
            ]
    if best_rect is None:
        return None
    return Polygon(points=[Point(x=x, y=y) for x, y in best_rect])


def _total_least_squares_line(points: Sequence[Point]) -> tuple[Point, tuple[float, float]]:
    """Best-fit infinite line through `points` minimizing PERPENDICULAR
    distance (total/orthogonal least squares via the covariance matrix's
    principal axis) -- unlike ordinary least squares this is well-defined
    for near-vertical lines too. Returns (a point on the line, a unit
    direction vector)."""
    n = len(points)
    cx = sum(p.x for p in points) / n
    cy = sum(p.y for p in points) / n
    sxx = sum((p.x - cx) ** 2 for p in points)
    syy = sum((p.y - cy) ** 2 for p in points)
    sxy = sum((p.x - cx) * (p.y - cy) for p in points)
    theta = 0.5 * math.atan2(2 * sxy, sxx - syy)
    return Point(x=cx, y=cy), (math.cos(theta), math.sin(theta))


def _line_perpendicular_distance(pt: Point, anchor: Point, direction: tuple[float, float]) -> float:
    dx, dy = pt.x - anchor.x, pt.y - anchor.y
    nx, ny = -direction[1], direction[0]
    return abs(dx * nx + dy * ny)


def _line_projection(pt: Point, anchor: Point, direction: tuple[float, float]) -> float:
    dx, dy = pt.x - anchor.x, pt.y - anchor.y
    return dx * direction[0] + dy * direction[1]


def _intersect_lines(
    anchor_a: Point, dir_a: tuple[float, float], anchor_b: Point, dir_b: tuple[float, float],
) -> Optional[Point]:
    """Intersection of two infinite lines given as (point, direction). None if (near-)parallel."""
    dax, day = dir_a
    dbx, dby = dir_b
    denom = dax * dby - day * dbx
    if abs(denom) < 1e-9:
        return None
    t = ((anchor_b.x - anchor_a.x) * dby - (anchor_b.y - anchor_a.y) * dbx) / denom
    return Point(x=anchor_a.x + t * dax, y=anchor_a.y + t * day)


def _hull_diameter(hull: Sequence[Point]) -> float:
    """Max distance between any two points of a convex polygon (its own
    diameter, independent of how it happens to sit relative to the X/Y
    axes) -- unlike an axis-aligned bounding-box diagonal, this doesn't
    inflate when the shape is rotated, so tolerances derived from it stay
    rotation-invariant. Brute-force O(h^2) over the hull (not the full
    point set), which `min_area_bounding_rectangle` already does the
    equivalent of per hull edge, so this is not a new complexity class."""
    h = list(hull)
    n = len(h)
    if n < 2:
        return 0.0
    return max(math.hypot(h[i].x - h[j].x, h[i].y - h[j].y) for i in range(n) for j in range(i + 1, n))


def reconstruct_boundary_from_fragments(
    points: Sequence[Point],
    min_side_support: int = 4,
    collinearity_angle_tol_deg: float = 10.0,
    perpendicular_tol_fraction: float = 0.04,
) -> Optional[Polygon]:
    """Recover a closed boundary polygon from a scattered point cloud
    sampled from a FRAGMENTED trace -- e.g. hundreds of individually
    closed dash quads from a dash-dot property line, or a wall centerline
    broken by door/window gaps -- without assuming any two fragments touch
    or lie within any fixed snap distance of each other.

    Unlike fitting a single minimum-area bounding rectangle over every
    point (which silently absorbs any stray point -- an unrelated
    dimension mark a little further out, say -- into the fitted
    rectangle's own extent), this looks for actual straight RUNS of
    points: fragments are grouped onto a shared boundary side by
    ORIENTATION and PERPENDICULAR DISTANCE to a common supporting line,
    never by proximity between fragments -- so a large intentional dash
    gap and a rotated boundary are both handled exactly like a densely
    drawn, axis-aligned one would be. A side is only trusted once enough
    independent points support the same line (`min_side_support`); that,
    not any endpoint-snap tolerance, is what lets a genuine multi-dash
    side outvote a single unrelated stray mark that happens to sit on the
    convex hull.

    Algorithm: take the convex hull of `points`; merge consecutive hull
    edges into one candidate side per straight run when the new edge both
    (a) stays within `collinearity_angle_tol_deg` of the run's starting
    direction AND (b) lies within tolerance of the run's own line --
    the second check is what stops an unrelated but coincidentally
    parallel cluster elsewhere from being swept into a real side just
    because it happens to be adjacent on the hull. This collapses the
    zig-zag a noisy dash trace leaves on the hull into one run per real
    side; for each candidate side, gather every input point within
    tolerance of that side's fitted line and re-fit on the full support
    set; drop sides with fewer than `min_side_support` supporting points;
    intersect each pair of consecutive surviving sides (in hull-perimeter
    order, so they need not touch) to get the polygon's corners. All
    distance tolerances scale off the point cloud's own diameter (max
    distance between any two of its convex-hull points) rather than its
    axis-aligned bounding box, so they stay the same regardless of the
    boundary's rotation.

    Returns None -- the caller should fall back to a coarser method, e.g.
    `min_area_bounding_rectangle` -- when fewer than 3 sides survive,
    consecutive sides are too close to parallel to form a corner, or the
    resulting vertices don't wind consistently (the fitted lines
    disagreed too badly to trust). Assumes the true boundary is convex
    (true for a site/plot boundary; a non-convex outline, e.g. a building
    footprint with wings, needs a different reconstruction approach).
    """
    pts = list(points)
    hull = convex_hull(pts)
    n = len(hull)
    if n < 3:
        return None
    diag = _hull_diameter(hull)
    if diag <= 0:
        return None
    perp_tol = perpendicular_tol_fraction * diag
    margin = max(perp_tol * 2, 0.02 * diag)

    edge_angles = [
        math.degrees(math.atan2(hull[(i + 1) % n].y - hull[i].y, hull[(i + 1) % n].x - hull[i].x)) % 360.0
        for i in range(n)
    ]

    def _angle_diff(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def _same_run(edge_index: int, ref_start: int) -> bool:
        if _angle_diff(edge_angles[edge_index], edge_angles[ref_start]) > collinearity_angle_tol_deg:
            return False
        ref_anchor, ref_dir = hull[ref_start], (
            math.cos(math.radians(edge_angles[ref_start])), math.sin(math.radians(edge_angles[ref_start])),
        )
        a, b = hull[edge_index], hull[(edge_index + 1) % n]
        return (
            _line_perpendicular_distance(a, ref_anchor, ref_dir) <= perp_tol
            and _line_perpendicular_distance(b, ref_anchor, ref_dir) <= perp_tol
        )

    runs: list[tuple[int, int]] = []  # (start hull-edge index, edge count)
    run_start, run_len = 0, 1
    for i in range(1, n):
        if _same_run(i, run_start):
            run_len += 1
        else:
            runs.append((run_start, run_len))
            run_start, run_len = i, 1
    runs.append((run_start, run_len))
    if len(runs) > 1 and _same_run(runs[-1][0], runs[0][0]):
        last_start, last_len = runs.pop()
        first_start, first_len = runs[0]
        runs[0] = (last_start, last_len + first_len)

    sides: list[tuple[Point, tuple[float, float]]] = []
    for start, length in runs:
        run_hull_points = [hull[(start + k) % n] for k in range(length + 1)]
        anchor, direction = _total_least_squares_line(run_hull_points)
        run_projections = [_line_projection(p, anchor, direction) for p in run_hull_points]
        proj_min, proj_max = min(run_projections), max(run_projections)
        support = [
            p for p in pts
            if _line_perpendicular_distance(p, anchor, direction) <= perp_tol
            and proj_min - margin <= _line_projection(p, anchor, direction) <= proj_max + margin
        ]
        if len(support) < min_side_support:
            continue
        sides.append(_total_least_squares_line(support))

    if len(sides) < 3:
        return None

    # A genuine corner of the real boundary sits at or near the point
    # cloud's own extent -- it is, after all, fitted FROM those points.
    # Two independently-fitted side lines that are close to but not quite
    # parallel (just outside the 5-degree rejection below) can still
    # intersect at a point far outside where any of the input points
    # actually are (a "shallow" intersection); the angle check alone
    # doesn't catch that. Bound every vertex to the input's own hull
    # extent (with a generous margin for a real corner sitting just past
    # its nearest supporting points) so a numerically-valid but physically
    # nonsensical corner is rejected the same way an inconsistent winding
    # already is below, rather than silently returned as a wildly
    # oversized polygon.
    hull_min_x = min(p.x for p in hull)
    hull_max_x = max(p.x for p in hull)
    hull_min_y = min(p.y for p in hull)
    hull_max_y = max(p.y for p in hull)
    vertex_margin = diag * 0.5

    m = len(sides)
    vertices: list[Point] = []
    for i in range(m):
        a_anchor, a_dir = sides[i]
        b_anchor, b_dir = sides[(i + 1) % m]
        angle_a = math.degrees(math.atan2(a_dir[1], a_dir[0])) % 180.0
        angle_b = math.degrees(math.atan2(b_dir[1], b_dir[0])) % 180.0
        if min(abs(angle_a - angle_b), 180.0 - abs(angle_a - angle_b)) < 5.0:
            return None  # consecutive sides too close to parallel to form a real corner
        vertex = _intersect_lines(a_anchor, a_dir, b_anchor, b_dir)
        if vertex is None:
            return None
        if not (
            hull_min_x - vertex_margin <= vertex.x <= hull_max_x + vertex_margin
            and hull_min_y - vertex_margin <= vertex.y <= hull_max_y + vertex_margin
        ):
            return None  # a "shallow" near-parallel intersection extrapolated far outside the input points
        vertices.append(vertex)

    polygon = Polygon(points=vertices)
    if polygon.area <= 0:
        return None
    # Consistent winding sanity check: vertices came from a hull-ordered
    # walk, so every turn should bend the same way. If the independently
    # fitted lines disagree badly enough to flip the turn direction
    # somewhere, the fit isn't trustworthy.
    turn_signs = {
        cross > 0
        for i in range(m)
        for cross in [(
            (vertices[i].x - vertices[i - 1].x) * (vertices[(i + 1) % m].y - vertices[i].y)
            - (vertices[i].y - vertices[i - 1].y) * (vertices[(i + 1) % m].x - vertices[i].x)
        )]
        if abs(cross) > 1e-9
    }
    if len(turn_signs) > 1:
        return None
    return polygon


def rectangularity(polygon: Polygon) -> float:
    """polygon_area / bounding_box_area, in [0, 1] for simple polygons. 1.0 = perfect rectangle."""
    bbox = polygon.bounding_box
    bbox_area = bbox.width * bbox.height
    if bbox_area <= 0:
        return 0.0
    return min(1.0, polygon.area / bbox_area)


def aspect_ratio(bbox: BoundingBox) -> float:
    """Long side / short side. Returns +inf for degenerate (zero-height/width) boxes."""
    w, h = bbox.width, bbox.height
    if min(w, h) <= 0:
        return math.inf
    return max(w, h) / min(w, h)


def point_segment_distance(p: Point, seg: Line) -> float:
    ax, ay = seg.start.x, seg.start.y
    bx, by = seg.end.x, seg.end.y
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return math.hypot(p.x - ax, p.y - ay)
    t = max(0.0, min(1.0, ((p.x - ax) * dx + (p.y - ay) * dy) / length_sq))
    proj_x, proj_y = ax + t * dx, ay + t * dy
    return math.hypot(p.x - proj_x, p.y - proj_y)


def segment_segment_distance(a: Line, b: Line) -> float:
    """Minimum distance between two finite segments (0 if they intersect)."""
    if _segments_intersect(a, b):
        return 0.0
    return min(
        point_segment_distance(a.start, b),
        point_segment_distance(a.end, b),
        point_segment_distance(b.start, a),
        point_segment_distance(b.end, a),
    )


def _orientation(p: Point, q: Point, r: Point) -> float:
    return (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x)


def _on_segment(p: Point, q: Point, r: Point) -> bool:
    return (
        min(p.x, r.x) - 1e-9 <= q.x <= max(p.x, r.x) + 1e-9
        and min(p.y, r.y) - 1e-9 <= q.y <= max(p.y, r.y) + 1e-9
    )


def _segments_intersect(a: Line, b: Line) -> bool:
    p1, q1, p2, q2 = a.start, a.end, b.start, b.end
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)
    if ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0)) and o1 != 0 and o2 != 0:
        return True
    if abs(o1) < 1e-9 and _on_segment(p1, p2, q1):
        return True
    if abs(o2) < 1e-9 and _on_segment(p1, q2, q1):
        return True
    if abs(o3) < 1e-9 and _on_segment(p2, p1, q2):
        return True
    if abs(o4) < 1e-9 and _on_segment(p2, q1, q2):
        return True
    return False


def point_in_polygon(p: Point, polygon: Polygon) -> bool:
    """Standard ray-casting point-in-polygon test."""
    pts = polygon.points
    n = len(pts)
    inside = False
    x, y = p.x, p.y
    x1, y1 = pts[-1].x, pts[-1].y
    for i in range(n):
        x2, y2 = pts[i].x, pts[i].y
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1
        ):
            inside = not inside
        x1, y1 = x2, y2
    return inside


def point_to_polygon_boundary_distance(p: Point, polygon: Polygon) -> float:
    return min(point_segment_distance(p, e) for e in polygon_edges(polygon))


def polygon_to_edge_distance(inner: Polygon, edge: Edge) -> float:
    """
    Min distance from an (assumed interior) polygon to a single boundary
    edge of an outer polygon — used for building-to-plot-edge setbacks.

    Considers both inner vertices and inner edges vs the target edge, so
    it is correct whether the nearest approach is vertex-to-edge or
    edge-to-edge (near-parallel walls).
    """
    vertex_min = min(point_segment_distance(v, edge) for v in inner.points)
    edge_min = min(segment_segment_distance(e, edge) for e in polygon_edges(inner))
    return min(vertex_min, edge_min)


def polygon_to_edges_distance(inner: Polygon, edges: Sequence[Edge]) -> float:
    """Min distance from `inner` to any edge in a group of boundary edges (one 'side')."""
    if not edges:
        return math.inf
    return min(polygon_to_edge_distance(inner, e) for e in edges)


def scale_polygon(polygon: Polygon, points_per_metre: float) -> Polygon:
    """Convert a PAGE_POINTS polygon into METRIC_PLAN, dividing by the scale factor."""
    if points_per_metre <= 0:
        raise ValueError("points_per_metre must be > 0")
    return Polygon(
        points=[Point(x=pt.x / points_per_metre, y=pt.y / points_per_metre) for pt in polygon.points]
    )


def scale_bbox(bbox: BoundingBox, points_per_metre: float) -> BoundingBox:
    if points_per_metre <= 0:
        raise ValueError("points_per_metre must be > 0")
    return BoundingBox(
        min_x=bbox.min_x / points_per_metre,
        min_y=bbox.min_y / points_per_metre,
        max_x=bbox.max_x / points_per_metre,
        max_y=bbox.max_y / points_per_metre,
    )


def scale_line(line: Line, points_per_metre: float) -> Line:
    if points_per_metre <= 0:
        raise ValueError("points_per_metre must be > 0")
    return Line(
        start=Point(x=line.start.x / points_per_metre, y=line.start.y / points_per_metre),
        end=Point(x=line.end.x / points_per_metre, y=line.end.y / points_per_metre),
    )


def bbox_to_polygon(bbox: BoundingBox) -> Polygon:
    return Polygon(
        points=[
            Point(x=bbox.min_x, y=bbox.min_y),
            Point(x=bbox.max_x, y=bbox.min_y),
            Point(x=bbox.max_x, y=bbox.max_y),
            Point(x=bbox.min_x, y=bbox.max_y),
        ]
    )


def line_orientation_degrees(line: Line) -> float:
    """Orientation of a line, in degrees, folded into [0, 180) (direction, not sense)."""
    dx, dy = line.end.x - line.start.x, line.end.y - line.start.y
    return math.degrees(math.atan2(dy, dx)) % 180.0


def orientation_alignment(a_deg: float, b_deg: float) -> float:
    """1.0 = perfectly parallel, 0.0 = perfectly perpendicular (both mod-180 degrees)."""
    diff = abs(a_deg - b_deg) % 180.0
    diff = min(diff, 180.0 - diff)
    return 1.0 - (diff / 90.0)


def edge_midpoint(edge: Edge) -> Point:
    return Point(x=(edge.start.x + edge.end.x) / 2.0, y=(edge.start.y + edge.end.y) / 2.0)


def outward_normal(edge: Edge, centroid: Point) -> tuple[float, float]:
    """
    Unit outward-pointing normal of a polygon edge, given the polygon's
    centroid (used to disambiguate winding order without assuming it).
    """
    dx, dy = edge.end.x - edge.start.x, edge.end.y - edge.start.y
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return (0.0, 0.0)
    # two candidate normals (perpendicular to the edge)
    n1 = (-dy / length, dx / length)
    n2 = (dy / length, -dx / length)
    mid = edge_midpoint(edge)
    to_mid = (mid.x - centroid.x, mid.y - centroid.y)
    # outward normal is the one pointing away from the centroid
    if n1[0] * to_mid[0] + n1[1] * to_mid[1] >= 0:
        return n1
    return n2


__all__ = [
    "Edge",
    "polygon_edges",
    "polygon_perimeter",
    "polygon_centroid",
    "is_valid_polygon",
    "is_page_frame_like",
    "dedupe_bboxes",
    "rectangularity",
    "aspect_ratio",
    "point_segment_distance",
    "segment_segment_distance",
    "point_in_polygon",
    "point_to_polygon_boundary_distance",
    "polygon_to_edge_distance",
    "polygon_to_edges_distance",
    "scale_polygon",
    "scale_bbox",
    "scale_line",
    "bbox_to_polygon",
    "line_orientation_degrees",
    "orientation_alignment",
    "edge_midpoint",
    "outward_normal",
]
