"""
Plots whose boundary is not an axis-aligned rectangle.

`site_plan.py` models a plot as an axis-aligned rectangle built from perfectly
horizontal/vertical lines. Many real plots are skewed: one or two edges lean a
few degrees (PLAN9's left edge leans 6.6 degrees, its top 0.7), their
dimension lines lean with them, and setbacks vary along a slanted edge (2.52 m
at one end of a gap, 1.00 m at the other). A rectangle model cannot represent
that, and worse it can fit the wrong thing: on PLAN8 a rectangle belonging to a
different drawing matched the stated areas by coincidence and won.

This module recovers such a plot as a QUADRILATERAL from line segments and
measures setbacks against the real edges:

  1. Merge collinear segments (solid or dashed, any tilt up to
     `MAX_TILT_DEG`) into edges. Boundary edges are drawn heavier than
     dimension lines, so thin lines are dropped relative to the heaviest.
  2. Enumerate quadrilaterals from two near-horizontal and two near-vertical
     edges; every corner must actually be reached by both edges meeting there.
  3. Validate by the sheet's own printed edge labels: each edge's length in
     points divided by a label printed beside it gives a scale, and the true
     plot is the quadrilateral on which several independent edges imply the
     SAME scale. A coincidental shape does not do that.
  4. Pick the building rectangle inside it, and measure each setback as the
     MINIMUM perpendicular distance from the building to its plot edge, the
     conservative value for a compliance check (a setback that is 2.5 m at one
     end and 1.0 m at the other must be read as 1.0 m).

It never imports `site_plan` (which imports it); rectangle candidates and
numeric labels are passed in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from backend.schemas.geometry import BoundingBox

MIN_TILT_DEG = 0.5          # below this an edge counts as axis-aligned
MAX_TILT_DEG = 12.0         # above this it is not a plot edge that "leans"
MIN_SEGMENT_PTS = 6.0       # short pieces are kept only so dashes can merge
MIN_EDGE_PTS = 60.0
MIN_SIDE_PTS = 45.0
DETECT_MIN_SEGMENT_PTS = 100.0
DETECT_MIN_STROKE = 0.6     # absolute floor: thin lines are never boundary evidence
RELATIVE_STROKE = 0.6       # boundary edges are >= this fraction of the heaviest
MERGE_ANGLE_DEG = 1.0
MERGE_OFFSET_PTS = 2.5
MERGE_GAP_PTS = 14.0        # dash gaps
CORNER_TOL_PTS = 6.0
CORNER_TOL_FRACTION = 0.04
SIDE_COVERAGE = 0.82
DASH_SPAN = 0.95
DASH_COVERAGE = 0.30
DASH_MIN_PIECES = 4
LABEL_BAND_PTS = 45.0
LABEL_REL_TOL = 0.03
LABEL_ABS_TOL = 0.05
SETBACK_LABEL_TOL = 0.15
MAX_EDGES_PER_AXIS = 30
MIN_BUILDING_FRACTION = 0.25
# Physical sanity limits on a validated plot. Without them small setback
# callouts ("1.10m") pass for edge labels and a tiny shape "validates" at an
# absurd scale (seen on a real sheet: a ~1 m quadrilateral at 176 pt/m).
MIN_EDGE_LABEL_M = 3.0
MIN_PLOT_SIDE_M = 3.0
MAX_PLOT_SIDE_M = 80.0
SCALE_RANGE_PTS_PER_M = (3.0, 120.0)     # 1:20 .. 1:1000 on the largest sheets


@dataclass(frozen=True)
class Edge:
    """A merged near-axis line. `a`->`b` runs left->right ('H') or top->bottom ('V')."""

    orientation: str
    a: tuple[float, float]
    b: tuple[float, float]
    stroke: float
    intervals: tuple[tuple[float, float], ...]   # inked spans, distance from `a` along the edge

    @property
    def length(self) -> float:
        return math.hypot(self.b[0] - self.a[0], self.b[1] - self.a[1])

    @property
    def unit(self) -> tuple[float, float]:
        length = self.length or 1.0
        return ((self.b[0] - self.a[0]) / length, (self.b[1] - self.a[1]) / length)

    @property
    def tilt_deg(self) -> float:
        ux, uy = self.unit
        ang = math.degrees(math.atan2(uy, ux)) % 180
        return min(ang % 90, 90 - ang % 90)


@dataclass
class SkewedPlot:
    corners: dict[str, tuple[float, float]]            # tl, tr, br, bl
    edges: dict[str, Edge]                             # top, right, bottom, left
    scale_pts_per_m: float
    edge_length_m: dict[str, float]                    # geometric, corner to corner
    edge_label_m: dict[str, Optional[float]]           # printed label that agreed, if any
    label_agreement: int
    tilt_deg: float
    building: Optional[BoundingBox] = None
    setback_geometry_m: dict[str, float] = field(default_factory=dict)   # min perpendicular distance
    setback_labels_m: dict[str, list[float]] = field(default_factory=dict)
    building_label_m: tuple[Optional[float], Optional[float]] = (None, None)   # printed width, depth
    notes: list[str] = field(default_factory=list)

    def bbox(self) -> BoundingBox:
        xs = [p[0] for p in self.corners.values()]
        ys = [p[1] for p in self.corners.values()]
        return BoundingBox(min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys))


# ---------------------------------------------------------------- edges ----


def _segments(lines, region: BoundingBox):
    out = []
    for rl in lines:
        s, e = rl.line.start, rl.line.end
        length = math.hypot(e.x - s.x, e.y - s.y)
        if length < MIN_SEGMENT_PTS:
            continue
        mx, my = (s.x + e.x) / 2, (s.y + e.y) / 2
        if not (region.min_x <= mx <= region.max_x and region.min_y <= my <= region.max_y):
            continue
        ang = math.degrees(math.atan2(e.y - s.y, e.x - s.x)) % 180
        tilt = min(ang % 90, 90 - ang % 90)
        if tilt > MAX_TILT_DEG:
            continue
        orient = "H" if (ang < 45 or ang > 135) else "V"
        p0, p1 = ((s.x, s.y), (e.x, e.y))
        if (orient == "H" and p0[0] > p1[0]) or (orient == "V" and p0[1] > p1[1]):
            p0, p1 = p1, p0
        out.append((orient, p0, p1, float(rl.stroke_width or 0.0), tilt, length))
    return out


def _line_params(orient: str, p0, p1) -> tuple[float, float]:
    """(signed angle in degrees from the axis, offset of the line from the origin)."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    if orient == "H":
        slope = dy / dx if dx else 0.0
        return math.degrees(math.atan(slope)), p0[1] - slope * p0[0]
    slope = dx / dy if dy else 0.0
    return math.degrees(math.atan(slope)), p0[0] - slope * p0[1]


def merge_edges(lines, region: BoundingBox) -> list[Edge]:
    """Merge collinear near-axis segments (solid or dashed) into edges."""
    edges: list[Edge] = []
    segs = _segments(lines, region)
    for orient in ("H", "V"):
        group = [s for s in segs if s[0] == orient]
        keyed = sorted(((*_line_params(o, p0, p1), (o, p0, p1, st, tl, ln)) for o, p0, p1, st, tl, ln in group),
                       key=lambda t: t[1])
        clusters: list[list] = []
        for phi, c, seg in keyed:
            placed = False
            for cl in clusters:
                if abs(c - cl[-1][1]) <= MERGE_OFFSET_PTS and abs(phi - cl[0][0]) <= MERGE_ANGLE_DEG:
                    cl.append((phi, c, seg))
                    placed = True
                    break
            if not placed:
                clusters.append([(phi, c, seg)])
        for cl in clusters:
            axis = 0 if orient == "H" else 1
            members = sorted((m[2] for m in cl), key=lambda s: s[1][axis])
            ref = max(members, key=lambda s: s[5])
            ux, uy = ref[2][0] - ref[1][0], ref[2][1] - ref[1][1]
            rl_ = math.hypot(ux, uy) or 1.0
            ux, uy = ux / rl_, uy / rl_
            ox, oy = ref[1]
            spans = sorted((((s[1][0] - ox) * ux + (s[1][1] - oy) * uy), ((s[2][0] - ox) * ux + (s[2][1] - oy) * uy), s[3])
                           for s in members)
            chain: list[list[float]] = []
            for a, b, st in spans:
                if a > b:
                    a, b = b, a
                if chain and a <= chain[-1][1] + MERGE_GAP_PTS:
                    chain[-1][1] = max(chain[-1][1], b)
                    chain[-1][2].append((a, b, st))
                else:
                    chain.append([a, b, [(a, b, st)]])
            for lo, hi, pieces in chain:
                if hi - lo < MIN_EDGE_PTS:
                    continue
                ivs = _union([(a, b) for a, b, _ in pieces])
                inked = sum(b - a for a, b in ivs)
                strokes = [(b - a) * st for a, b, st in pieces]
                stroke = sum(strokes) / max(1e-9, sum(b - a for a, b, _ in pieces))
                edges.append(Edge(
                    orientation=orient,
                    a=(ox + ux * lo, oy + uy * lo), b=(ox + ux * hi, oy + uy * hi),
                    stroke=stroke,
                    intervals=tuple((a - lo, b - lo) for a, b in ivs),
                ))
    return edges


def _union(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def slanted_boundary_evidence(lines, region: BoundingBox) -> int:
    """Long, heavy, near-axis-but-leaning segments inside `region`. Zero on any
    sheet drawn with exactly axis-aligned lines, which is what isolates this
    module from plots the rectangle model already handles."""
    count = 0
    for orient, p0, p1, stroke, tilt, length in _segments(lines, region):
        if length >= DETECT_MIN_SEGMENT_PTS and stroke >= DETECT_MIN_STROKE and MIN_TILT_DEG <= tilt <= MAX_TILT_DEG:
            count += 1
    return count


# ---------------------------------------------------------------- quads ----


def _intersect(e1: Edge, e2: Edge) -> Optional[tuple[float, float]]:
    (px, py), (ux, uy) = e1.a, e1.unit
    (qx, qy), (vx, vy) = e2.a, e2.unit
    det = ux * (-vy) - uy * (-vx)
    if abs(det) < 1e-9:
        return None
    dx, dy = qx - px, qy - py
    t = (dx * (-vy) - dy * (-vx)) / det
    return (px + ux * t, py + uy * t)


def _t_along(edge: Edge, q: tuple[float, float]) -> float:
    ux, uy = edge.unit
    return (q[0] - edge.a[0]) * ux + (q[1] - edge.a[1]) * uy


def _reaches(edge: Edge, q: tuple[float, float]) -> bool:
    tol = max(CORNER_TOL_PTS, CORNER_TOL_FRACTION * edge.length)
    return -tol <= _t_along(edge, q) <= edge.length + tol


def _side_drawn(edge: Edge, q0: tuple[float, float], q1: tuple[float, float]) -> bool:
    lo, hi = sorted((_t_along(edge, q0), _t_along(edge, q1)))
    side = hi - lo
    if side <= 0:
        return False
    clipped = [(max(a, lo), min(b, hi)) for a, b in edge.intervals if min(b, hi) > max(a, lo)]
    if not clipped:
        return False
    covered = sum(b - a for a, b in clipped)
    tol = max(CORNER_TOL_PTS, CORNER_TOL_FRACTION * side)
    if min(a for a, _ in clipped) - lo > tol or hi - max(b for _, b in clipped) > tol:
        return False
    if covered >= SIDE_COVERAGE * side:
        return True
    span = max(b for _, b in clipped) - min(a for a, _ in clipped)
    return span >= DASH_SPAN * side and covered >= DASH_COVERAGE * side and len(clipped) >= DASH_MIN_PIECES


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex(pts) -> bool:
    signs = [_cross(pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]) for i in range(4)]
    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


def _polygon_area(pts) -> float:
    return abs(sum(pts[i][0] * pts[(i + 1) % 4][1] - pts[(i + 1) % 4][0] * pts[i][1] for i in range(4))) / 2


def _boundary_edges(edges: list[Edge]) -> list[Edge]:
    """Drop lines much thinner than the heaviest long line: dimension lines."""
    long_edges = [e for e in edges if e.length >= DETECT_MIN_SEGMENT_PTS]
    if not long_edges:
        return []
    heaviest = max(e.stroke for e in long_edges)
    return [e for e in edges if e.stroke >= RELATIVE_STROKE * heaviest]


def _mid(e: Edge) -> tuple[float, float]:
    return ((e.a[0] + e.b[0]) / 2, (e.a[1] + e.b[1]) / 2)


def _overlap(lo1: float, hi1: float, lo2: float, hi2: float) -> float:
    return max(0.0, min(hi1, hi2) - max(lo1, lo2))


def candidate_quads(
    edges: list[Edge], focus: Optional[tuple[float, float]] = None
) -> list[tuple[dict[str, tuple[float, float]], dict[str, Edge]]]:
    """Quadrilaterals from two near-horizontal and two near-vertical edges.

    On a sheet the window also holds tables and other drawings, whose long rules
    would crowd out the plot's own edges if edges were ranked by length alone;
    with a `focus` (the SITE PLAN caption) the nearest edges are kept instead."""
    boundary = _boundary_edges(edges)

    def rank(e: Edge):
        if focus is None:
            return -e.length
        return math.dist(_mid(e), focus)

    hs = sorted((e for e in boundary if e.orientation == "H"), key=rank)[:MAX_EDGES_PER_AXIS]
    vs = sorted((e for e in boundary if e.orientation == "V"), key=rank)[:MAX_EDGES_PER_AXIS]
    quads = []
    for i, h1 in enumerate(hs):
        for h2 in hs[i + 1:]:
            top, bottom = sorted((h1, h2), key=lambda e: _mid(e)[1])
            # The two horizontal edges must face each other, well apart.
            if _overlap(min(top.a[0], top.b[0]), max(top.a[0], top.b[0]),
                        min(bottom.a[0], bottom.b[0]), max(bottom.a[0], bottom.b[0])) < 0.5 * min(top.length, bottom.length):
                continue
            if _mid(bottom)[1] - _mid(top)[1] < MIN_SIDE_PTS:
                continue
            for j, v1 in enumerate(vs):
                for v2 in vs[j + 1:]:
                    left, right = sorted((v1, v2), key=lambda e: _mid(e)[0])
                    if _overlap(min(left.a[1], left.b[1]), max(left.a[1], left.b[1]),
                                min(right.a[1], right.b[1]), max(right.a[1], right.b[1])) < 0.5 * min(left.length, right.length):
                        continue
                    if _mid(right)[0] - _mid(left)[0] < MIN_SIDE_PTS:
                        continue
                    tl, tr = _intersect(top, left), _intersect(top, right)
                    br, bl = _intersect(bottom, right), _intersect(bottom, left)
                    if None in (tl, tr, br, bl):
                        continue
                    pts = [tl, tr, br, bl]
                    if not _convex(pts):
                        continue
                    sides = [math.dist(pts[k], pts[(k + 1) % 4]) for k in range(4)]
                    if min(sides) < MIN_SIDE_PTS or max(sides) / min(sides) > 8:
                        continue
                    if not (_reaches(top, tl) and _reaches(top, tr) and _reaches(bottom, bl) and _reaches(bottom, br)
                            and _reaches(left, tl) and _reaches(left, bl) and _reaches(right, tr) and _reaches(right, br)):
                        continue
                    if not (_side_drawn(top, tl, tr) and _side_drawn(bottom, bl, br)
                            and _side_drawn(left, tl, bl) and _side_drawn(right, tr, br)):
                        continue
                    if max(top.tilt_deg, bottom.tilt_deg, left.tilt_deg, right.tilt_deg) < MIN_TILT_DEG:
                        continue   # an axis-aligned rectangle: the rectangle path owns it
                    quads.append(({"tl": tl, "tr": tr, "br": br, "bl": bl},
                                  {"top": top, "right": right, "bottom": bottom, "left": left}))
    return quads


# ---------------------------------------------------------------- labels ----


def _labels_for_edge(nums, edge: Edge, p0, p1) -> list[float]:
    """Numeric labels printed beside an edge (in metres, as parsed)."""
    ux, uy = edge.unit
    nx, ny = -uy, ux
    length = math.dist(p0, p1)
    mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
    out = []
    for value, item, _unit in nums:
        c = item.bounding_box.center
        along = (c.x - mid[0]) * ux + (c.y - mid[1]) * uy
        perp = (c.x - mid[0]) * nx + (c.y - mid[1]) * ny
        if abs(along) <= length / 2 + 15 and abs(perp) <= LABEL_BAND_PTS and MIN_EDGE_LABEL_M <= value <= MAX_PLOT_SIDE_M:
            out.append(value)
    return out


PRINTED_SCALE_SNAP = 0.06


def _snap_to_printed(scale: float, printed: Sequence[float]) -> float:
    """A printed scale is exact for its view; edge lengths in points carry
    line-weight and extent noise. When the label-derived scale is within a few
    percent of a printed one, the printed one is the better estimate."""
    near = [p for p in printed if p > 0 and abs(scale - p) / p <= PRINTED_SCALE_SNAP]
    return min(near, key=lambda p: abs(scale - p)) if near else scale


def _agrees(length_pts: float, scale: float, value: float) -> bool:
    return abs(length_pts / scale - value) <= max(LABEL_ABS_TOL, LABEL_REL_TOL * value)


def _solve_scale(lengths: dict[str, float], labels: dict[str, list[float]], printed: Sequence[float]):
    """The scale on which the most independent edges agree with a printed label."""
    candidates: list[float] = list(printed)
    for name, values in labels.items():
        candidates += [lengths[name] / v for v in values if v > 0]
    best = None
    for s in candidates:
        if s <= 0:
            continue
        agreeing = {n: v for n, vs in labels.items() for v in [next((x for x in vs if _agrees(lengths[n], s, x)), None)]
                    if v is not None}
        orients = {("H" if n in ("top", "bottom") else "V") for n in agreeing}
        score = len(agreeing) + (0.5 if any(abs(s - p) / p < 0.03 for p in printed) else 0)
        if best is None or (score, len(orients)) > (best[0], best[3]):
            best = (score, s, agreeing, len(orients))
    return best


# ---------------------------------------------------------------- analyse ----


def _point_in_quad(p, pts, tol: float = 2.0) -> bool:
    """Inside the convex quadrilateral, allowing `tol` points of slack outside an edge."""
    dists = []
    for i in range(4):
        a, b = pts[i], pts[(i + 1) % 4]
        length = math.dist(a, b) or 1.0
        dists.append(_cross(a, b, p) / length)
    orientation = 1.0 if _cross(pts[0], pts[1], pts[2]) > 0 else -1.0
    return all(d * orientation >= -tol for d in dists)


def _perp_distance(edge: Edge, q: tuple[float, float]) -> float:
    ux, uy = edge.unit
    return abs((q[0] - edge.a[0]) * (-uy) + (q[1] - edge.a[1]) * ux)


def analyse(
    lines,
    region: BoundingBox,
    *,
    nums,
    rect_candidates: Sequence[BoundingBox],
    printed_scales: Sequence[float] = (),
    plot_area_target_m2: Optional[float] = None,
    footprint_area_target_m2: Optional[float] = None,
    caption: Optional[BoundingBox] = None,
) -> tuple[str, Optional[SkewedPlot], list[str]]:
    """('not_skewed'|'unresolved'|'resolved', plot, notes)."""
    if slanted_boundary_evidence(lines, region) < 1:
        return "not_skewed", None, []
    notes = [
        "the site plan's boundary contains heavy, slanted edges, so the axis-aligned rectangle model does not apply; "
        "the plot was modelled as a quadrilateral."
    ]
    edges = merge_edges(lines, region)
    best = None
    focus = None
    if caption is not None:
        focus = ((caption.min_x + caption.max_x) / 2, (caption.min_y + caption.max_y) / 2)
    for corners, qedges in candidate_quads(edges, focus):
        pts = [corners[k] for k in ("tl", "tr", "br", "bl")]
        lengths = {"top": math.dist(corners["tl"], corners["tr"]), "right": math.dist(corners["tr"], corners["br"]),
                   "bottom": math.dist(corners["bl"], corners["br"]), "left": math.dist(corners["tl"], corners["bl"])}
        labels = {
            "top": _labels_for_edge(nums, qedges["top"], corners["tl"], corners["tr"]),
            "right": _labels_for_edge(nums, qedges["right"], corners["tr"], corners["br"]),
            "bottom": _labels_for_edge(nums, qedges["bottom"], corners["bl"], corners["br"]),
            "left": _labels_for_edge(nums, qedges["left"], corners["tl"], corners["bl"]),
        }
        solved = _solve_scale(lengths, labels, printed_scales)
        if solved is None:
            continue
        score, scale, agreeing, n_orient = solved
        scale = _snap_to_printed(scale, printed_scales)
        if not (SCALE_RANGE_PTS_PER_M[0] <= scale <= SCALE_RANGE_PTS_PER_M[1]):
            continue
        if not all(MIN_PLOT_SIDE_M <= v / scale <= MAX_PLOT_SIDE_M for v in lengths.values()):
            continue
        if len(agreeing) < 2 or n_orient < 2 and not printed_scales:
            continue
        area_m2 = _polygon_area(pts) / scale ** 2
        area_bonus = 0.0
        area_agrees = bool(plot_area_target_m2) and abs(area_m2 - plot_area_target_m2) / plot_area_target_m2 <= 0.03
        if area_agrees:
            area_bonus = 2.0
        # Two label-agreeing edges alone can be coincidence (a wrong shape on a
        # real sheet passed with two); require a third edge, or the stated plot
        # area as independent confirmation.
        if len(agreeing) < 3 and not area_agrees:
            continue
        stroke = sum(e.stroke for e in qedges.values()) / 4
        rank = (score + area_bonus, stroke, _polygon_area(pts))
        if best is None or rank > best[0]:
            best = (rank, corners, qedges, lengths, scale, agreeing, area_m2)
    if best is None:
        notes.append(
            "no quadrilateral could be validated: none had three edges whose printed labels implied one common "
            "drawing scale (or two plus the stated plot area). Nothing is asserted from it."
        )
        return "unresolved", None, notes

    _rank, corners, qedges, lengths, scale, agreeing, area_m2 = best
    pts = [corners[k] for k in ("tl", "tr", "br", "bl")]
    plot = SkewedPlot(
        corners=corners, edges=qedges, scale_pts_per_m=scale,
        edge_length_m={k: v / scale for k, v in lengths.items()},
        edge_label_m={k: agreeing.get(k) for k in ("top", "right", "bottom", "left")},
        label_agreement=len(agreeing),
        tilt_deg=max(e.tilt_deg for e in qedges.values()),
        notes=notes,
    )
    plot.notes.append(
        f"quadrilateral validated: {len(agreeing)} of 4 edges' printed labels agree on one scale "
        f"({scale:.2f} pt/m); sides top/right/bottom/left = "
        + " / ".join(f"{plot.edge_length_m[k]:.2f}" for k in ("top", "right", "bottom", "left")) + " m."
    )
    _place_building(plot, pts, nums, rect_candidates, footprint_area_target_m2)
    return "resolved", plot, plot.notes


def _place_building(plot: SkewedPlot, pts, nums, rects: Sequence[BoundingBox],
                    footprint_area_target_m2: Optional[float] = None) -> None:
    s = plot.scale_pts_per_m
    plot_area = _polygon_area(pts)
    best = None
    for r in rects:
        corners = [(r.min_x, r.min_y), (r.max_x, r.min_y), (r.max_x, r.max_y), (r.min_x, r.max_y)]
        if not all(_point_in_quad(c, pts) for c in corners):
            continue
        area = r.width * r.height
        if not (MIN_BUILDING_FRACTION * plot_area <= area <= 0.98 * plot_area):
            continue
        # Printed labels along the rectangle's own edges: width along the top,
        # depth along the left. Two independent agreements identify the building.
        w_labels = _labels_for_edge(nums, _axis_edge("H", (r.min_x, r.min_y), (r.max_x, r.min_y)), (r.min_x, r.min_y), (r.max_x, r.min_y))
        d_labels = _labels_for_edge(nums, _axis_edge("V", (r.min_x, r.min_y), (r.min_x, r.max_y)), (r.min_x, r.min_y), (r.min_x, r.max_y))
        w_ok = next((v for v in w_labels if _agrees(r.width, s, v)), None)
        d_ok = next((v for v in d_labels if _agrees(r.height, s, v)), None)
        score = (w_ok is not None) + (d_ok is not None)
        # No printed building dimensions on some sheets: the stated footprint
        # area is then the identifying evidence (as in the rectangle path).
        if footprint_area_target_m2 and abs(area / s ** 2 - footprint_area_target_m2) / footprint_area_target_m2 <= 0.03:
            score += 1
        if best is None or (score, area) > (best[0], best[1]):
            best = (score, area, r, w_ok, d_ok)
    if best is None or best[0] == 0:
        plot.notes.append(
            "no building rectangle inside the plot has printed width/depth labels agreeing with the scale, "
            "nor an area matching the stated footprint."
        )
        return
    _score, _area, r, w_ok, d_ok = best
    plot.building = r
    plot.building_label_m = (w_ok, d_ok)
    plot.notes.append(
        f"building rectangle {r.width / s:.2f} x {r.height / s:.2f} m (printed labels: "
        f"{w_ok if w_ok else 'none'} x {d_ok if d_ok else 'none'})."
    )
    # Setback per side: minimum perpendicular distance from the building's
    # near corners to that plot edge.
    corners = {"tl": (r.min_x, r.min_y), "tr": (r.max_x, r.min_y), "br": (r.max_x, r.max_y), "bl": (r.min_x, r.max_y)}
    near = {"top": ("tl", "tr"), "bottom": ("bl", "br"), "left": ("tl", "bl"), "right": ("tr", "br")}
    for side, (c1, c2) in near.items():
        plot.setback_geometry_m[side] = min(_perp_distance(plot.edges[side], corners[c1]),
                                            _perp_distance(plot.edges[side], corners[c2])) / s
    # Printed callouts sitting in each gap, for cross-checking.
    for side in near:
        edge = plot.edges[side]
        vals = []
        for value, item, _u in nums:
            c = item.bounding_box.center
            if not _point_in_quad((c.x, c.y), pts, tol=0):
                continue
            if r.min_x <= c.x <= r.max_x and r.min_y <= c.y <= r.max_y:
                continue
            if _perp_distance(edge, (c.x, c.y)) <= 0.5 * max(1.0, plot.setback_geometry_m[side] * s) + 25 and value <= 6.0:
                vals.append(value)
        plot.setback_labels_m[side] = vals


def _axis_edge(orient: str, a, b) -> Edge:
    return Edge(orientation=orient, a=a, b=b, stroke=0.0, intervals=())


__all__ = ["Edge", "SkewedPlot", "analyse", "candidate_quads", "merge_edges", "slanted_boundary_evidence"]
