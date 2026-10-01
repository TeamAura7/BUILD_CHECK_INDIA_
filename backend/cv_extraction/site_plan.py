"""Deterministic, independent CV/native-text site-plan resolver.

This module is intentionally separate from the legacy candidate-resolution
pipeline.  It answers one narrow validation question:

    "Can the PDF/CV side independently recover the canonical site-plan
     dimensions without Vision?"

For vector PDFs it uses only PDF-native text and vector line geometry.  It
never imports the Vision stack.  It detects a SITE PLAN region, reconstructs
nested axis-aligned rectangles from vector lines, anchors explicit dimension
labels to the outer rectangle, derives the building footprint from the inner
rectangle, and assigns setback labels by their spatial position in the gaps.

The resolver is conservative: if the evidence is not strong enough it returns
None for that field instead of guessing.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

from backend.cv_extraction import skewed_plot
from backend.cv_extraction.raw_types import RawLine, RawRectangle, RawTextItem
from backend.cv_extraction.scale_note import ScaleNote, detect_scale_notes, nearest_scale_note
from backend.schemas.geometry import BoundingBox
from backend.schemas.independent_measurements import IndependentCVResult, IndependentMeasurement

_SITE_RE = re.compile(r"\bSITE\s+PLAN\b", re.I)
# "7.30m Wide Road", "10m WIDE ROAD", "9.14M WIDE ROAD", "25 FEET ROAD".
# The unit word is captured because it is not always metres: PLAN4 states
# its road as "25 FEET ROAD", which read as 25 m would be a four-lane
# highway rather than a 7.62 m residential street.
_ROAD_VALUE_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>m|mt|mtr|metres?|meters?|ft|feet|foot)?\s*"
    r"(?:WIDE\s+)?R\s*O\s*A\s*D\b",
    re.I,
)
_ROAD_UNIT_TO_METRES = {"ft": 0.3048, "feet": 0.3048, "foot": 0.3048}


def _road_width_metres(match) -> float | None:
    """The road width in metres, honouring the unit printed alongside it."""
    try:
        value = float(match.group("value"))
    except (TypeError, ValueError):
        return None
    unit = (match.group("unit") or "").strip().lower()
    return value * _ROAD_UNIT_TO_METRES.get(unit, 1.0)
_NUM_RE = re.compile(r"(?<![A-Za-z])(?P<value>\d*\.\d+|\d+)(?:\s*(?P<unit>mm|cm|m|ft|feet|in|\"|'))?", re.I)
# Feet and inches, with or without the hyphen draughtsmen usually put
# between them: 8'9", 8' 9", 8'-9". The hyphen form was not matched, so
# PLAN4's 8'-9" setbacks parsed as a bare 8 feet -- 2.44 m instead of
# 2.67 m, a 9% error that looked entirely plausible.
# `dimension_candidates._FEET_INCHES_RE` already allowed the hyphen; this
# copy of the same idea had drifted out of step with it.
_FEET_RE = re.compile(r"(?P<ft>\d+(?:\.\d+)?)\s*'\s*-?\s*(?P<inch>\d+(?:\.\d+)?)?\s*\"?", re.I)
# "2.00X1.35", "1.20 X 1.20", "0.90X2.10" -- door/window/room schedule
# notation ("width X height"), always two numbers joined by a bare X/x with
# no unit. A plot/building edge dimension is never written this way (it's
# always a single number with an optional unit), so any text matching this
# shape is schedule-table noise, not a site-plan dimension. Confirmed on a
# real plan: a "2.00X1.35" window-schedule entry was picked up as
# plot.width=2.0 because nothing filtered this pattern out before the first
# number inside it got treated as a normal edge-dimension candidate.
_SCHEDULE_SIZE_RE = re.compile(r"\d+(?:\.\d+)?\s*[xX]\s*\d+(?:\.\d+)?")
# Just the word, for locating the road relative to the plot. Deliberately
# separate from `_ROAD_VALUE_RE` (which needs the width figure too): the
# direction question only needs to know where the road is, and on an
# OCR'd sheet the full "<n>m WIDE ROAD" phrase survives only inside a merged
# line group whose bounding box is useless for position.
_ROAD_WORD_RE = re.compile(r"\broads?\b", re.I)


@dataclass(frozen=True)
class _Rect:
    bbox: BoundingBox

    @property
    def width(self) -> float:
        return self.bbox.width

    @property
    def height(self) -> float:
        return self.bbox.height

    @property
    def area(self) -> float:
        return self.width * self.height


def _bbox_gap(a: BoundingBox, b: BoundingBox) -> float:
    dx = max(a.min_x - b.max_x, b.min_x - a.max_x, 0.0)
    dy = max(a.min_y - b.max_y, b.min_y - a.max_y, 0.0)
    return math.hypot(dx, dy)


def _inside(a: BoundingBox, b: BoundingBox, tol: float = 2.0) -> bool:
    return (
        a.min_x >= b.min_x - tol and a.max_x <= b.max_x + tol
        and a.min_y >= b.min_y - tol and a.max_y <= b.max_y + tol
    )


# Segments shorter than this are sub-pixel rendering noise, not draughting.
# This is NOT a "how long must a boundary line be" threshold -- that check
# happens after collinear segments are merged, in `_dominant_axis_positions`.
_MIN_SEGMENT_LENGTH_PTS = 1.0


def _cluster_lines(lines: Iterable[RawLine], region: BoundingBox, min_length: float = _MIN_SEGMENT_LENGTH_PTS):
    """
    Group axis-aligned segments inside `region` by orientation.

    `min_length` deliberately admits very short segments. It used to be 35pt,
    on the assumption that a boundary is drawn as one long stroke -- but a
    plot boundary is conventionally drawn as a DASHED (dash-dot) property
    line, i.e. as dozens of individually tiny segments. On PLAN5 the plot's
    top edge is 71 separate dashes spanning 249.2pt (17.58 m at the sheet's
    printed 1:200, matching its printed "17.59" label to 0.06%), and the
    longest single dash is 8.9pt. The 35pt filter discarded all 71, so the
    plot rectangle could never be reconstructed and every geometric field on
    that plan came back MISSING.

    Length is now judged on the MERGED run at each axis position rather than
    on individual segments, which treats a dashed line and a solid line the
    same way -- as it should, since they describe the same edge.
    """
    horizontals: list[tuple[float, float, float]] = []  # y, x0, x1
    verticals: list[tuple[float, float, float]] = []  # x, y0, y1
    for raw in lines:
        l = raw.line
        x0, x1 = sorted((l.start.x, l.end.x))
        y0, y1 = sorted((l.start.y, l.end.y))
        if not region.intersects(BoundingBox(min_x=x0, min_y=y0, max_x=x1, max_y=y1)):
            continue
        dx, dy = x1 - x0, y1 - y0
        if dx >= min_length and dy <= 1.5:
            horizontals.append(((y0 + y1) / 2.0, x0, x1))
        elif dy >= min_length and dx <= 1.5:
            verticals.append(((x0 + x1) / 2.0, y0, y1))
    return horizontals, verticals


def _coverage(interval_start: float, interval_end: float, segments: list[tuple[float, float]], tol: float = 2.5) -> float:
    if interval_end <= interval_start:
        return 0.0
    clipped = []
    for a, b in segments:
        lo, hi = max(interval_start, a), min(interval_end, b)
        if hi >= lo - tol:
            clipped.append((lo, hi))
    if not clipped:
        return 0.0
    clipped.sort()
    total = 0.0
    cur_a, cur_b = clipped[0]
    for a, b in clipped[1:]:
        if a <= cur_b + tol:
            cur_b = max(cur_b, b)
        else:
            total += max(0.0, cur_b - cur_a)
            cur_a, cur_b = a, b
    total += max(0.0, cur_b - cur_a)
    return total


# Upper bound on distinct horizontal/vertical axis positions carried into
# rectangle enumeration. Rectangle search is quadratic in each axis, so an
# unbounded axis list makes cost quartic in the number of lines: a dense
# working drawing (or the raster fallback, where every hatching stroke
# becomes a segment) can push this from milliseconds to minutes. When the
# cap binds, the axis lines with the longest total drawn extent are kept --
# plot and building boundaries are among the longest lines in their region,
# while noise is short.
_MAX_AXIS_POSITIONS = 60

# Minimum side length (page-points) for a reconstructed rectangle, and the
# fraction of each side that must actually be drawn for it to count.
_MIN_RECT_SIDE_PTS = 45.0
# Fraction of a rectangle's side that must be inked for a SOLID edge.
_MIN_SIDE_COVERAGE = 0.82

# A dash-dot property line -- the drawing convention for a plot boundary --
# is a real edge that is mostly gap. PLAN5's plot boundary inks only 41% of
# its own length, so no coverage threshold that also excludes noise can
# accept it. What distinguishes it from noise is not how much is inked but
# WHERE: its dashes run end to end across the side, evenly, in numbers.
#
# A dashed side is therefore accepted when its marks SPAN essentially the
# whole side, there are enough of them to be a pattern rather than two stray
# ticks at the corners, and enough ink to be a drawn line at all.
_DASHED_MIN_SPAN_FRACTION = 0.95
_DASHED_MIN_COVERAGE = 0.30
_DASHED_MIN_SEGMENTS = 4
_MAX_RECT_ASPECT = 8.0
_AXIS_SNAP_TOLERANCE_PTS = 2.0

# A drawn rectangle side must actually REACH both of its corners: its first
# and last marks have to start/end within this distance of the interval's
# ends (the larger of an absolute floor for stroke-width noise and a small
# fraction of the side's own length). Coverage alone cannot enforce this --
# on PLAN1 a "plot" rectangle was stitched together from a thin sheet-frame
# line (left/top edges) plus the building's own red outline (right/bottom
# edges), each of which covered >80% of the side it was standing in for but
# stopped 13-19% short of the corner the OTHER object's line occupies. That
# chimera scored above the real, fully-closed plot boundary (which was in the
# candidate pool all along), so the plot, the building nested in it, and all
# four setbacks came out wrong. A genuine drawn edge -- solid or dashed --
# runs corner to corner; this is the same principle the dashed branch's
# `_DASHED_MIN_SPAN_FRACTION` already applies, extended to the solid branch.
_SIDE_END_ANCHOR_MIN_PTS = 4.0
_SIDE_END_ANCHOR_FRACTION = 0.04


def _dominant_axis_positions(
    grouped: list[tuple[float, float, float]], limit: int = _MAX_AXIS_POSITIONS
) -> list[float]:
    """
    Distinct axis positions from `(position, start, end)` segments, capped at
    `limit` and ranked by total drawn extent at that position.

    Architectural sheets redraw the same edge several times with sub-point
    coordinate noise, so positions are first snapped to 0.1pt.
    """
    extent_by_position: dict[float, float] = {}
    for position, start, end in grouped:
        key = round(position, 1)
        extent_by_position[key] = extent_by_position.get(key, 0.0) + max(0.0, end - start)
    # Now that individual dashes are admitted (see `_cluster_lines`), the
    # "is this a real edge" test lives here, on the merged run: a position
    # whose segments do not add up to at least the minimum rectangle side
    # cannot form one, whether it is one stroke or fifty dashes.
    extent_by_position = {
        position: extent
        for position, extent in extent_by_position.items()
        if extent >= _MIN_RECT_SIDE_PTS
    }
    if len(extent_by_position) <= limit:
        return sorted(extent_by_position)
    strongest = sorted(extent_by_position.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return sorted(position for position, _extent in strongest)


def _side_span(interval_start: float, interval_end: float, segments: list[tuple[float, float]]) -> tuple[float, int]:
    """Extent from the first to the last mark inside the interval, and how many marks."""
    inside = [
        (max(interval_start, a), min(interval_end, b))
        for a, b in segments
        if min(interval_end, b) >= max(interval_start, a)
    ]
    if not inside:
        return 0.0, 0
    return max(b for _a, b in inside) - min(a for a, _b in inside), len(inside)


def _side_reaches_both_corners(
    interval_start: float, interval_end: float, segments: list[tuple[float, float]]
) -> bool:
    """Whether marks inside the interval start and end at its two corners."""
    side_length = interval_end - interval_start
    tolerance = max(_SIDE_END_ANCHOR_MIN_PTS, _SIDE_END_ANCHOR_FRACTION * side_length)
    inside = [
        (max(interval_start, a), min(interval_end, b))
        for a, b in segments
        if min(interval_end, b) >= max(interval_start, a)
    ]
    if not inside:
        return False
    first = min(a for a, _b in inside)
    last = max(b for _a, b in inside)
    return (first - interval_start) <= tolerance and (interval_end - last) <= tolerance


def _side_is_drawn(
    interval_start: float, interval_end: float, segments: list[tuple[float, float]]
) -> bool:
    """
    Whether a rectangle side is actually drawn, solid OR dashed.

    Testing inked fraction alone cannot express this: a solid edge inks ~100%
    of its length and a dash-dot property line inks ~40%, so any single
    threshold either rejects real plot boundaries or accepts arbitrary
    collinear noise.
    """
    side_length = interval_end - interval_start
    if side_length <= 0:
        return False
    if not _side_reaches_both_corners(interval_start, interval_end, segments):
        return False
    covered = _coverage(interval_start, interval_end, segments)
    if covered >= _MIN_SIDE_COVERAGE * side_length:
        return True
    span, count = _side_span(interval_start, interval_end, segments)
    return (
        span >= _DASHED_MIN_SPAN_FRACTION * side_length
        and covered >= _DASHED_MIN_COVERAGE * side_length
        and count >= _DASHED_MIN_SEGMENTS
    )


def _rectangles_from_lines(lines: list[RawLine], region: BoundingBox) -> list[_Rect]:
    hs, vs = _cluster_lines(lines, region)
    # Keep only strong, distinct axis lines.  Architectural drawing sheets
    # often repeat the same edge several times with tiny coordinate noise.
    ys = _dominant_axis_positions(hs)
    xs = _dominant_axis_positions(vs)

    # Bucket segments by snapped axis position once, instead of rescanning
    # every segment inside the innermost loop.
    h_segments: dict[float, list[tuple[float, float]]] = {y: [] for y in ys}
    for y, a, b in hs:
        for candidate in ys:
            if abs(y - candidate) <= _AXIS_SNAP_TOLERANCE_PTS:
                h_segments[candidate].append((a, b))
    v_segments: dict[float, list[tuple[float, float]]] = {x: [] for x in xs}
    for x, a, b in vs:
        for candidate in xs:
            if abs(x - candidate) <= _AXIS_SNAP_TOLERANCE_PTS:
                v_segments[candidate].append((a, b))

    rects: list[_Rect] = []
    for i, x0 in enumerate(xs):
        for x1 in xs[i + 1:]:
            width = x1 - x0
            if width < _MIN_RECT_SIDE_PTS:
                continue
            # Horizontal lines that actually span this x-range. Computing
            # this once per x-pair (rather than once per x-pair AND y-pair)
            # is what removes the quartic term: the surviving `spanning_ys`
            # list is typically a handful of entries even when `ys` is long.
            spanning_ys = [y for y in ys if _side_is_drawn(x0, x1, h_segments[y])]
            if len(spanning_ys) < 2:
                continue
            for j, y0 in enumerate(spanning_ys):
                for y1 in spanning_ys[j + 1:]:
                    height = y1 - y0
                    if height < _MIN_RECT_SIDE_PTS:
                        continue
                    if max(width / height, height / width) > _MAX_RECT_ASPECT:
                        continue
                    if not _side_is_drawn(y0, y1, v_segments[x0]):
                        continue
                    if not _side_is_drawn(y0, y1, v_segments[x1]):
                        continue
                    rects.append(
                        _Rect(BoundingBox(min_x=x0, min_y=y0, max_x=x1, max_y=y1))
                    )
    return _dedupe_rectangles(rects)


def _dedupe_rectangles(rects: list[_Rect]) -> list[_Rect]:
    """Collapse rectangles that describe the same edges within line-weight."""
    def nearly_same(a: BoundingBox, b: BoundingBox) -> bool:
        return (
            abs(a.min_x - b.min_x) <= 3.0 and
            abs(a.min_y - b.min_y) <= 3.0 and
            abs(a.max_x - b.max_x) <= 3.0 and
            abs(a.max_y - b.max_y) <= 3.0
        )

    unique: list[_Rect] = []
    for r in sorted(rects, key=lambda x: x.area, reverse=True):
        if any(nearly_same(r.bbox, u.bbox) for u in unique):
            continue
        unique.append(r)
    return unique


# Plot sides in clockwise order in page space (Y grows downward).
_SIDES_CLOCKWISE = ("top", "right", "bottom", "left")


def _front_side_from_road(outer: BoundingBox, road_boxes: list[BoundingBox]) -> str:
    """
    Which geometric side of the plot faces the road.

    The front setback is the one measured toward the road, so every other
    side's identity follows from this. It used to be assumed that the road is
    always at the bottom of the page, which holds only for an upright sheet.
    PLAN5's sheet is drawn rotated 90 degrees, putting its "10m WIDE ROAD" on
    the LEFT -- so all four setbacks were measured correctly and then labelled
    a quarter-turn out: the real 3.00 m front setback was reported as the left
    setback, and the real 1.00 m left setback was reported as the front.
    Values that are individually right but attached to the wrong side are
    especially dangerous downstream, because each one is then checked against
    the wrong regulation.
    """
    if not road_boxes:
        return "bottom"
    centre = outer.center
    nearest = min(road_boxes, key=lambda b: _bbox_gap(outer, b))
    dx = nearest.center.x - centre.x
    dy = nearest.center.y - centre.y
    if abs(dx) > abs(dy):
        return "left" if dx < 0 else "right"
    return "top" if dy < 0 else "bottom"


_NEW_FRONT_EVIDENCE_TIERS = ("FRONT label", "STREET text", "ACCESS text", "MAIN ENTRY text", "GATE text")


def _front_side_from_shared_evidence(
    outer: BoundingBox, road_boxes: list[BoundingBox], text_items: list[RawTextItem], region: BoundingBox,
) -> tuple[str, str] | None:
    """Extend `_front_side_from_road` with the evidence tiers this module
    never checked at all: an explicit "FRONT" text label, or STREET/
    ACCESS/MAIN ENTRY/GATE text -- see `backend.spatial_reasoning.
    front_side.resolve_front_side`'s priority-ordered evidence hierarchy,
    already shared by the DXF pipeline (Rank 3 of the master extraction-
    architecture audit: thread as an ADDITIONAL signal, not a replacement
    of this module's own already-100%-correct ROAD-candidate/fallback
    behavior for the tiers it already implements).

    Returns `(side_name, evidence_level)` only when one of the genuinely
    NEW tiers above actually fired; returns None otherwise so the caller
    falls back to `_front_side_from_road`'s own existing, unchanged
    ROAD-candidate-or-"bottom" behavior -- this function is never trusted
    to make that determination itself, only to catch stronger evidence
    when it exists.
    """
    from backend.schemas.evidence import TextEvidence
    from backend.schemas.geometry import Point, Polygon
    from backend.spatial_reasoning import geometry_utils as geo
    from backend.spatial_reasoning.front_side import resolve_front_side
    from backend.spatial_reasoning.road_access import collect_access_evidence

    text_evidence = [
        TextEvidence(raw_text=t.text or "", page=t.page, bounding_box=t.bounding_box)
        for t in text_items if region.intersects(t.bounding_box)
    ]
    access_evidence = collect_access_evidence(text_evidence, page=None)
    if not any(a.kind in ("front", "street", "access", "main_entry", "gate") for a in access_evidence):
        return None

    b = outer
    plot_polygon = Polygon(points=[
        Point(x=b.min_x, y=b.min_y), Point(x=b.max_x, y=b.min_y),
        Point(x=b.max_x, y=b.max_y), Point(x=b.min_x, y=b.max_y),
    ])
    road_bbox = min(road_boxes, key=lambda box: _bbox_gap(b, box)) if road_boxes else None
    result = resolve_front_side(plot_polygon, road_bbox, access_evidence)
    if result.evidence_level not in _NEW_FRONT_EVIDENCE_TIERS:
        return None
    edges = geo.polygon_edges(plot_polygon)
    try:
        idx = edges.index(result.front_edges[0])
    except (ValueError, IndexError):
        return None
    return _SIDES_CLOCKWISE[idx], result.evidence_level


def _setback_labels_for_front(front_side: str) -> dict[str, str]:
    """
    Map each geometric side to its setback name, given which side is the front.

    Left and right are taken from the viewpoint of someone standing on the
    road looking into the plot, which is the convention building byelaws use.
    With the front at the bottom of an upright sheet this reduces to the
    obvious mapping (bottom=front, top=rear, left=left, right=right).
    """
    if front_side not in _SIDES_CLOCKWISE:
        front_side = "bottom"
    i = _SIDES_CLOCKWISE.index(front_side)
    return {
        front_side: "setbacks.front",
        _SIDES_CLOCKWISE[(i + 2) % 4]: "setbacks.rear",
        _SIDES_CLOCKWISE[(i + 1) % 4]: "setbacks.left",
        _SIDES_CLOCKWISE[(i + 3) % 4]: "setbacks.right",
    }


def _candidate_rectangles(
    lines: list[RawLine], region: BoundingBox, rectangles: list[RawRectangle] | None = None
) -> list[_Rect]:
    """
    Every axis-aligned rectangle in `region`, from BOTH sources a PDF can
    express one with.

    A CAD export draws some rectangles as four separate stroked line
    segments and others as a single `re` operator in the content stream.
    Only the first kind was ever considered here, because the caller
    discarded PyMuPDF's rectangle output. On PLAN5 the building footprint is
    an `re` rectangle measuring 13.09 x 7.14 m -- exactly its printed
    dimensions, and 93.42 sq.m against the sheet's stated 93.46 -- so the
    single most reliable piece of geometry on the drawing was thrown away
    before matching began, and the footprint had to be guessed from
    line-reconstructed rectangles that did not include it.
    """
    found = _rectangles_from_lines(lines, region)
    for raw in rectangles or []:
        box = raw.bounding_box
        if not region.intersects(box):
            continue
        if box.width < _MIN_RECT_SIDE_PTS or box.height < _MIN_RECT_SIDE_PTS:
            continue
        if max(box.width / box.height, box.height / box.width) > _MAX_RECT_ASPECT:
            continue
        found.append(_Rect(box))
    return _dedupe_rectangles(found)


def _find_site_anchor(text_items: list[RawTextItem]) -> RawTextItem | None:
    matches = [t for t in text_items if _SITE_RE.search(t.text or "")]
    if not matches:
        return None
    # Prefer the clearest/longest SITE PLAN label.
    return max(matches, key=lambda t: (len(t.text), -t.bounding_box.min_y))


# The site-plan search window, expressed as fractions of the sheet's own
# dimensions rather than absolute page-points.
#
# These were absolute constants (+-420pt horizontally, -470/+140pt
# vertically). Absolute point offsets encode an assumption about sheet size
# that does not survive contact with real drawings: the same offsets cover
# most of an A3 sheet but a small corner of an A0 one. On the 2384x1684pt
# (A1) sheets in `data/test_plans/` the +-420pt horizontal window reached
# 400pt to the RIGHT of the "SITE PLAN" caption, straight into the
# terms-and-conditions text column, which is how numbered legal clauses
# ("46.Due to non-compliance...") ended up being scored as plot-edge
# dimension labels.
_REGION_HALF_WIDTH_FRACTION = 0.18
_REGION_ABOVE_FRACTION = 0.30
_REGION_BELOW_FRACTION = 0.06

# Fallback widening, used ONLY when the tight window above finds zero
# substantial rectangles. The tight window's whole purpose is avoiding
# noise (see the "terms-and-conditions text" case in the docstring above),
# so it stays the default. But that window's other assumption -- that the
# site-plan drawing is roughly CENTERED on its own caption -- also doesn't
# hold universally: on one real sheet tested, "SITE PLAN" is captioned at
# the drawing's bottom-LEFT rather than centered beneath it, so the tight
# window's right edge fell ~100pt short of the actual plot boundary and
# zero rectangles were ever reconstructed, regardless of how good the line/
# dash-reconstruction logic downstream was. Retrying wider only after the
# tight window has already failed costs nothing on sheets where it
# succeeds, and downstream candidate selection already validates against
# the sheet's own printed area statement rather than trusting the window
# — so a wider, noisier candidate set on a previously-failing sheet is
# still expected to resolve to the right rectangle, not a spurious one.
_REGION_HALF_WIDTH_FRACTION_WIDE = 0.32
_REGION_ABOVE_FRACTION_WIDE = 0.45
_REGION_BELOW_FRACTION_WIDE = 0.10


def _site_region(
    anchor: RawTextItem, page_width: float, page_height: float, *, wide: bool = False
) -> BoundingBox:
    """
    A search window around the "SITE PLAN" caption, sized relative to the
    sheet and oriented to match the caption.

    The caption is printed alongside its drawing, so the window extends much
    further in one direction than the other. Which direction depends on how
    the caption is set: on a sheet laid out in landscape but stored as an
    unrotated portrait page, the whole drawing including its captions is
    rotated 90 degrees, and a caption that reads bottom-to-top has a tall,
    narrow bounding box with its drawing to the SIDE rather than above.
    PLAN5 is exactly this -- 437 of its 522 OCR items are vertical, and its
    "SITE PLAN SCALE 1:200" caption measures 9pt wide by 147pt tall. Applying
    the upright layout assumption there put the window over the area
    statement table instead of the drawing.

    Detecting this from the caption's own aspect ratio needs no page-rotation
    metadata, which is important because the page reports rotation 0 -- the
    rotation is baked into the content stream, not declared.

    This is only a bound on the rectangle search. Which candidate rectangle
    is actually the plot is decided afterwards by agreement with the sheet's
    own area statement, not by this window.
    """
    box = anchor.bounding_box
    c = box.center
    caption_is_rotated = box.height > box.width

    half_width_frac = _REGION_HALF_WIDTH_FRACTION_WIDE if wide else _REGION_HALF_WIDTH_FRACTION
    above_frac = _REGION_ABOVE_FRACTION_WIDE if wide else _REGION_ABOVE_FRACTION
    below_frac = _REGION_BELOW_FRACTION_WIDE if wide else _REGION_BELOW_FRACTION

    if caption_is_rotated:
        # The caption runs along Y, so the drawing sits to one side along X.
        # Both X directions are searched rather than guessing the rotation
        # sense from glyph order, which OCR does not reliably preserve.
        along = page_height * half_width_frac
        across = page_width * above_frac
        return BoundingBox(
            min_x=max(0.0, c.x - across),
            min_y=max(0.0, c.y - along),
            max_x=min(page_width, c.x + across),
            max_y=min(page_height, c.y + along),
        )

    half_width = page_width * half_width_frac
    return BoundingBox(
        min_x=max(0.0, c.x - half_width),
        min_y=max(0.0, c.y - page_height * above_frac),
        max_x=min(page_width, c.x + half_width),
        max_y=min(page_height, c.y + page_height * below_frac),
    )


def _numbers_in_region(text_items: list[RawTextItem], region: BoundingBox):
    out = []
    for item in text_items:
        if getattr(item, "is_line_group", False):
            # Numeric candidates need the precise per-word bounding box for
            # distance-based edge/gap picking (`_pick_edge_dimension`); a
            # merged multi-word line's wider bbox would add imprecise/
            # duplicate candidates for the same value already present via
            # its own word-level item. Line-group items exist only for
            # phrase matching (site anchor, road/ID/area labels), never for
            # numeric-value extraction -- see `RawTextItem.is_line_group`.
            continue
        if not region.intersects(item.bounding_box):
            continue
        text = item.text.strip()
        if not text:
            continue
        # Skip obvious area statements and IDs.
        if any(tok in text.lower() for tok in ("sq.m", "sqm", "sq.ft", "sqft", "bearing", "scale")):
            continue
        # Skip door/window/room schedule size notation ("2.00X1.35") -- see
        # `_SCHEDULE_SIZE_RE` above. Must come before the feet/inches and
        # general numeric regexes below, since both would otherwise happily
        # extract the first number out of a schedule entry as if it were a
        # real edge dimension.
        if _SCHEDULE_SIZE_RE.search(text):
            continue
        m = _FEET_RE.search(text)
        if m:
            ft = float(m.group("ft")); inch = float(m.group("inch") or 0)
            if inch < 12:
                out.append(((ft + inch / 12.0) * 0.3048, item, "ft_in"))
                continue
        for m in _NUM_RE.finditer(text):
            try:
                value = float(m.group("value"))
            except ValueError:
                continue
            unit = (m.group("unit") or "").lower()
            if unit in ("mm",): value /= 1000.0
            elif unit in ("cm",): value /= 100.0
            elif unit in ("ft", "feet", "'"): value *= 0.3048
            elif unit in ("in", '"'): value *= 0.0254
            if 0 < value <= 60:
                out.append((value, item, unit or "m"))
    return out


def _near(values, target: float, tol: float = 0.08):
    return [v for v in values if abs(v[0] - target) <= tol]


def _pick_edge_dimension(nums, outer: BoundingBox, orientation: str, road_text_boxes: list | None = None):
    candidates = []
    road_text_boxes = road_text_boxes or []
    for value, item, unit in nums:
        text = (item.text or "").strip()
        if re.search(r"r\s*o\s*a\s*d", text, re.I):
            continue
        # The substring check above only catches "road" landing inside the
        # SAME token as the number, which basically never happens with
        # OCR-sourced numbers (Tesseract emits "10" and "ROAD" as separate
        # word-level items -- see `RawTextItem.is_line_group`). Without this,
        # a road-width label like "10m WIDE ROAD" gets misread as a
        # plot/building edge dimension whenever it sits just outside the
        # candidate rectangle, which is exactly where road labels live.
        # `road_text_boxes` comes from scanning ALL text items (word +
        # line-group) for the full "<value> WIDE ROAD" phrase, so this
        # exclusion works regardless of which granularity supplied it.
        if any(item.bounding_box.intersects(rb) for rb in road_text_boxes):
            continue
        c = item.bounding_box.center
        if orientation == "horizontal":
            outside = c.y < outer.min_y - 1.0 or c.y > outer.max_y + 1.0
            overlap = c.x >= outer.min_x - 30 and c.x <= outer.max_x + 30
            d = min(abs(c.y - outer.min_y), abs(c.y - outer.max_y))
            if outside and overlap and d <= 90:
                candidates.append((d, value, item))
        else:
            outside = c.x < outer.min_x - 1.0 or c.x > outer.max_x + 1.0
            overlap = c.y >= outer.min_y - 30 and c.y <= outer.max_y + 30
            d = min(abs(c.x - outer.min_x), abs(c.x - outer.max_x))
            if outside and overlap and d <= 90:
                candidates.append((d, value, item))
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1:] if candidates else None


# How many labels per edge to consider before settling on one.
_EDGE_LABEL_CANDIDATES = 4


def _edge_dimension_candidates(
    nums, outer: BoundingBox, orientation: str, road_text_boxes: list | None = None
) -> list[tuple[float, object]]:
    """
    The plausible dimension labels for one edge, nearest first.

    `_pick_edge_dimension` returns only the nearest, which is wrong whenever
    a drawing stacks dimension lines -- the near-universal convention of
    printing an overall dimension outside a chain of partial ones. PLAN4's
    site plan is dimensioned 3'-0" | 54'-0" | 3'-0" with 60'-0" above it, so
    the nearest label to the plot's top edge is a 0.91 m setback, and reading
    it as the plot width made the whole sheet unresolvable. Which of the
    stacked labels is the edge's own dimension cannot be settled locally --
    it is decided by which choice makes the rest of the drawing consistent --
    so the choice is deferred to the caller.
    """
    scored: list[tuple[float, float, object]] = []
    road_text_boxes = road_text_boxes or []
    for value, item, _unit in nums:
        text = (item.text or "").strip()
        if re.search(r"r\s*o\s*a\s*d", text, re.I):
            continue
        if any(item.bounding_box.intersects(rb) for rb in road_text_boxes):
            continue
        c = item.bounding_box.center
        if orientation == "horizontal":
            outside = c.y < outer.min_y - 1.0 or c.y > outer.max_y + 1.0
            overlap = outer.min_x - 30 <= c.x <= outer.max_x + 30
            distance = min(abs(c.y - outer.min_y), abs(c.y - outer.max_y))
        else:
            outside = c.x < outer.min_x - 1.0 or c.x > outer.max_x + 1.0
            overlap = outer.min_y - 30 <= c.y <= outer.max_y + 30
            distance = min(abs(c.x - outer.min_x), abs(c.x - outer.max_x))
        if not (outside and overlap and distance <= 90):
            continue
        if value < 2.0:
            continue
        scored.append((distance, value, item))
    scored.sort(key=lambda t: t[0])
    return [(value, item) for _d, value, item in scored[:_EDGE_LABEL_CANDIDATES]]


def _pick_gap_dimension(
    nums, outer: BoundingBox, inner: BoundingBox, side: str, max_setback_m: float | None = None
):
    """
    Find the printed setback label sitting in the plot/building gap on `side`.

    `max_setback_m` bounds what counts as a plausible setback reading. This
    used to be a hardcoded `value > 3.0` cut, on the reasoning that "setback
    labels on normal urban plans are small". That is not a property of
    setbacks, it is a property of small plots: BBMP's own setback table
    requires 6m+ front setbacks once a plot exceeds roughly 24m of depth, so
    the constant silently discarded every correct reading on a larger plot.
    Callers now derive the bound from the plot's own measured geometry --
    a setback cannot exceed the gap it is annotating.
    """
    candidates = []
    for value, item, unit in nums:
        c = item.bounding_box.center
        if max_setback_m is not None and value > max_setback_m:
            continue
        if side == "top" and outer.min_y <= c.y <= inner.min_y and outer.min_x <= c.x <= outer.max_x:
            distance = abs(c.y - (outer.min_y + inner.min_y) / 2)
        elif side == "bottom" and inner.max_y <= c.y <= outer.max_y and outer.min_x <= c.x <= outer.max_x:
            distance = abs(c.y - (inner.max_y + outer.max_y) / 2)
        elif side == "left" and outer.min_x <= c.x <= inner.min_x and outer.min_y <= c.y <= outer.max_y:
            distance = abs(c.x - (outer.min_x + inner.min_x) / 2)
        elif side == "right" and inner.max_x <= c.x <= outer.max_x and outer.min_y <= c.y <= outer.max_y:
            distance = abs(c.x - (inner.max_x + outer.max_x) / 2)
        else:
            continue
        candidates.append((distance, value, item))
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1:] if candidates else None


_AREA_LABELS = [
    ("plot.area", re.compile(r"^AREA\s+OF\s+PLOT\s*\(\s*Minimum\s*\)", re.I), "m2"),
    # Not every authority uses BBMP's "AREA OF PLOT (Minimum)" wording --
    # PLAN5's sheet states "SITE AREA : 160.77 Sq.m", which matched none of
    # the patterns above, so this measurement (and the `final_fusion`
    # AGREED/CONFLICT machinery downstream of it) had nothing to work with
    # even though `_area_targets()` below already recognises this exact
    # wording for its own internal rectangle-selection cross-check.
    # Deliberately reusing that already-proven, unanchored pattern here
    # rather than inventing a new one; deliberately NOT anchored with `^`
    # for the same reason documented on `_AREA_TARGET_LABELS` -- OCR on a
    # rotated sheet can merge this label into a longer run of neighbouring
    # text. `plot.area` is checked first in the loop below and the loop
    # keeps only the first pattern that matches per field, so a sheet using
    # BBMP's own wording is unaffected.
    ("plot.area", re.compile(r"\bSITE\s+AREA\b", re.I), "m2"),
    # The NET plot area (gross minus any road-widening deduction) is a
    # DIFFERENT physical quantity bounded by a DIFFERENT rectangle on the
    # drawing, so it gets its own field. Both used to be emitted as
    # "plot.area", which meant a sheet stating both (the common case on a
    # BBMP sanctioned plan) produced two contradictory values for one field
    # and whichever was consumed last silently won.
    ("plot.net_area", re.compile(r"^NET\s+AREA\s+OF\s+PLOT", re.I), "m2"),
    ("building.footprint_area", re.compile(r"^PROPOSED\s+COVERAGE\s+AREA", re.I), "m2"),
    # Same generalisation as plot.area above, reusing the synonyms
    # `_area_targets()` already proved out on real sheets: "GROUND COVERAGE
    # AREA", "G.F AREA" (ground-floor area used as the footprint on a sheet
    # that tabulates area per floor), and "PROP. PLINTH AREA IN G.F".
    ("building.footprint_area", re.compile(r"\bGROUND\s+COVERAGE\s+AREA\b", re.I), "m2"),
    ("building.footprint_area", re.compile(r"\bG\.?\s*F\.?\s+AREA\b", re.I), "m2"),
    ("building.footprint_area", re.compile(r"\bPLINTH\s+AREA\s+IN\s+G\.?\s*F\.?", re.I), "m2"),
    ("coverage", re.compile(r"^PROPOSED\s+COVERAGE\s+AREA", re.I), "%"),
    # Standalone "Coverage %" / "% Coverage Achieved" style labels, where the
    # percentage is a separate table cell rather than embedded in the label
    # text itself -- handled by the percent-cell fallback in
    # `_area_measurements` below, not the inline-parenthetical extraction
    # `^PROPOSED COVERAGE AREA` needs.
    ("coverage", re.compile(r"\bCOVERAGE\b.{0,20}%|%.{0,20}\bCOVERAGE\b", re.I), "%"),
    ("far.area", re.compile(r"^PROPOSED\s+FAR\s+AREA", re.I), "m2"),
    ("far", re.compile(r"^ACHIEVED\s+NET\s+FAR\s+AREA", re.I), "ratio"),
    ("far.area", re.compile(r"^ACHIEVED\s+NET\s+FAR\s+AREA", re.I), "m2"),
    # Standalone "FAR Achieved" / "FSI Achieved" labels with the ratio in its
    # own cell (handled by the ratio-cell fallback below).
    ("far", re.compile(r"\b(?:FAR|FSI)\b.{0,20}\bACHIEVED\b|\bACHIEVED\b.{0,20}\b(?:FAR|FSI)\b", re.I), "ratio"),
    ("building.gross_built_up_area", re.compile(r"^PROPOSED\s+BUILTUP\s+AREA", re.I), "m2"),
    ("building.gross_built_up_area", re.compile(r"\bTOTAL\s+BUILT[\s-]?UP\s+AREA\b", re.I), "m2"),
]

# A plot/footprint/built-up area figure this small is almost certainly the
# wrong table cell (e.g. an adjacent setback or floor-count number), not a
# genuine area -- same reasoning and threshold as `_MIN_PLAUSIBLE_AREA_M2`
# used by `_area_targets()`. Only applied to the magnitude (m2) fields; a
# coverage percentage or FAR ratio has its own, much smaller, plausible
# range and is bounds-checked separately below.
_AREA_MAGNITUDE_FIELDS = {"plot.area", "plot.net_area", "building.footprint_area", "far.area", "building.gross_built_up_area"}
_NUM_ITEM_RE = re.compile(r"(?<![A-Za-z])(?:\d+(?:\.\d+)?|\.\d+)(?![A-Za-z])")

# Area-statement labels used to CROSS-CHECK reconstructed geometry, keyed by
# what the value physically bounds. Distinct from `_AREA_LABELS` above, which
# reports areas as measurements in their own right: here both the gross and
# the net plot area matter separately, because they correspond to two
# different rectangles on the drawing (the plot boundary, and the plot
# boundary minus a road-widening strip).
_AREA_TARGET_LABELS = [
    ("plot_area_gross", re.compile(r"^AREA\s+OF\s+PLOT\s*\(\s*Minimum\s*\)", re.I)),
    # Not every authority uses BBMP's wording. PLAN5's sheet states
    # "SITE AREA : 160.77 Sq.m", which matched none of the patterns above,
    # so the area cross-check had nothing to validate against and the
    # resolver abstained on a plan whose geometry was in fact recoverable.
    # Deliberately NOT anchored with `^`: OCR on a rotated sheet merges this
    # label into a longer run of neighbouring text
    # ("3 5e8340 on 00X00 SITE AREA : 160.77 Sq.m" on PLAN5), so anchoring
    # meant it never matched on exactly the scanned plans that need it most.
    ("plot_area_gross", re.compile(r"SITE\s+AREA\b", re.I)),
    ("plot_area_net", re.compile(r"^NET\s+AREA\s+OF\s+PLOT", re.I)),
    ("building_footprint_area", re.compile(r"^PROPOSED\s+COVERAGE\s+AREA", re.I)),
    ("building_footprint_area", re.compile(r"GROUND\s+COVERAGE\s+AREA", re.I)),
    # "G.F AREA" -- ground-floor area, i.e. the footprint, in a sheet that
    # tabulates area per floor rather than naming a coverage area.
    ("building_footprint_area", re.compile(r"\bG\.?\s*F\.?\s+AREA\b", re.I)),
    # "PROP. PLINTH AREA IN G.F" -- same concept, different word order
    # (label ends in G.F rather than starting with it) and phrased as
    # "plinth area" rather than "coverage"/"G.F area". A real sheet states
    # its ground-floor plinth area exactly this way, with identical values
    # repeated for each upper floor (G.F/F.F/S.F/T.F all equal) confirming
    # a constant footprint across floors -- i.e. this genuinely is the
    # building footprint, not a per-floor built-up figure that could differ
    # from the footprint on a stepped/tapering building.
    ("building_footprint_area", re.compile(r"PLINTH\s+AREA\s+IN\s+G\.?\s*F\.?", re.I)),
]

# "SITE AREA : 160.77 Sq.m" -- label and value in a single text run, rather
# than as two cells of a table. Both layouts occur on real sheets.
# Smallest value that could be a plot or building-footprint area, in sq m.
_MIN_PLAUSIBLE_AREA_M2 = 10.0

_INLINE_AREA_VALUE_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?:sq\.?\s*m|sqm|m2|m²|smt)\b", re.I
)
# A number that is NOT a percentage. The negative lookahead is essential:
# BBMP sheets write the coverage label as "Proposed Coverage Area (79.15 %)"
# with the actual area in the next table cell, so an unguarded "first number
# after the label" reads the percentage as if it were an area in sq m.
# The digit boundaries on both sides are not decoration. Without the
# trailing `(?![\d.])` the engine backtracks around the percent guard and
# matches a PREFIX of the number instead: "(79.15 %)" yielded 79.1, which is
# both wrong and plausible-looking.
_ANY_NUMBER_RE = re.compile(r"(?<![\d.])(?P<value>\d+(?:\.\d+)?)(?![\d.])(?!\s*%)")


def _inline_area_value(text: str, search_from: int = 0) -> float | None:
    """
    The area figure stated inside a label run itself, if there is one.

    Searches only AFTER the label, and takes the FIRST number found rather
    than the last. Both matter: an area table is commonly emitted as one text
    run per row carrying every column at once
    ("SITE AREA    303.79    3270"), and its first numeric column is the
    metric one -- taking the last would have read PLAN4's site area as
    3270 sq m instead of 303.79, since the trailing column is square feet.
    A value that is immediately followed by an area unit still wins outright,
    which covers "SITE AREA : 160.77 Sq.m".
    """
    tail = text[search_from:]
    for pattern in (_INLINE_AREA_VALUE_RE, _ANY_NUMBER_RE):
        match = pattern.search(tail)
        if match:
            try:
                return float(match.group("value"))
            except ValueError:
                continue
    return None


def _area_targets(text_items: list[RawTextItem]) -> dict[str, float]:
    """
    The sheet's own stated plot/coverage areas, in square metres.

    These are the closed-loop check that makes geometry reconstruction
    trustworthy without any printed edge dimension: a candidate rectangle
    measured at the sheet's printed scale either reproduces the sheet's
    stated area or it does not. On the plans in `data/test_plans/` the
    correct rectangle matches to within 0.2%, while the dozens of competing
    dimension frames, hatching boxes and title blocks do not come close.
    """
    ordered = sorted(text_items, key=lambda t: (t.bounding_box.min_y, t.bounding_box.min_x))
    targets: dict[str, float] = {}
    for key, pattern in _AREA_TARGET_LABELS:
        if key in targets:
            continue
        for label_item in ordered:
            label_text = (label_item.text or "").strip()
            label_match = pattern.search(label_text)
            if not label_match:
                continue
            value = _inline_area_value(label_text, label_match.end())
            if value is None:
                numeric_item = _nearest_numeric_item(label_item, ordered)
                if numeric_item is None:
                    continue
                try:
                    value = float(numeric_item.text.strip())
                except ValueError:
                    continue
            # A plot or footprint area is never a fraction of a square metre.
            # Without this floor, "GROUND COVERAGE AREA" on PLAN5 latched onto
            # the neighbouring "3.00" setback cell and asserted a 3 sq.m
            # building footprint.
            if value < _MIN_PLAUSIBLE_AREA_M2:
                # Keep looking: an implausible value means this label
                # occurrence was paired with the wrong cell, not that the
                # sheet lacks the figure. Breaking here let one bad pairing
                # suppress a correct one later on the same sheet.
                continue
            targets[key] = value
            break
    return targets


# A reconstructed rectangle counts as reproducing a stated area when it is
# within this relative tolerance. Vector line coordinates and the printed
# scale are both exact, so real agreement is well inside 1%; the allowance
# covers line-weight offsets on where an edge's centreline actually sits.
_AREA_MATCH_TOLERANCE = 0.03


def _area_agreement(rect: _Rect, target_m2: float, points_per_metre: float) -> float | None:
    """Relative error between a rectangle's real-world area and `target_m2`."""
    if points_per_metre is None or points_per_metre <= 0 or target_m2 <= 0:
        return None
    area_m2 = (rect.width / points_per_metre) * (rect.height / points_per_metre)
    return abs(area_m2 - target_m2) / target_m2


# A candidate whose gap to its container is under this fraction of the
# container's own side length, on BOTH opposite sides at once, counts as
# "flush" on that axis -- i.e. it spans nearly the container's full width
# or full depth, leaving no room for a setback on that entire axis.
_FLUSH_GAP_FRACTION = 0.05


def _spans_container_on_an_axis(bbox: BoundingBox, container: BoundingBox) -> bool:
    """True if `bbox` is flush with `container`'s full width or full depth
    -- near-zero gap on both opposite sides of one axis at once."""
    w_tol = container.width * _FLUSH_GAP_FRACTION
    h_tol = container.height * _FLUSH_GAP_FRACTION
    spans_x = (bbox.min_x - container.min_x) <= w_tol and (container.max_x - bbox.max_x) <= w_tol
    spans_y = (bbox.min_y - container.min_y) <= h_tol and (container.max_y - bbox.max_y) <= h_tol
    return spans_x or spans_y


def _best_rect_for_area(
    rects: list[_Rect], target_m2: float | None, points_per_metre: float | None,
    *, container_bbox: BoundingBox | None = None,
) -> tuple[_Rect, float] | None:
    """The rectangle whose scaled area best reproduces `target_m2`, if any does.

    When `container_bbox` (the plot/outer rectangle this candidate is
    nested inside) is given, a candidate that spans nearly the FULL width
    or depth of its container is deprioritized relative to one with real
    margin on all sides, even when its own area match is marginally
    closer. Confirmed directly on a real sheet (PLAN1.pdf): two nested
    candidates both reproduced the stated footprint area to within 0.5%
    (0.30% and 0.47% error), and picking strictly by smaller error chose
    the one flush with the plot's own width across its ENTIRE horizontal
    extent -- almost certainly a duplicate/compound-wall/hatch trace
    rather than the real building footprint, since a real footprint is
    inset from its plot by a setback on every side by definition. That
    wrong pick then zeroed out two of the four derived setbacks (nothing
    left to measure on an axis with no gap at all).
    """
    if target_m2 is None or points_per_metre is None or points_per_metre <= 0:
        return None
    scored = []
    for rect in rects:
        error = _area_agreement(rect, target_m2, points_per_metre)
        if error is not None and error <= _AREA_MATCH_TOLERANCE:
            flush = _spans_container_on_an_axis(rect.bbox, container_bbox) if container_bbox else False
            scored.append((flush, error, rect))
    if not scored:
        return None
    flush, error, rect = min(scored, key=lambda triple: (triple[0], triple[1]))
    return rect, error


def _nearest_numeric_item(label_item: RawTextItem, items: list[RawTextItem]) -> RawTextItem | None:
    """Find the numeric value in the same table row, normally far to the right.

    Requires only that the candidate sit to the right of the label (`dx >=
    0`), matching `_nearest_percent_item`'s own convention for the same
    same-row lookup task. A stricter `dx < 80` floor here previously
    discarded a genuinely correct, nearby value: on a real sheet
    ("PROP. PLINTH AREA IN G.F  96.34  1037", the metric figure ~77pt from
    the label and its square-foot twin ~139pt further right, both in the
    same row), the closer/correct sq-m figure was rejected outright by that
    80pt floor, leaving the sq-ft figure as the only surviving candidate and
    silently shipping an 11x-too-large "sq m" area. The `score` below (which
    already prefers the CLOSER of any surviving candidates) is what should
    do this job, not an arbitrary minimum gap.
    """
    best = None
    best_score = float("inf")
    ly = label_item.bounding_box.center.y
    for item in items:
        if item is label_item:
            continue
        text = (item.text or "").strip()
        if not _NUM_ITEM_RE.fullmatch(text):
            continue
        c = item.bounding_box.center
        dy = abs(c.y - ly)
        dx = c.x - label_item.bounding_box.max_x
        if dx < 0 or dy > 18:
            continue
        score = dy * 20.0 + dx * 0.01
        if score < best_score:
            best_score = score
            best = item
    return best


_PERCENT_CELL_RE = re.compile(r"^(?P<value>\d+(?:\.\d+)?)\s*%$")


def _nearest_percent_item(label_item: RawTextItem, items: list[RawTextItem]) -> tuple[RawTextItem, float] | None:
    """Same row/proximity rule as `_nearest_numeric_item`, for a standalone '<num>%' cell.

    Kept separate from `_nearest_numeric_item` (whose `_NUM_ITEM_RE.fullmatch`
    rejects a trailing '%') rather than relaxing that shared helper, since
    plot/footprint/built-up-area magnitude lookups must never accidentally
    match a percentage cell.
    """
    best = None
    best_score = float("inf")
    ly = label_item.bounding_box.center.y
    for item in items:
        if item is label_item:
            continue
        m = _PERCENT_CELL_RE.match((item.text or "").strip())
        if not m:
            continue
        c = item.bounding_box.center
        dy = abs(c.y - ly)
        dx = c.x - label_item.bounding_box.max_x
        if dx < 0 or dy > 18:
            continue
        score = dy * 20.0 + dx * 0.01
        if score < best_score:
            best_score = score
            best = (item, float(m.group("value")))
    return best


# A coverage percentage this large, or a FAR/FSI ratio this large, is
# printed nowhere in Indian municipal practice and almost certainly means
# the nearby-cell fallback below latched onto an unrelated number (e.g. a
# plot area in the thousands misread as a ratio). Same defensive-bound
# spirit as `_MIN_PLAUSIBLE_AREA_M2`.
_MAX_PLAUSIBLE_COVERAGE_PCT = 100.0
_MAX_PLAUSIBLE_FAR_RATIO = 10.0


def _area_measurements(text_items: list[RawTextItem], page_number: int) -> list[IndependentMeasurement]:
    """Extract explicitly labelled area/coverage/FAR values from native PDF text.

    Label/value association is spatial, not a loose regex over the whole page.
    This prevents unrelated numbers (room dimensions, doors, etc.) from being
    accidentally captured as an area value.
    """
    ordered = sorted(text_items, key=lambda t: (t.bounding_box.min_y, t.bounding_box.min_x))
    found: list[IndependentMeasurement] = []
    seen: set[tuple[str, float]] = set()
    for field, pattern, unit in _AREA_LABELS:
        for label_item in ordered:
            label = (label_item.text or "").strip()
            label_match = pattern.search(label)
            if not label_match:
                continue
            value: float | None = None
            evidence: str | None = None
            bbox_item = label_item

            if field == "coverage":
                pct_match = re.search(r"\(\s*(\d+(?:\.\d+)?)\s*%", label)
                if pct_match:
                    value = float(pct_match.group(1))
                    evidence = label
                else:
                    # Standalone "Coverage %" label with the percentage in
                    # its own table cell rather than embedded in the label.
                    percent_hit = _nearest_percent_item(label_item, ordered)
                    if percent_hit is not None:
                        bbox_item, value = percent_hit
                        evidence = f"{label} -> {bbox_item.text.strip()}"
                if value is not None and not (0.0 < value <= _MAX_PLAUSIBLE_COVERAGE_PCT):
                    value = None
            elif field == "far":
                ratio_match = re.search(r"\(\s*(\d+(?:\.\d+)?)\s*\)", label)
                if ratio_match:
                    value = float(ratio_match.group(1))
                    evidence = label
                else:
                    # Standalone "FAR/FSI Achieved" label with the ratio in
                    # its own cell. Reuses `_nearest_numeric_item` (a plain
                    # number, no '%') but bounds it to a plausible ratio so
                    # an unrelated large number (an area in the thousands)
                    # is not misread as FAR.
                    numeric_item = _nearest_numeric_item(label_item, ordered)
                    if numeric_item is not None:
                        try:
                            candidate = float(numeric_item.text.strip())
                        except ValueError:
                            candidate = None
                        if candidate is not None and 0.0 < candidate <= _MAX_PLAUSIBLE_FAR_RATIO:
                            bbox_item, value = numeric_item, candidate
                            evidence = f"{label} -> {numeric_item.text.strip()}"
            else:
                # Try the value embedded in the SAME text run first (e.g.
                # "SITE AREA : 160.77 Sq.m", one OCR/PDF text item) before
                # looking for a separate table cell -- the same two-step
                # approach `_area_targets()` already uses for its internal
                # geometry cross-check, and necessary for the same reason:
                # a label picked up by the newly-generalised patterns above
                # is just as likely to have its value inline as in its own
                # cell.
                candidate = _inline_area_value(label, label_match.end())
                if candidate is not None and (
                    field not in _AREA_MAGNITUDE_FIELDS or candidate >= _MIN_PLAUSIBLE_AREA_M2
                ):
                    value = candidate
                    evidence = label
                else:
                    numeric_item = _nearest_numeric_item(label_item, ordered)
                    if numeric_item is not None:
                        try:
                            candidate = float(numeric_item.text.strip())
                        except ValueError:
                            candidate = None
                        if candidate is not None and (
                            field not in _AREA_MAGNITUDE_FIELDS or candidate >= _MIN_PLAUSIBLE_AREA_M2
                        ):
                            bbox_item, value = numeric_item, candidate
                            evidence = f"{label} -> {numeric_item.text.strip()}"

            if value is None:
                continue
            key = (field, round(value, 4))
            if key in seen:
                continue
            seen.add(key)
            found.append(IndependentMeasurement(
                field=field,
                value=value,
                unit=unit,
                source="NATIVE_TEXT",
                confidence=0.99,
                evidence=[evidence or label],
                note="Explicit labelled area/coverage/FAR value from the PDF native text layer; no Vision input.",
                page=page_number,
                geometry_bbox_pts=[round(bbox_item.bounding_box.min_x,2), round(bbox_item.bounding_box.min_y,2), round(bbox_item.bounding_box.max_x,2), round(bbox_item.bounding_box.max_y,2)],
            ))
    return found

def _measurement(field: str, value: float | None, source: str | None, confidence: float, evidence: list[str], note: str, page: int, bbox: BoundingBox | None = None):
    return IndependentMeasurement(
        field=field,
        value_m=None if value is None else round(float(value), 4),
        source=source,
        confidence=confidence,
        evidence=evidence,
        note=note,
        page=page,
        geometry_bbox_pts=(None if bbox is None else [round(bbox.min_x,2),round(bbox.min_y,2),round(bbox.max_x,2),round(bbox.max_y,2)]),
    )


# The scale a drawing states in print and the scale implied by its own
# dimension labels must agree; if they disagree by more than this, something
# is wrong (mis-read label, wrong note associated with the view) and neither
# is asserted confidently.
_SCALE_AGREEMENT_TOLERANCE = 0.05

# A setback cannot be larger than the plot side it is measured across. Used
# to bound plausible setback readings in place of a hardcoded metre value.
_MAX_SETBACK_AS_PLOT_FRACTION = 0.5

# Physical bounds on a single plot side, in metres. Any candidate rectangle
# that scales outside this is not a plot boundary -- it is a detail view, a
# title block, a legend box, or the same drawing read at the wrong scale.
# The lower bound in particular is what stops a detail-view scale note (a
# sheet may print 1:25, 1:50, 1:75 and 1:100 side by side) from being applied
# to the site plan: at 1:25 a 10m plot measures 0.94m, which is rejected here.
_MIN_PLOT_SIDE_M = 3.0
_MAX_PLOT_SIDE_M = 200.0

# An explicit edge label is only believed when it agrees with the same edge
# measured off the geometry at the resolved scale. This is what prevents a
# nearby non-dimension number from being adopted as a plot dimension: on a
# real sheet a rotated "SITE NO-09" caption sits exactly where a depth label
# would, and was read as plot.depth = 9.0 m on a plot 13.10 m deep.
_EDGE_LABEL_AGREEMENT_TOLERANCE = 0.06

# How far a nested rectangle's edge may fall OUTSIDE its parent's before it
# stops counting as nested. Non-zero only to absorb line-weight: a boundary
# drawn with a wide stroke puts the two centrelines a fraction of a point
# apart even where the building genuinely abuts the plot line.
_NESTED_EDGE_TOLERANCE_PTS = 1.5


def _distinct_scale_notes(notes: list[ScaleNote], region_centre) -> list[ScaleNote | None]:
    """
    The distinct scales worth trying for a drawing region, best guess first,
    always ending with `None` (meaning "no printed scale -- fall back to
    edge-label-derived scale only").

    Notes are deduplicated by denominator, because the same scale printed
    under several views is one hypothesis, not several. Ordering is by
    whether the note carried the literal word "scale" and then by distance
    to the region, which only breaks ties: the actual choice is made by
    scoring each hypothesis against the drawing in `extract_site_plan_measurements`.
    """
    by_denominator: dict[int, ScaleNote] = {}
    for note in notes:
        existing = by_denominator.get(note.denominator)
        if existing is None or (
            (not existing.has_scale_keyword, existing.distance_to(region_centre))
            > (not note.has_scale_keyword, note.distance_to(region_centre))
        ):
            by_denominator[note.denominator] = note
    ordered = sorted(
        by_denominator.values(),
        key=lambda n: (not n.has_scale_keyword, n.distance_to(region_centre)),
    )
    return [*ordered, None]


def _plausible_plot_rect(rect: _Rect, points_per_metre: float | None) -> bool:
    """Whether `rect` could physically be a plot boundary at this scale."""
    if points_per_metre is None or points_per_metre <= 0:
        return True  # no scale to judge by; other signals must decide
    width_m = rect.width / points_per_metre
    depth_m = rect.height / points_per_metre
    return all(_MIN_PLOT_SIDE_M <= side <= _MAX_PLOT_SIDE_M for side in (width_m, depth_m))


def _label_agrees_with_geometry(
    label_value_m: float | None, side_pts: float, points_per_metre: float | None
) -> bool:
    """Whether an explicit edge label matches that edge measured at `points_per_metre`."""
    if label_value_m is None or points_per_metre is None or points_per_metre <= 0:
        return False
    measured_m = side_pts / points_per_metre
    if measured_m <= 0:
        return False
    return abs(label_value_m - measured_m) / measured_m <= _EDGE_LABEL_AGREEMENT_TOLERANCE


def _resolve_scale(
    label_samples: list[float], printed: ScaleNote | None
) -> tuple[float | None, float | None, str]:
    """
    Reconcile the two independent statements of a drawing's scale.

    Returns `(points_per_metre, confidence, explanation)`.

    Priority is deliberate. When printed and label-derived scales AGREE they
    corroborate each other and the result is the most trustworthy value the
    resolver can produce. When only one exists it is used on its own. When
    both exist and DISAGREE, the printed note wins -- it is an exact
    statement by the drafting software, whereas a label-derived sample
    depends on having correctly associated a number with an edge, which is
    exactly the association this module is trying to establish.
    """
    label_scale = sum(label_samples) / len(label_samples) if label_samples else None

    if printed is not None and label_scale is not None:
        disagreement = abs(printed.points_per_metre - label_scale) / printed.points_per_metre
        if disagreement <= _SCALE_AGREEMENT_TOLERANCE:
            return (
                printed.points_per_metre,
                0.99,
                f"printed drawing scale 1:{printed.denominator} "
                f"({printed.points_per_metre:.4f} pt/m) corroborated by edge-label-derived "
                f"scale ({label_scale:.4f} pt/m, {disagreement:.2%} apart)",
            )
        return (
            printed.points_per_metre,
            0.80,
            f"printed drawing scale 1:{printed.denominator} "
            f"({printed.points_per_metre:.4f} pt/m) used, but the edge-label-derived scale "
            f"({label_scale:.4f} pt/m) disagrees by {disagreement:.2%} -- confidence reduced",
        )

    if printed is not None:
        return (
            printed.points_per_metre,
            0.95,
            f"printed drawing scale 1:{printed.denominator} "
            f"({printed.points_per_metre:.4f} pt/m); no edge dimension label available to "
            "corroborate it",
        )

    if label_scale is not None:
        consistent = (
            len(label_samples) == 2
            and abs(label_samples[0] - label_samples[1]) / label_scale < 0.02
        )
        return (
            label_scale,
            0.99 if consistent else 0.80,
            f"scale derived from {len(label_samples)} explicit edge dimension label(s); "
            "no printed scale note found on the sheet",
        )

    return None, None, "no printed scale note and no explicit edge dimension label"


# Cap on any measurement produced by the rectangle model on a plot known to have a
# slanted edge it cannot represent (below the LOW/MEDIUM boundary).
_SKEW_UNRESOLVED_CONFIDENCE_CAP = 0.4


def _extract_from_skewed_plot(
    text_items, lines, region, anchor, page_number, scale_notes, rectangles, notes, *, strong_evidence: bool,
):
    """Plot/building/setbacks for a plot bounded by slanted edges (see
    `skewed_plot`). Abstains, rather than fall back to the rectangle model that
    is known not to apply, when no quadrilateral can be validated. Returns None
    to let the rectangle model run when the slanted evidence is only a single line."""
    nums = _numbers_in_region(text_items, region)
    rects = [r.bbox for r in _candidate_rectangles(lines, region, rectangles)]
    area_targets = _area_targets(text_items)
    state, plot, skew_notes = skewed_plot.analyse(
        lines, region, nums=nums, rect_candidates=rects,
        printed_scales=[n.points_per_metre for n in (scale_notes or []) if n],
        plot_area_target_m2=area_targets.get("plot_area_gross") or area_targets.get("plot_area_net"),
        footprint_area_target_m2=area_targets.get("building_footprint_area"),
        caption=anchor.bounding_box,
    )
    if state != "resolved" or plot is None:
        # One stray slanted line is weak evidence and must not stop a normal
        # rectangular plot from being read; two or more heavy slanted edges
        # mean the rectangle model is known not to apply, so answer nothing.
        if not strong_evidence:
            return None
        notes.extend(skew_notes)
        return [], region, None, None, notes
    notes.extend(skew_notes)

    outer_bbox = plot.bbox()
    length = plot.edge_length_m
    label = plot.edge_label_m

    def side_value(k: str) -> float:
        return label[k] if label[k] else length[k]

    measurements: list[IndependentMeasurement] = []

    def dim(name: str, a: str, b: str, what: str):
        ka, kb = (a, b) if side_value(a) >= side_value(b) else (b, a)
        printed = label[ka] is not None
        measurements.append(_measurement(
            name, side_value(ka), "NATIVE_TEXT" if printed else "VECTOR_GEOMETRY", 0.99 if printed else 0.92,
            [f"{ka} edge {length[ka]:.2f} m" + (f", printed label {label[ka]}" if printed else "")],
            f"{what}: the longer of the two opposite sides of a plot bounded by slanted edges "
            f"({a} {side_value(a):.2f} m, {b} {side_value(b):.2f} m).",
            page_number, outer_bbox,
        ))
        if abs(side_value(a) - side_value(b)) > 0.05 * max(side_value(a), side_value(b)):
            notes.append(
                f"irregular plot: opposite sides differ ({a} {side_value(a):.2f} m vs {b} {side_value(b):.2f} m); "
                f"{name} reports the longer."
            )

    dim("plot.width", "top", "bottom", "Plot width")
    dim("plot.depth", "left", "right", "Plot depth")

    s = plot.scale_pts_per_m
    if plot.building is not None:
        bw_label, bd_label = plot.building_label_m
        bw = bw_label if bw_label else plot.building.width / s
        bd = bd_label if bd_label else plot.building.height / s
        for name, value, printed, what in (("building.width", bw, bw_label, "width"), ("building.depth", bd, bd_label, "depth")):
            measurements.append(_measurement(
                name, value, "NATIVE_TEXT" if printed else "VECTOR_GEOMETRY", 0.97 if printed else 0.90,
                [f"building rectangle {plot.building.width / s:.2f} x {plot.building.height / s:.2f} m"],
                f"Building {what} from the rectangle inside the quadrilateral plot; printed label used when it agrees.",
                page_number, plot.building,
            ))
        road_text_boxes = [
            item.bounding_box for item in text_items
            if region.intersects(item.bounding_box) and _ROAD_VALUE_RE.search(item.text or "")
        ]
        road_direction_boxes = [
            item.bounding_box for item in text_items
            if region.intersects(item.bounding_box) and not getattr(item, "is_line_group", False)
            and _ROAD_WORD_RE.search(item.text or "")
        ] or road_text_boxes
        front_side = _front_side_from_road(outer_bbox, road_direction_boxes)
        notes.append(
            f"front setback taken on the {front_side} side of the plot "
            f"({'road label position' if road_direction_boxes else 'no road label found; assumed bottom'})."
        )
        max_setback = 0.5 * min(side_value("top"), side_value("bottom"), side_value("left"), side_value("right"))
        for side, setback_name in _setback_labels_for_front(front_side).items():
            geometric = plot.setback_geometry_m.get(side)
            if geometric is None or geometric > max_setback:
                continue
            printed = sorted(plot.setback_labels_m.get(side, []), key=lambda v: abs(v - geometric))
            agreeing = next((v for v in printed if abs(v - geometric) <= max(0.15, 0.05 * geometric)), None)
            if agreeing is not None:
                measurements.append(_measurement(
                    setback_name, agreeing, "NATIVE_TEXT", 0.99, [f"printed callout {agreeing}"],
                    f"Printed setback callout that agrees with the minimum perpendicular distance from the building "
                    f"to the {side} plot edge ({geometric:.2f} m).", page_number, outer_bbox,
                ))
            else:
                measurements.append(_measurement(
                    setback_name, geometric, "DERIVED", 0.88, [f"{side} minimum perpendicular gap"],
                    f"Minimum perpendicular distance from the building to the {side} (possibly slanted) plot edge; "
                    "no printed callout agreed.", page_number, outer_bbox,
                ))
    for item in text_items:
        if not region.intersects(item.bounding_box):
            continue
        m = _ROAD_VALUE_RE.search(item.text or "")
        if m:
            width_m = _road_width_metres(m)
            if width_m is not None:
                measurements.append(_measurement(
                    "road.width", width_m, "NATIVE_TEXT", 0.99, [item.text.strip()],
                    "Explicit road-width label; not inferred from the road rectangle.", page_number, outer_bbox,
                ))
                break
    return measurements, outer_bbox, s, 0.95, notes


def extract_site_plan_measurements(
    text_items: list[RawTextItem],
    lines: list[RawLine],
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    scale_notes: list[ScaleNote] | None = None,
    rectangles: list[RawRectangle] | None = None,
) -> tuple[list[IndependentMeasurement], BoundingBox | None, float | None, float | None, list[str]]:
    """Site-plan measurements. A plot bounded by heavy SLANTED edges is not an
    axis-aligned rectangle, and the rectangle model can fit the wrong thing to
    it (on a real sheet a rectangle from another drawing matched the stated
    areas by coincidence). Sheets drawn with exactly axis-aligned lines have no
    slanted-edge evidence, so for them this is a straight call to the rectangle
    model."""
    anchor = _find_site_anchor(text_items)
    if anchor is not None:
        region = _site_region(anchor, page_width, page_height)
        evidence = skewed_plot.slanted_boundary_evidence(lines, region)
        if evidence >= 1:
            notes: list[str] = []
            skew_out = _extract_from_skewed_plot(
                text_items, lines, region, anchor, page_number, scale_notes, rectangles, notes,
                strong_evidence=evidence >= 2,
            )
            if skew_out is not None:
                return skew_out
            # No quadrilateral validated and the evidence is a single slanted
            # line: let the rectangle model run, but its assumption is known to
            # be violated, so nothing it says may be more than LOW confidence.
            measurements, bbox, scale, scale_conf, rect_notes = _extract_site_plan_rectangles(
                text_items, lines, page_number=page_number, page_width=page_width, page_height=page_height,
                scale_notes=scale_notes, rectangles=rectangles,
            )
            capped = [
                m.model_copy(update={
                    "confidence": min(m.confidence, _SKEW_UNRESOLVED_CONFIDENCE_CAP),
                    "note": ((m.note or "") + " [confidence capped: the plot boundary has a slanted edge the "
                             "rectangle model cannot represent, and no quadrilateral could be validated]").strip(),
                })
                for m in measurements
            ]
            return capped, bbox, scale, scale_conf, notes + rect_notes
    return _extract_site_plan_rectangles(
        text_items, lines, page_number=page_number, page_width=page_width, page_height=page_height,
        scale_notes=scale_notes, rectangles=rectangles,
    )


def _extract_site_plan_rectangles(
    text_items: list[RawTextItem],
    lines: list[RawLine],
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    scale_notes: list[ScaleNote] | None = None,
    rectangles: list[RawRectangle] | None = None,
) -> tuple[list[IndependentMeasurement], BoundingBox | None, float | None, float | None, list[str]]:
    notes: list[str] = []
    anchor = _find_site_anchor(text_items)
    if anchor is None:
        notes.append("no 'SITE PLAN' anchor text found; cannot locate the site-plan sub-region at all.")
        return [], None, None, None, notes
    region = _site_region(anchor, page_width, page_height)

    rects = _candidate_rectangles(lines, region, rectangles)
    substantial = [r for r in rects if min(r.width, r.height) >= 70]
    if not substantial:
        # Fallback: the tight, caption-centred window found nothing. Retry
        # once with a wider window before giving up — see the comment on
        # _REGION_HALF_WIDTH_FRACTION_WIDE for why a caption-centred window
        # can legitimately miss a real plot boundary on some sheet layouts.
        wide_region = _site_region(anchor, page_width, page_height, wide=True)
        wide_rects = _candidate_rectangles(lines, wide_region, rectangles)
        wide_substantial = [r for r in wide_rects if min(r.width, r.height) >= 70]
        if wide_substantial:
            notes.append(
                "anchor found, but the standard (tight) site-plan search window reconstructed "
                "no rectangle; recovered using a widened fallback window — the site-plan drawing "
                "is not centred on its own caption on this sheet."
            )
            region, rects, substantial = wide_region, wide_rects, wide_substantial
        else:
            notes.append(
                f"anchor found and site-plan region located, but 0 rectangles of sufficient size "
                f"(>=70pt on the shorter side) were reconstructed from {len(lines)} line(s) in that "
                "region -- geometry reconstruction itself is the failure point, before any numeric "
                "labels are even considered. Retried with a widened search window; still no "
                "rectangle was found."
            )
            return [], region, None, None, notes

    nums = _numbers_in_region(text_items, region)

    # The sheet's printed scale for THIS view, and the areas it states for
    # itself. Together these let geometry be resolved on a drawing that
    # carries no printed edge dimensions at all -- the common case that
    # previously produced null plot/building/setback values on 4 of the 5
    # bundled test plans.
    area_targets = _area_targets(text_items)
    plot_area_target = area_targets.get("plot_area_gross") or area_targets.get("plot_area_net")

    # A sheet may print several scales, one per drawing view (PLAN4 carries
    # 1:25, 1:50, 1:75 and 1:100). Which one governs the site plan cannot be
    # decided from the note's position alone -- captions crowd together in a
    # title block. So scale and plot rectangle are chosen JOINTLY: every
    # (scale note, candidate rectangle) pairing is scored, and the pairing
    # that best reproduces the sheet's stated plot area -- while staying
    # physically plausible -- wins. Proximity is retained only as a
    # tie-breaker.
    candidate_notes = _distinct_scale_notes(scale_notes or [], region.center)

    # Select the plot from geometry + explicit edge dimensions, not by raw
    # area.  This is the key fix for architectural sheets containing many
    # unrelated rectangles/dimension frames around the actual site plan.
    #
    # IMPORTANT: scan ALL `text_items` here (word-level AND is_line_group
    # merged-line items), not just `nums` -- `nums` deliberately excludes
    # line-group items (see `_numbers_in_region`), and the full phrase
    # "10m WIDE ROAD" almost never appears inside a single OCR word-level
    # token. Restricting this to `nums` meant road labels were silently
    # invisible to the exclusion logic below on any OCR-sourced page,
    # letting a road-width label get picked up as a plot/building edge
    # dimension instead of being excluded.
    road_text_boxes = [
        item.bounding_box for item in text_items
        if region.intersects(item.bounding_box) and _ROAD_VALUE_RE.search(item.text or "")
    ]
    scored_rects = []
    for printed_scale in candidate_notes:
      scale_pts_per_m = printed_scale.points_per_metre if printed_scale else None
      for candidate in substantial:
        if not _plausible_plot_rect(candidate, scale_pts_per_m):
            continue
        # Each edge may offer several stacked labels; the right one is the
        # one that makes this rectangle agree with the sheet's stated areas.
        # A candidate's own edge label also supplies the scale when the sheet
        # prints none: pt/m = the rectangle's width in points over the width
        # that label claims. That is not circular for the area test, because
        # the scale comes from ONE axis and the area then tests the OTHER.
        width_options = _edge_dimension_candidates(
            nums, candidate.bbox, "horizontal", road_text_boxes
        ) or [None]
        depth_options = _edge_dimension_candidates(
            nums, candidate.bbox, "vertical", road_text_boxes
        ) or [None]

        best_variant = None
        for wd in width_options:
            for dd in depth_options:
                if scale_pts_per_m is not None:
                    trial_scale = scale_pts_per_m
                elif wd and wd[0] > 0:
                    trial_scale = candidate.width / wd[0]
                elif dd and dd[0] > 0:
                    trial_scale = candidate.height / dd[0]
                else:
                    trial_scale = None
                agreement = 0.0
                if trial_scale is not None and plot_area_target is not None:
                    err = _area_agreement(candidate, plot_area_target, trial_scale)
                    if err is not None and err <= _AREA_MATCH_TOLERANCE:
                        agreement += 10.0 * (1.0 - err / _AREA_MATCH_TOLERANCE)
                # When both axes carry a label, they must imply the same
                # scale; a partial dimension read as a full edge will not.
                if wd and dd and wd[0] > 0 and dd[0] > 0:
                    sw, sd = candidate.width / wd[0], candidate.height / dd[0]
                    if max(sw, sd) > 0:
                        agreement += 4.0 * max(0.0, 1.0 - abs(sw - sd) / max(sw, sd) / 0.05)
                if best_variant is None or agreement > best_variant[0]:
                    best_variant = (agreement, wd, dd, trial_scale)

        _agreement, wd, dd, effective_scale = best_variant
        nested_for_candidate = []
        for r in substantial:
            if r is candidate or not _inside(r.bbox, candidate.bbox, tol=3.0):
                continue
            ratio = r.area / max(candidate.area, 1.0)
            gaps = [
                r.bbox.min_x - candidate.bbox.min_x,
                candidate.bbox.max_x - r.bbox.max_x,
                r.bbox.min_y - candidate.bbox.min_y,
                candidate.bbox.max_y - r.bbox.max_y,
            ]
            # `min(gaps) >= -_NESTED_EDGE_TOLERANCE_PTS`, not `>= 2.0`.
            # Requiring a positive gap on all four sides assumes the building
            # never touches the plot boundary. Zero-setback construction is
            # the norm on small urban Indian plots, and on the sample plan
            # (the original permit drawing) the building footprint shares its
            # left edge with the plot boundary exactly -- so the real
            # footprint was excluded here, and building.width/depth plus all
            # four setbacks came back null on a plot whose geometry had in
            # fact been reconstructed correctly.
            if 0.45 <= ratio <= 0.98 and min(gaps) >= -_NESTED_EDGE_TOLERANCE_PTS:
                nested_for_candidate.append(r)
        score = 0.0

        # An edge label only counts in favour of this rectangle when it
        # agrees with that same edge measured at this scale. Without the
        # agreement check any nearby number scored as confirmation, which is
        # how a rotated "SITE NO-09" caption printed alongside the plot was
        # credited as a 9.0 m depth dimension.
        width_label_ok = bool(wd) and (
            _label_agrees_with_geometry(wd[0], candidate.width, scale_pts_per_m)
            if scale_pts_per_m else 5.0 <= wd[0] <= 60.0
        )
        depth_label_ok = bool(dd) and (
            _label_agrees_with_geometry(dd[0], candidate.height, scale_pts_per_m)
            if scale_pts_per_m else 5.0 <= dd[0] <= 60.0
        )
        if width_label_ok:
            score += 4.0
        if depth_label_ok:
            score += 4.0

        # Agreement with the sheet's own stated plot area, measured at the
        # printed scale. This is the strongest single signal available: it
        # is a closed loop between two independent parts of the document
        # (the drawing's vector geometry, and the area statement's text),
        # neither of which was used to produce the other. It is weighted
        # above the edge-label bonuses precisely because it still works on
        # a drawing with no edge labels.
        if effective_scale is not None and plot_area_target is not None:
            error = _area_agreement(candidate, plot_area_target, effective_scale)
            if error is not None and error <= _AREA_MATCH_TOLERANCE:
                score += 10.0 * (1.0 - error / _AREA_MATCH_TOLERANCE)

        # Does this candidate contain a rectangle that reproduces the stated
        # COVERAGE area at this candidate's own scale?
        #
        # This is the test that separates a plot boundary from a dimension
        # frame drawn around it. Those two are often geometrically similar --
        # on PLAN4 the frame and the true boundary differ in aspect ratio by
        # 0.24% -- so a plot-area check derived from one axis cannot tell
        # them apart, because it reduces to an aspect-ratio comparison. The
        # building inside them is not similar: read against the frame it
        # measures 153.6 sq.m, against the true boundary 185.6 sq.m, and the
        # sheet says 185.62.
        if effective_scale is not None and nested_for_candidate:
            stated = area_targets.get("building_footprint_area")
            footprint = (
                _best_rect_for_area(nested_for_candidate, stated, effective_scale, container_bbox=candidate.bbox)
                if stated is not None else None
            )
            if footprint is not None:
                # Graded, not pass/fail. A dimension frame drawn around the
                # plot is close enough in shape that it also contains SOME
                # rectangle within tolerance of the stated coverage area --
                # on PLAN4 the frame's best is 1.31% out while the true
                # boundary's is 0.01%. Both clear a binary threshold; only
                # the margin between them says which is the real boundary.
                score += 8.0 * (1.0 - footprint[1] / _AREA_MATCH_TOLERANCE)

        if nested_for_candidate:
            score += 3.0
            inner_probe = max(nested_for_candidate, key=lambda r: r.area)
            gaps = [
                inner_probe.bbox.min_x - candidate.bbox.min_x,
                candidate.bbox.max_x - inner_probe.bbox.max_x,
                inner_probe.bbox.min_y - candidate.bbox.min_y,
                candidate.bbox.max_y - inner_probe.bbox.max_y,
            ]
            if min(gaps) >= 2.0:
                score += 1.0
            # A real building footprint is normally inset on all four sides;
            # dimension frames are often much more asymmetric.
            mean_gap = sum(gaps) / 4.0
            if mean_gap > 0:
                spread = max(gaps) / mean_gap
                if spread <= 1.7:
                    score += 1.0
        # A dimension frame can be larger than the actual plot and may
        # swallow the road-width annotation below the plot.  A true plot
        # boundary must not contain its own road label.
        if any(candidate.bbox.intersects(rb) for rb in road_text_boxes):
            score -= 4.0
        score += min(1.0, candidate.area / max(r.area for r in substantial))
        scored_rects.append(
            (score, candidate, wd if width_label_ok else None,
             dd if depth_label_ok else None, nested_for_candidate, printed_scale,
             effective_scale)
        )

    if not scored_rects:
        notes.append(
            f"{len(substantial)} candidate rectangle(s) were reconstructed near the site-plan "
            f"anchor, but none is a physically plausible plot ({_MIN_PLOT_SIDE_M}-"
            f"{_MAX_PLOT_SIDE_M} m per side) under any scale the sheet states "
            f"({[n.denominator for n in candidate_notes if n]}). Nothing is asserted."
        )
        return [], region, None, None, notes

    scored_rects.sort(key=lambda x: (x[0], x[1].area), reverse=True)
    _score, outer, width_dim, depth_dim, nested, printed_scale, winner_scale = scored_rects[0]

    # Whether ANYTHING independent of the geometry itself confirms that this
    # rectangle is the plot boundary at this scale.
    #
    # Without this gate, "the best-scoring rectangle near a SITE PLAN
    # caption" is asserted as the plot even on a sheet that contains no site
    # plan at all. PLAN4 is exactly that sheet -- floor plans, a foundation
    # detail, a staircase detail and a percolation pit, no plot boundary
    # anywhere -- and it produced a confident 3.76 x 3.61 m "plot" from a
    # construction-detail box read at a detail view's 1:100 note. A wrong
    # number presented as a measurement is worse than a missing one here,
    # because the compliance engine downstream treats MISSING as
    # INSUFFICIENT_DATA (correct) but treats a value as fact.
    scale_for_winner = winner_scale
    area_confirmed = scale_for_winner is not None and plot_area_target is not None and (
        (err := _area_agreement(outer, plot_area_target, scale_for_winner)) is not None
        and err <= _AREA_MATCH_TOLERANCE
    )
    label_confirmed = width_dim is not None or depth_dim is not None
    # Avoid accidentally choosing a nearby 1.00/0.80 label if the dimension
    # text is not clearly outside the outer boundary.
    if width_dim is None or width_dim[0] < 2.0:
        width_dim = None
    if depth_dim is None or depth_dim[0] < 2.0:
        depth_dim = None

    plot_w = width_dim[0] if width_dim else None
    plot_d = depth_dim[0] if depth_dim else None

    label_samples = []
    if plot_w and outer.width > 0:
        label_samples.append(outer.width / plot_w)
    if plot_d and outer.height > 0:
        label_samples.append(outer.height / plot_d)
    scale, scale_conf, scale_reason = _resolve_scale(label_samples, printed_scale)
    notes.append(f"scale: {scale_reason}.")

    if not (area_confirmed or label_confirmed):
        notes.append(
            f"the best-scoring rectangle near the site-plan anchor "
            f"({len(substantial)} considered, score={_score:.1f}) is confirmed by NEITHER an "
            "edge dimension label that agrees with its geometry NOR the sheet's stated plot "
            f"area (stated area available: {plot_area_target is not None}). Plot dimensions are "
            "left unresolved rather than asserted from an unconfirmed rectangle -- a sheet with "
            "no site plan on it will always contain some best-scoring rectangle."
        )

    if plot_w is None and plot_d is None and scale is None:
        notes.append(
            f"picked a candidate rectangle ({len(substantial)} candidate(s) considered near the "
            f"site-plan anchor, score={_score:.1f}), but found NO numeric label within range on "
            "either its horizontal or vertical edges AND no printed scale note -- plot.width and "
            "plot.depth cannot be resolved, which also blocks building.width/depth and all 4 "
            "setbacks even if a nested building rectangle exists."
        )

    measurements: list[IndependentMeasurement] = []
    if plot_w is not None:
        measurements.append(_measurement("plot.width", plot_w, "NATIVE_TEXT", 0.99, [width_dim[1].text.strip()], "Explicit site-plan width label anchored to outer plot rectangle.", page_number, outer.bbox))
    elif scale and (area_confirmed or label_confirmed):
        # No printed edge label for THIS side, but the rectangle is confirmed
        # as the plot by the sheet's own area statement (or by the other
        # side's label), and the scale is established -- so measure it.
        # Lower confidence than a printed label, and marked VECTOR_GEOMETRY
        # so downstream reconciliation can tell the two apart.
        plot_w = outer.width / scale
        measurements.append(_measurement("plot.width", plot_w, "VECTOR_GEOMETRY", 0.92, [f"outer rectangle width {outer.width:.2f} pt", scale_reason], "Plot width measured from the site-plan boundary rectangle at the drawing's resolved scale; no printed edge label was present.", page_number, outer.bbox))
    if plot_d is not None:
        measurements.append(_measurement("plot.depth", plot_d, "NATIVE_TEXT", 0.99, [depth_dim[1].text.strip()], "Explicit site-plan depth label anchored to outer plot rectangle.", page_number, outer.bbox))
    elif scale and (area_confirmed or label_confirmed):
        plot_d = outer.height / scale
        measurements.append(_measurement("plot.depth", plot_d, "VECTOR_GEOMETRY", 0.92, [f"outer rectangle height {outer.height:.2f} pt", scale_reason], "Plot depth measured from the site-plan boundary rectangle at the drawing's resolved scale; no printed edge label was present.", page_number, outer.bbox))

    # The building footprint: prefer the nested rectangle that reproduces the
    # sheet's stated coverage area, falling back to the largest nested one.
    inner = None
    inner_note = ""
    footprint_match = _best_rect_for_area(
        nested, area_targets.get("building_footprint_area"), scale, container_bbox=outer.bbox
    )
    stated_footprint = area_targets.get("building_footprint_area")
    if footprint_match is not None:
        inner, footprint_error = footprint_match
        inner_note = (
            f"footprint rectangle reproduces the sheet's stated coverage area "
            f"({stated_footprint} sq.m) to within {footprint_error:.2%}"
        )
    elif nested and stated_footprint is None:
        # No stated coverage area anywhere on the sheet, so there is nothing
        # to tell the building footprint apart from any other nested
        # rectangle -- a dimension frame, a hatched setback band, a parking
        # bay. Picking the largest was a guess, and on PLAN5 it picked a
        # rectangle spanning the full plot width and reported a 17.59 m wide
        # building. Plot dimensions still stand on their own evidence; the
        # building simply is not established.
        notes.append(
            f"{len(nested)} rectangle(s) are nested inside the resolved plot boundary, but the "
            "sheet states no coverage area to identify which one is the building footprint. "
            "building.width/depth and the setbacks are left unresolved rather than guessed from "
            "the largest nested rectangle."
        )
    elif nested:
        # The sheet DOES state a coverage area, and no nested rectangle
        # reproduces it. Falling back to "largest nested rectangle" here
        # asserts a footprint the document actively contradicts: on PLAN5
        # that picked a rectangle spanning the full plot width, giving a
        # building 17.59 m wide against a stated 93.46 sq.m footprint, and
        # dragged all four setbacks wrong with it. Confirming the PLOT does
        # not confirm the BUILDING -- they are separate rectangles and need
        # separate evidence.
        notes.append(
            f"the plot boundary was resolved, but no nested rectangle reproduces the sheet's "
            f"stated coverage area ({stated_footprint} sq.m) within "
            f"{_AREA_MATCH_TOLERANCE:.0%}. building.width/depth and the setbacks are left "
            "unresolved rather than derived from a footprint the document contradicts."
        )

    # Everything below is measured relative to the plot rectangle, so it
    # inherits that rectangle's confirmation status. A nested rectangle that
    # independently reproduces the sheet's stated coverage area is itself a
    # confirmation, so it counts too.
    nest_confirmed = area_confirmed or label_confirmed or footprint_match is not None
    if inner is not None and scale and nest_confirmed:
        b_w = inner.width / scale
        b_d = inner.height / scale
        measurements.append(_measurement("building.width", b_w, "VECTOR_GEOMETRY", 0.97, [f"inner rectangle width {inner.width:.2f} pt", f"scale {scale:.3f} pt/m", inner_note], "Building width derived from the nested building footprint geometry and site-plan scale; no Vision input.", page_number, inner.bbox))
        measurements.append(_measurement("building.depth", b_d, "VECTOR_GEOMETRY", 0.97, [f"inner rectangle height {inner.height:.2f} pt", f"scale {scale:.3f} pt/m", inner_note], "Building depth derived from the nested building footprint geometry and site-plan scale; no Vision input.", page_number, inner.bbox))

        # A setback is bounded by the plot side it crosses; anything larger
        # is a plot/building dimension that wandered into the gap.
        plot_short_side_m = min(
            plot_w if plot_w else float("inf"), plot_d if plot_d else float("inf")
        )
        max_setback_m = (
            plot_short_side_m * _MAX_SETBACK_AS_PLOT_FRACTION
            if plot_short_side_m != float("inf") else None
        )

        # Prefer precise, single-item road labels for the direction decision.
        # A merged OCR line group can span half the sheet ("7.14X6.50 BELOW
        # BELOW 10m WIDE ROAD WINDOW 1.20 X 1.20" on PLAN5), and its centre
        # then points nowhere near the actual road.
        road_direction_boxes = [
            item.bounding_box for item in text_items
            if region.intersects(item.bounding_box)
            and not getattr(item, "is_line_group", False)
            and _ROAD_WORD_RE.search(item.text or "")
        ] or road_text_boxes
        # Only consult the shared evidence hierarchy's extra tiers (FRONT
        # label, STREET/ACCESS/MAIN-ENTRY/GATE text) when THIS module's own
        # road-position detection found no road at all. `_front_side_from_
        # road`'s dx/dy check is already validated as correct on a real,
        # rotated sheet (a road on the LEFT, not below -- see its own
        # docstring); a bare "front"/"gate"/etc. text match elsewhere on
        # the sheet (a "FRONT SETBACK" dimension label, say, sitting
        # nearest a DIFFERENT edge than the true frontage on a rotated
        # sheet) must never be allowed to override a working road-based
        # determination -- confirmed as a real regression risk directly:
        # trusting it unconditionally flipped PLAN5's setbacks a quarter
        # turn again, the exact bug class this whole mechanism exists to
        # prevent.
        shared_evidence = (
            _front_side_from_shared_evidence(outer.bbox, road_direction_boxes, text_items, region)
            if not road_direction_boxes else None
        )
        if shared_evidence is not None:
            front_side, evidence_level = shared_evidence
            notes.append(
                f"front setback taken on the {front_side} side of the plot "
                f"(explicit {evidence_level} found on the sheet)."
            )
        else:
            front_side = _front_side_from_road(outer.bbox, road_direction_boxes)
            notes.append(
                f"front setback taken on the {front_side} side of the plot "
                f"({'road label position' if road_direction_boxes else 'no road label found; assumed bottom'})."
            )
        side_labels = _setback_labels_for_front(front_side)
        for side, label in side_labels.items():
            picked = _pick_gap_dimension(nums, outer.bbox, inner.bbox, side, max_setback_m)
            if picked:
                val, item = picked
                measurements.append(_measurement(label, val, "NATIVE_TEXT", 0.99, [item.text.strip()], f"Explicit setback label spatially located in the {side} plot/building gap.", page_number, outer.bbox))
            else:
                # Geometry-derived fallback only if the gap is substantial and
                # both edges are clearly parallel. This is still independent CV
                # geometry, but is marked DERIVED and lower confidence.
                if side == "top": gap = (inner.bbox.min_y - outer.bbox.min_y) / scale
                elif side == "bottom": gap = (outer.bbox.max_y - inner.bbox.max_y) / scale
                elif side == "left": gap = (inner.bbox.min_x - outer.bbox.min_x) / scale
                else: gap = (outer.bbox.max_x - inner.bbox.max_x) / scale
                if 0 <= gap <= (max_setback_m if max_setback_m is not None else 5.0):
                    measurements.append(_measurement(label, gap, "DERIVED", 0.88, [f"{side} geometric gap"], f"No explicit label was associated; setback derived from nested rectangles. Marked lower confidence.", page_number, outer.bbox))
    elif inner is not None and scale and not nest_confirmed:
        notes.append(
            "a nested rectangle was found inside the winning outer candidate, but the outer "
            "candidate is not confirmed as the plot boundary (see above) and the nested one "
            "does not reproduce any stated coverage area -- building.width/depth and the "
            "setbacks are therefore left unresolved rather than derived from an unverified "
            "pair of rectangles."
        )
    elif inner is None and scale:
        notes.append(
            "plot.width/depth resolved and scale computed, but no nested (building footprint) "
            "rectangle was found inside the winning outer candidate -- building.width/depth and "
            "all 4 setbacks cannot be derived without one, even though the plot dimensions "
            "themselves succeeded."
        )

    road = None
    # Scan ALL text_items (word + line-group), not just `nums` -- same
    # reasoning as `road_text_boxes` above: the full "<value> WIDE ROAD"
    # phrase needs a merged line-group item to match at all on an
    # OCR-sourced page, since `nums` (word-level only) never contains it.
    for item in text_items:
        if not region.intersects(item.bounding_box):
            continue
        m = _ROAD_VALUE_RE.search(item.text or "")
        if m:
            width_m = _road_width_metres(m)
            if width_m is None:
                continue
            road = (width_m, item)
            break
    if road is not None:
        measurements.append(_measurement("road.width", road[0], "NATIVE_TEXT", 0.99, [road[1].text.strip()], "Explicit road-width label; not inferred from the road rectangle.", page_number, outer.bbox))

    return measurements, outer.bbox, scale, scale_conf, notes


def extract_independent_cv(document_path, document_id: str) -> IndependentCVResult:
    from backend.cv_extraction import ocr_fallback, pdf_native

    doc = pdf_native.open_document(document_path)
    all_measurements: list[IndependentMeasurement] = []
    warnings: list[str] = []
    pages: list[int] = []
    site_page = None
    site_bbox = None
    scale = None
    scale_conf = None
    try:
        for page_number in range(doc.page_count):
            page = doc.load_page(page_number)
            meta = pdf_native.extract_page_metadata(page, page_number)
            text_items = pdf_native.extract_text_items(page, page_number)
            lines, native_rects, _polys = pdf_native.extract_vector_geometry(page, page_number)
            image = None
            if not pdf_native.has_sufficient_native_text(text_items):
                try:
                    image = ocr_fallback.rasterize_page(page, dpi=200.0)
                    # Orientation-aware: a landscape sheet stored as an
                    # unrotated portrait page renders all its text sideways,
                    # which plain OCR reads badly without ever failing.
                    text_items = ocr_fallback.ocr_page_any_orientation(
                        image, page_number, dpi=200.0
                    )
                except Exception as exc:
                    warnings.append(f"page {page_number+1}: OCR fallback failed: {exc}")
            page_scale_notes = detect_scale_notes(text_items, page_number)
            measurements, bbox, page_scale, page_scale_conf, notes = extract_site_plan_measurements(
                text_items, lines, page_number=page_number, page_width=meta.width_pts,
                page_height=meta.height_pts, scale_notes=page_scale_notes,
                rectangles=native_rects,
            )
            for n in notes:
                warnings.append(f"page {page_number+1}: {n}")
            # Scanned/mixed plans have no PDF vector lines. Re-run the SAME
            # site-plan resolver over raster line evidence instead of falling
            # back to the legacy global candidate pool. This keeps CV independent
            # and preserves the site-plan spatial semantics.
            if not measurements and image is not None:
                try:
                    from backend.cv_extraction import opencv_geometry
                    cv_evidence = opencv_geometry.geometry_evidence_for_page(image, page_number, dpi=200.0)
                    measurements, bbox, page_scale, page_scale_conf, raster_notes = extract_site_plan_measurements(
                        text_items, cv_evidence["lines"], page_number=page_number,
                        page_width=meta.width_pts, page_height=meta.height_pts,
                        scale_notes=detect_scale_notes(text_items, page_number),
                    )
                    for n in raster_notes:
                        warnings.append(f"page {page_number+1} (raster fallback): {n}")
                    if measurements:
                        warnings.append(f"page {page_number+1}: independent CV used raster line/contour evidence (vector geometry unavailable).")
                except Exception as exc:
                    warnings.append(f"page {page_number+1}: raster CV site-plan fallback failed: {exc}")
            measurements.extend(_area_measurements(text_items, page_number))
            if measurements:
                pages.append(page_number + 1)
                all_measurements.extend(measurements)
                # Only a page that actually resolved site-plan GEOMETRY may
                # claim to be the site-plan page. Previously every page with
                # any measurement overwrote these, so on a multi-page set a
                # later page carrying nothing but an area-statement table
                # (bbox=None, scale=None) silently erased the real site
                # page's resolved bounding box and scale.
                if bbox is not None and page_scale is not None:
                    site_page = page_number + 1
                    site_bbox = bbox
                    scale = page_scale
                    scale_conf = page_scale_conf
                elif site_page is None:
                    site_page = page_number + 1
    finally:
        doc.close()
    if not all_measurements:
        warnings.append("No site-plan geometry with sufficient independent CV/native-text evidence was resolved.")
    return IndependentCVResult(
        document_id=document_id,
        pages_analyzed=pages,
        measurements=all_measurements,
        warnings=warnings,
        site_plan_page=site_page,
        site_plan_bbox_pts=(None if site_bbox is None else [site_bbox.min_x, site_bbox.min_y, site_bbox.max_x, site_bbox.max_y]),
        scale_points_per_metre=scale,
        scale_confidence=scale_conf,
    )
