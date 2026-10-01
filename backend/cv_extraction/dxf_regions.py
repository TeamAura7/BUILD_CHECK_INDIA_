"""
Generic architectural-sheet region detection for DXF geometry.

Real architectural DXFs often place multiple distinct drawings on ONE sheet
(a site plan, several floor plans, elevations, sections, a title block)
with no distinguishing layer names at all -- confirmed directly on a real
production file (19,622 POLYLINE entities, all on layer '0', zero TEXT/
MTEXT/DIMENSION entities anywhere: the whole sheet, including every wall,
dimension line, and printed digit, was vectorized into bare polyline
strokes with no semantic tagging whatsoever).

Feeding all of that geometry into a single global plot/building
role-inference graph at once (the previous behavior) has two failure modes:

  (a) the sheet's own overall bounding frame -- or some other rectangle
      that happens to enclose most of the page -- can be selected as "the
      plot" purely because it is the single largest closed polygon on the
      page, with no check for whether it actually belongs to one specific
      drawing rather than the sheet as a whole; and

  (b) fragmented-line-work reconstruction (`dxf_reconstruction.py`) sees
      one enormous connected component spanning the WHOLE sheet (every
      drawing's line-work touching or nearly touching every other's, via
      shared dimension/leader/hatch strokes) and bails out on it entirely
      once it exceeds `_MAX_COMPONENT_EDGES`, so the real building outline
      never gets a chance to be considered as a reconstruction candidate
      at all.

Naive "flood-fill every occupied grid cell" clustering does NOT separate
a sheet like this: measured directly on the real file above, 8-connected
flood-fill over an occupancy grid collapses into one giant blob (2674 of
2729 occupied cells) because long, thin connecting strokes -- a sheet
border, a grid line, a long dimension string -- snake across the whole
page and bridge otherwise-distinct drawings together. This module instead
uses a two-stage density-erosion approach, verified empirically on that
same real file to separate it into ~10 distinct dense regions matching a
typical multi-view sheet layout:

  1. Rasterize every entity's own points (with segment interpolation, so a
     long two-point segment doesn't skip over a real gap) onto a grid
     sized relative to the sheet's own extent.
  2. For every occupied cell, count how many OTHER occupied cells lie in a
     small neighborhood around it ("local density"). A real drawing's
     interior (walls, hatching, nearby dimension lines) is locally dense;
     a thin connecting stroke passing through empty space is not.
  3. Keep only cells above an ADAPTIVE density percentile (computed from
     this file's own density distribution, never a fixed absolute count)
     as "core" cells, and flood-fill THOSE into seed blobs -- this erodes
     away the thin bridging strokes while leaving genuinely dense regions
     intact and separated.
  4. Multi-source BFS assigns every originally-occupied cell (including
     the eroded-away thin/bridging ones) to its nearest seed blob, so the
     final regions still include their own connecting geometry (e.g. a
     dimension line reaching just outside a drawing's dense core).

When a DXF genuinely contains only one drawing (the common case exercised
by this project's existing synthetic single-view fixtures), this collapses
to a single region and every function here becomes a no-op relative to the
previous global behavior.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

from backend.schemas.geometry import BoundingBox

Point2 = tuple[float, float]
Cell = tuple[int, int]

# Grid resolution relative to the sheet's own extent (never an absolute
# coordinate value), targeting roughly this many cells along the longer of
# the sheet's two dimensions.
_TARGET_CELLS_ALONG_LONG_AXIS = 90

# Neighborhood radius (in cells) used to compute each occupied cell's local
# density -- wide enough to distinguish a genuinely dense drawing interior
# from a single thin connecting stroke, without being so wide that two
# separate but moderately close drawings merge into one density plateau.
_DENSITY_RADIUS_CELLS = 2

# Percentile of this file's OWN local-density distribution used as the
# core/erosion threshold. Not an absolute density value -- different DXFs
# vary hugely in how densely they're drawn, so the threshold is always
# relative to what this specific file's content actually looks like.
_CORE_DENSITY_PERCENTILE = 0.65

# A seed blob smaller than this fraction of the largest seed blob's cell
# count is dropped as noise rather than kept as a spurious tiny region.
_MIN_SEED_RELATIVE_SIZE = 0.02

_MIN_CELL_SIZE_FRACTION = 1e-6  # guards against a degenerate zero-extent sheet


@dataclass
class DxfRegion:
    """One spatially distinct cluster of DXF geometry on a sheet."""

    id: int
    bbox: BoundingBox
    polygon_indices: list[int] = field(default_factory=list)
    segment_indices: list[int] = field(default_factory=list)
    text_indices: list[int] = field(default_factory=list)
    entity_count: int = 0


def _bbox_union(boxes: Sequence[BoundingBox]) -> Optional[BoundingBox]:
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return BoundingBox(
        min_x=min(b.min_x for b in boxes),
        min_y=min(b.min_y for b in boxes),
        max_x=max(b.max_x for b in boxes),
        max_y=max(b.max_y for b in boxes),
    )


def _bbox_area(b: BoundingBox) -> float:
    return max(0.0, b.max_x - b.min_x) * max(0.0, b.max_y - b.min_y)


def _bbox_overlap_fraction(inner: BoundingBox, outer: BoundingBox) -> float:
    """Fraction of `inner`'s own area that lies within `outer`."""
    ix0 = max(inner.min_x, outer.min_x)
    iy0 = max(inner.min_y, outer.min_y)
    ix1 = min(inner.max_x, outer.max_x)
    iy1 = min(inner.max_y, outer.max_y)
    inter_area = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    inner_area = _bbox_area(inner)
    if inner_area <= 0:
        return 0.0
    return inter_area / inner_area


def _interpolate(p1: Point2, p2: Point2, max_step: float) -> list[Point2]:
    d = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if d <= max_step:
        return [p2]
    n = max(1, int(d / max_step))
    return [
        (p1[0] + (p2[0] - p1[0]) * k / n, p1[1] + (p2[1] - p1[1]) * k / n)
        for k in range(1, n + 1)
    ]


def _rasterize(points: Sequence[Point2], cell: float, min_x: float, min_y: float) -> set[Cell]:
    if not points:
        return set()

    def _key(p: Point2) -> Cell:
        return (int((p[0] - min_x) // cell), int((p[1] - min_y) // cell))

    cells = {_key(points[0])}
    prev = points[0]
    max_step = cell * 0.5
    for p in points[1:]:
        for ip in _interpolate(prev, p, max_step):
            cells.add(_key(ip))
        prev = p
    return cells


def cluster_entity_points(
    entity_points: Sequence[Sequence[Point2]],
    target_cells_along_long_axis: int = _TARGET_CELLS_ALONG_LONG_AXIS,
) -> list[list[int]]:
    """Partition entities into spatially distinct clusters.

    `entity_points[i]` is entity `i`'s own point sequence (a polygon's
    vertices, a segment's two endpoints, or a single-point text position).
    Returns a list of clusters (each a list of indices into
    `entity_points`), sorted by descending entity count. Entities that
    fall in a spurious/noise cluster relative to the largest one are
    dropped from the result entirely.
    """
    indexed = [(i, pts) for i, pts in enumerate(entity_points) if pts]
    if not indexed:
        return []

    all_points = [p for _, pts in indexed for p in pts]
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    long_axis = max(max_x - min_x, max_y - min_y)
    if long_axis <= _MIN_CELL_SIZE_FRACTION:
        return [[i for i, _ in indexed]]
    cell = max(long_axis / max(target_cells_along_long_axis, 1), long_axis * 1e-4)

    entity_cells: dict[int, set[Cell]] = {}
    cell_to_entities: dict[Cell, list[int]] = {}
    for i, pts in indexed:
        cells = _rasterize(pts, cell, min_x, min_y)
        entity_cells[i] = cells
        for c in cells:
            cell_to_entities.setdefault(c, []).append(i)

    occupied = set(cell_to_entities.keys())
    if not occupied:
        return []

    # Stage 1: local density per occupied cell, measured as the number of
    # DISTINCT ENTITIES touching a small neighborhood -- not the number of
    # occupied cells in that neighborhood. A single long line (or any one
    # entity interpolated into many sample points) fills a whole trail of
    # consecutive cells with itself alone; counting occupied CELLS there
    # would call that trail "dense" purely from one entity's own
    # self-interpolation, which is exactly what let a thin connecting
    # stroke masquerade as real content and bridge two unrelated drawings
    # together in an earlier version of this algorithm. Counting distinct
    # entities instead only calls a neighborhood dense when MULTIPLE
    # different things (several walls, several dashes, a dimension line
    # plus its text) actually cluster there.
    r = _DENSITY_RADIUS_CELLS
    density: dict[Cell, int] = {}
    for (cx, cy) in occupied:
        touching: set[int] = set()
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                touching.update(cell_to_entities.get((cx + dx, cy + dy), ()))
        density[(cx, cy)] = len(touching)

    sorted_density = sorted(density.values())
    idx = min(len(sorted_density) - 1, int(len(sorted_density) * _CORE_DENSITY_PERCENTILE))
    threshold = sorted_density[idx]
    core = {c for c, d in density.items() if d >= threshold}
    if not core:
        core = occupied  # degenerate: everything equally sparse, fall back to flat flood-fill

    # A floor well below the core threshold, used only to bound stage 3's
    # BFS below: a cell touched by just one or two distinct entities is
    # exactly the "thin connecting stroke passing through" case that must
    # not bridge two seed blobs back together, even though it IS a real,
    # occupied cell that should still end up assigned to whichever seed
    # it's actually closest to.
    floor_idx = min(len(sorted_density) - 1, int(len(sorted_density) * 0.15))
    floor_threshold = max(1, sorted_density[floor_idx])
    traversable = {c for c, d in density.items() if d >= floor_threshold}

    # Stage 2: flood-fill CORE cells into seed blobs (8-connectivity).
    seed_of: dict[Cell, int] = {}
    seeds: list[list[Cell]] = []
    for start in core:
        if start in seed_of:
            continue
        sid = len(seeds)
        stack = [start]
        members: list[Cell] = []
        while stack:
            c = stack.pop()
            if c in seed_of:
                continue
            seed_of[c] = sid
            members.append(c)
            cx, cy = c
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nb = (cx + dx, cy + dy)
                    if nb in core and nb not in seed_of:
                        stack.append(nb)
        seeds.append(members)

    if not seeds:
        return []
    largest = max(len(s) for s in seeds)
    kept_seed_ids = {sid for sid, s in enumerate(seeds) if len(s) >= max(1, largest * _MIN_SEED_RELATIVE_SIZE)}

    # Stage 3: multi-source BFS assigns every cell ABOVE the floor density
    # (including eroded-away-from-core-but-still-multi-entity cells) to its
    # nearest surviving seed blob. Deliberately traverses `traversable`, NOT
    # `occupied` -- a cell touched by only one or two distinct entities
    # (below the floor) is exactly a thin connecting stroke passing through
    # otherwise-empty space, and must not let the BFS bridge across it back
    # into one giant region again.
    cell_region: dict[Cell, int] = {}
    frontier: deque[Cell] = deque()
    for c, sid in seed_of.items():
        if sid in kept_seed_ids:
            cell_region[c] = sid
            frontier.append(c)
    while frontier:
        c = frontier.popleft()
        rid = cell_region[c]
        cx, cy = c
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nb = (cx + dx, cy + dy)
                if nb in traversable and nb not in cell_region:
                    cell_region[nb] = rid
                    frontier.append(nb)

    # Any occupied cell unreachable from every seed (isolated far-away
    # patch) is left unassigned -- entities living only in such cells are
    # dropped as noise, same as a too-small seed blob would be.
    entity_to_region: dict[int, int] = {}
    for i, cells in entity_cells.items():
        for c in cells:
            if c in cell_region:
                entity_to_region[i] = cell_region[c]
                break

    clusters: dict[int, list[int]] = {}
    for i, rid in entity_to_region.items():
        clusters.setdefault(rid, []).append(i)
    return sorted(clusters.values(), key=len, reverse=True)


def build_regions(
    polygon_points: Sequence[Sequence[Point2]],
    segment_points: Sequence[Sequence[Point2]],
    text_points: Sequence[Sequence[Point2]],
) -> list[DxfRegion]:
    """Cluster every piece of a DXF's geometry (closed polygons, open
    line-work segments, text/attribute positions) into spatially distinct
    regions, using ALL of it together so a region's extent reflects its
    full content, not just its closed shapes."""
    combined: list[Sequence[Point2]] = list(polygon_points) + list(segment_points) + list(text_points)
    n_poly = len(polygon_points)
    n_seg = len(segment_points)

    clusters = cluster_entity_points(combined)
    regions: list[DxfRegion] = []
    for cid, members in enumerate(clusters):
        poly_idx = [i for i in members if i < n_poly]
        seg_idx = [i - n_poly for i in members if n_poly <= i < n_poly + n_seg]
        text_idx = [i - n_poly - n_seg for i in members if i >= n_poly + n_seg]
        member_points = [p for i in members for p in combined[i]]
        if not member_points:
            continue
        xs = [p[0] for p in member_points]
        ys = [p[1] for p in member_points]
        bbox = BoundingBox(min_x=min(xs), min_y=min(ys), max_x=max(xs), max_y=max(ys))
        regions.append(DxfRegion(
            id=cid, bbox=bbox, polygon_indices=poly_idx,
            segment_indices=seg_idx, text_indices=text_idx,
            entity_count=len(members),
        ))
    return regions


# A candidate polygon is only eligible to be flagged as a cross-cutting
# sheet-border/frame if its own bounding box covers at least this fraction
# of the ENTIRE sheet's extent -- a real single-drawing plot boundary that
# happens to be the largest thing in its own (single) region should never
# be penalized just for being large relative to its own region; this
# threshold is about the whole sheet, not any one region.
_FRAME_MIN_SHEET_COVERAGE = 0.5


def detect_frame_polygon_indices(
    polygon_bboxes: Sequence[Optional[BoundingBox]],
    regions: list[DxfRegion],
) -> set[int]:
    """Identify which closed polygons are sheet-border/frame-like rather
    than belonging to one specific drawing.

    Evidence used (relational, not absolute-coordinate-based):
      1. The polygon's own bounding box covers a large fraction of the
         WHOLE sheet's extent (`_FRAME_MIN_SHEET_COVERAGE`) -- necessary
         but not sufficient, since a real plot boundary on a sheet with
         only one drawing on it can legitimately be almost the whole
         sheet too.
      2. The polygon's bounding box substantially overlaps MORE THAN ONE
         distinct region -- the discriminating signal: a real plot
         boundary belongs to exactly one drawing/region; something that
         spans multiple independently-clustered regions is, by
         construction, not scoped to any single drawing and is instead
         cutting across the whole sheet layout the way a page border or a
         title-block frame does.

    If there is only one region in total, nothing can "span multiple
    regions" by definition, so this never flags anything -- exactly
    preserving prior behavior on a normal single-drawing DXF.
    """
    if len(regions) < 2:
        return set()
    overall = _bbox_union([r.bbox for r in regions])
    if overall is None:
        return set()
    overall_area = _bbox_area(overall)
    if overall_area <= 0:
        return set()

    frame_indices: set[int] = set()
    for i, b in enumerate(polygon_bboxes):
        if b is None:
            continue
        coverage = _bbox_area(b) / overall_area
        if coverage < _FRAME_MIN_SHEET_COVERAGE:
            continue
        spanned = sum(1 for r in regions if _bbox_overlap_fraction(r.bbox, b) > 0.5)
        if spanned >= 2:
            frame_indices.add(i)
    return frame_indices


__all__ = [
    "DxfRegion",
    "cluster_entity_points",
    "build_regions",
    "detect_frame_polygon_indices",
]
