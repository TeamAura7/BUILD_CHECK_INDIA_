"""
Direct unit tests for `backend.cv_extraction.dxf_wall_union` -- the area-
based (buffer/union) building reconstruction added for Critical
Requirement 3, covering exactly the failure modes
`dxf_reconstruction.reconstruct_building_polygon`'s pure cycle-detection
approach cannot: door/window gaps, T-junctions, double-line (thick) walls,
and overlapping/duplicate line-work.

Pure-geometry tests (no DXF, no extractor) so the algorithm itself is
pinned independently of DXF entity collection -- see `test_dxf_extractor.py`
for end-to-end DXF-level coverage.
"""

from __future__ import annotations

import pytest

from backend.cv_extraction.dxf_wall_union import (
    _bbox_diagonal,
    estimate_wall_thickness,
    find_gap_bridge_segments,
    reconstruct_building_via_wall_union,
)


def _rect_walls(x0, y0, x1, y1, layer="WALL"):
    return [
        (layer, (x0, y0), (x1, y0)),
        (layer, (x1, y0), (x1, y1)),
        (layer, (x1, y1), (x0, y1)),
        (layer, (x0, y1), (x0, y0)),
    ]


def test_closed_rectangle_reconstructs_correctly():
    """Sanity baseline: a plain closed rectangle (no gaps, no T-junctions)
    must still reconstruct to approximately its own area -- the area-
    based method must not be worse than the trivial case."""
    segs = _rect_walls(0, 0, 10, 8)
    result = reconstruct_building_via_wall_union(segs, None)
    assert result is not None
    assert result.polygon.area == pytest.approx(80.0, rel=0.02)


def test_door_gap_in_one_wall_is_bridged():
    """A door-sized gap in one wall breaks `reconstruct_building_polygon`'s
    cycle detection outright (no graph edge spans the gap, so no cycle
    passes through it) -- the wall-union method must still recover the
    full footprint by bridging the gap between the two nearest dangling
    wall endpoints."""
    segs = [
        ("WALL", (0, 0), (4, 0)),
        ("WALL", (6, 0), (10, 0)),  # 2-unit door gap in the bottom wall
        ("WALL", (10, 0), (10, 8)),
        ("WALL", (10, 8), (0, 8)),
        ("WALL", (0, 8), (0, 0)),
    ]
    result = reconstruct_building_via_wall_union(segs, None)
    assert result is not None
    assert result.polygon.area == pytest.approx(80.0, rel=0.05)
    assert "opening-bridge" in result.reason


def test_two_gaps_in_different_walls_are_both_bridged():
    segs = [
        ("WALL", (0, 0), (4, 0)), ("WALL", (6, 0), (10, 0)),   # gap in bottom wall
        ("WALL", (10, 0), (10, 3)), ("WALL", (10, 5), (10, 8)),  # gap in right wall
        ("WALL", (10, 8), (0, 8)),
        ("WALL", (0, 8), (0, 0)),
    ]
    result = reconstruct_building_via_wall_union(segs, None)
    assert result is not None
    assert result.polygon.area == pytest.approx(80.0, rel=0.05)


def test_gap_too_wide_to_be_a_plausible_opening_is_not_bridged():
    """A "gap" comparable to the whole structure's own scale is more
    likely two separate, unrelated pieces of line-work than a real
    door/window -- must not be bridged, and no plausible building should
    be fabricated from mostly-disconnected fragments."""
    segs = [
        ("WALL", (0, 0), (1, 0)), ("WALL", (9, 0), (10, 0)),  # 8-unit "gap" on a 10-unit side
        ("WALL", (10, 0), (10, 8)),
        ("WALL", (10, 8), (0, 8)),
        ("WALL", (0, 8), (0, 0)),
    ]
    scale = _bbox_diagonal([(s, e) for _, s, e in segs])
    bridges = find_gap_bridge_segments([(s, e) for _, s, e in segs], 0.05, scale)
    assert bridges == []


def test_t_junction_interior_partition_does_not_distort_outer_footprint():
    """An interior partition wall meeting an exterior wall (a T-junction,
    not a plain 4-way or 2-way node) must not change the recovered OUTER
    footprint -- the union's exterior boundary is the whole building's
    outer envelope regardless of what's nested inside it."""
    segs = _rect_walls(0, 0, 10, 8) + [("WALL", (5, 2), (5, 8))]
    result = reconstruct_building_via_wall_union(segs, None)
    assert result is not None
    assert result.polygon.area == pytest.approx(80.0, rel=0.02)


def test_double_line_wall_thickness_is_estimated_and_not_double_counted():
    """Walls drawn as two parallel lines (inner + outer face) a constant
    distance apart -- the distance itself IS the wall's thickness,
    inferred from this document's own geometry. The reconstructed
    footprint must reflect that real thickness once, not the drawn face
    distance PLUS an extra buffer on top of it (which would inflate the
    footprint beyond the true outer face)."""
    t = 0.2
    w, h = 10.0, 8.0
    segs = _rect_walls(0, 0, w, h) + _rect_walls(t, t, w - t, h - t)
    structural = [(s, e) for _, s, e in segs]
    scale = _bbox_diagonal(structural)
    assert estimate_wall_thickness(structural, scale) == pytest.approx(t, rel=0.05)

    result = reconstruct_building_via_wall_union(segs, None)
    assert result is not None
    # True footprint (outer face + half the real thickness outward on
    # each side) is (w+t) x (h+t) -- NOT (w+2t) x (h+2t), which is what
    # double-counting the thickness would produce.
    assert result.polygon.area == pytest.approx((w + t) * (h + t), rel=0.05)


def test_duplicate_overlapping_trace_does_not_distort_the_result():
    """A wall re-traced twice with a tiny offset (common in vectorized/
    imprecise-tracing DXFs) must not inflate or otherwise distort the
    footprint -- union of overlapping areas is idempotent."""
    segs = _rect_walls(0, 0, 10, 8) + [
        ("WALL", (0.01, 0.005), (10.01, 0.005)),  # near-duplicate bottom wall trace
    ]
    result = reconstruct_building_via_wall_union(segs, None)
    assert result is not None
    assert result.polygon.area == pytest.approx(80.0, rel=0.03)


def test_annotation_layers_are_excluded():
    """A dimension line sitting right next to the building must not be
    treated as a wall -- same layer-filtering rule as
    `dxf_reconstruction.reconstruct_building_polygon`."""
    segs = _rect_walls(0, 0, 10, 8) + [("DIMENSIONS", (-2, -1), (12, -1))]
    result = reconstruct_building_via_wall_union(segs, None)
    assert result is not None
    assert result.polygon.area == pytest.approx(80.0, rel=0.02)


def test_too_few_segments_returns_none():
    result = reconstruct_building_via_wall_union([("WALL", (0, 0), (1, 0))], None)
    assert result is None


def test_implausible_relative_to_plot_is_rejected():
    from backend.schemas.geometry import Point, Polygon

    plot = Polygon(points=[Point(x=0, y=0), Point(x=100, y=0), Point(x=100, y=100), Point(x=0, y=100)])
    segs = _rect_walls(0, 0, 1, 1)  # 1x1 "building" inside a 100x100 plot -- 0.01% coverage
    result = reconstruct_building_via_wall_union(segs, plot)
    assert result is None
