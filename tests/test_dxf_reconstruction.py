"""
Tests for `backend.cv_extraction.dxf_reconstruction.reconstruct_building_polygon`
-- the deterministic fragmented-open-line-work-to-closed-polygon pipeline
(endpoint snapping, collinear-segment merging, closed-loop cycle
detection) tried before the DXF Vision-render fallback.
"""

from __future__ import annotations

import time

import pytest

from backend.cv_extraction.dxf_reconstruction import _cap_segments_for_reconstruction, reconstruct_building_polygon
from backend.schemas.geometry import Point, Polygon

_PLOT = Polygon(points=[Point(x=0, y=0), Point(x=20, y=0), Point(x=20, y=10), Point(x=0, y=10)])


def _rect_segments(x0, y0, x1, y1, layer="0"):
    return [
        (layer, (x0, y0), (x1, y0)),
        (layer, (x1, y0), (x1, y1)),
        (layer, (x1, y1), (x0, y1)),
        (layer, (x0, y1), (x0, y0)),
    ]


def test_reconstructs_exact_closed_rectangle_from_four_disconnected_lines():
    segments = _rect_segments(5, 3, 15, 7)
    result = reconstruct_building_polygon(segments, _PLOT)
    assert result is not None
    assert result.polygon.area == pytest.approx(40.0, abs=1e-6)
    bbox = result.polygon.bounding_box
    assert bbox.width == pytest.approx(10.0, abs=1e-6)
    assert bbox.height == pytest.approx(4.0, abs=1e-6)


def test_tolerates_small_endpoint_gaps_within_snap_tolerance():
    segments = [
        ("0", (5.0, 3.0), (15.0, 3.0)),
        ("0", (15.02, 3.0), (15.0, 7.0)),  # 0.02 gap, well under default 0.05 tolerance
        ("0", (15.0, 7.01), (5.0, 7.0)),
        ("0", (5.0, 7.0), (5.0, 3.02)),
    ]
    result = reconstruct_building_polygon(segments, _PLOT)
    assert result is not None
    assert result.polygon.area == pytest.approx(40.0, abs=0.5)


def test_returns_none_when_a_real_gap_leaves_no_closed_loop():
    # Only three of the four walls -- no closed loop exists at any tolerance.
    segments = _rect_segments(5, 3, 15, 7)[:3]
    result = reconstruct_building_polygon(segments, _PLOT)
    assert result is None


def test_merges_collinear_split_segments_without_adding_spurious_corners():
    # The "bottom" wall is split into three collinear pieces by an export
    # artifact -- must still reconstruct the same 10x4 rectangle.
    segments = [
        ("0", (5, 3), (8, 3)),
        ("0", (8, 3), (12, 3)),
        ("0", (12, 3), (15, 3)),
        ("0", (15, 3), (15, 7)),
        ("0", (15, 7), (5, 7)),
        ("0", (5, 7), (5, 3)),
    ]
    result = reconstruct_building_polygon(segments, _PLOT)
    assert result is not None
    assert result.polygon.area == pytest.approx(40.0, abs=1e-6)


def test_ignores_segments_on_annotation_layers():
    # A dimension line cutting straight across the interior must not be
    # stitched into the outline as if it were a wall.
    segments = _rect_segments(5, 3, 15, 7) + [("DIMENSIONS", (5, 5), (15, 5))]
    result = reconstruct_building_polygon(segments, _PLOT)
    assert result is not None
    assert result.polygon.area == pytest.approx(40.0, abs=1e-6)


def test_ignores_dimension_only_layer_leaves_nothing_to_reconstruct():
    segments = [("DIM", (5, 3), (15, 3)), ("TEXT", (15, 3), (15, 7))]
    result = reconstruct_building_polygon(segments, _PLOT)
    assert result is None


def test_rejects_a_loop_implausibly_small_relative_to_the_plot():
    # A tiny 0.2 x 0.2 closed loop (e.g. a furniture symbol traced as a
    # closed shape) is not a building footprint on a 20x10 plot.
    segments = _rect_segments(1.0, 1.0, 1.2, 1.2)
    result = reconstruct_building_polygon(segments, _PLOT)
    assert result is None


def test_no_plot_polygon_still_reconstructs_from_shape_alone():
    segments = _rect_segments(5, 3, 15, 7)
    result = reconstruct_building_polygon(segments, None)
    assert result is not None
    assert result.polygon.area == pytest.approx(40.0, abs=1e-6)


def test_empty_input_returns_none():
    assert reconstruct_building_polygon([], _PLOT) is None


def test_rejects_self_intersecting_bowtie_cycle():
    # Four fragments whose only closed loop is a figure-eight/bowtie: the
    # cycle-basis detector guarantees a closed sequence of nodes, not a
    # SIMPLE (non-crossing) one. A=(10,10), D=(10,0), B=(0,5), C=(0,0);
    # edges A-D and B-C are the two "sides" and D-B/C-A are the two
    # crossing diagonals of the loop -- the resulting quadrilateral has a
    # nonzero shoelace area despite crossing itself, so this must be
    # rejected by an explicit simple-polygon check, not by area alone.
    segments = [
        ("0", (10.0, 10.0), (10.0, 0.0)),  # A-D
        ("0", (10.0, 0.0), (0.0, 5.0)),    # D-B
        ("0", (0.0, 5.0), (0.0, 0.0)),     # B-C
        ("0", (0.0, 0.0), (10.0, 10.0)),   # C-A
    ]
    assert reconstruct_building_polygon(segments, _PLOT) is None


def test_cap_keeps_longest_segments_when_over_the_limit(monkeypatch):
    """`_cap_segments_for_reconstruction` bounds the O(n^2) _merge_collinear
    stage (see backend.config.Settings.max_dxf_reconstruction_segments) --
    pin that it keeps the LONGEST segments (real walls/boundary runs), not
    an arbitrary prefix, when the input exceeds the cap."""
    from backend.config import get_settings

    get_settings.cache_clear()
    long_segments = [((0.0, float(i)), (10.0, float(i))) for i in range(5)]
    short_segments = [((0.0, float(i) + 0.5), (0.01, float(i) + 0.5)) for i in range(20)]
    monkeypatch.setattr(
        "backend.config.get_settings",
        lambda: get_settings().model_copy(update={"max_dxf_reconstruction_segments": 5}),
    )
    capped = _cap_segments_for_reconstruction(long_segments + short_segments)
    assert len(capped) == 5
    assert all(seg in long_segments for seg in capped)


def test_snap_endpoints_never_sees_more_than_the_capped_segment_count(monkeypatch):
    """Precise (non-timing-based) regression: however large the input
    segment list is, the expensive graph-building stages
    (_snap_endpoints -> _merge_collinear) must only ever run on at most
    `max_dxf_reconstruction_segments` of them -- that bound is what
    actually protects _merge_collinear's worst-case O(nodes^2) iteration
    from scaling with untrusted input size."""
    import backend.cv_extraction.dxf_reconstruction as recon

    seen_lengths = []
    real_snap_endpoints = recon._snap_endpoints

    def _spy(segments, tol):
        seen_lengths.append(len(segments))
        return real_snap_endpoints(segments, tol)

    monkeypatch.setattr(recon, "_snap_endpoints", _spy)

    real = _rect_segments(5, 3, 15, 7)
    noise = [("0", (20.0 + 0.01 * i, 20.0), (20.0 + 0.01 * i + 0.005, 20.0)) for i in range(20000)]

    from backend.config import get_settings

    get_settings.cache_clear()
    limit = get_settings().max_dxf_reconstruction_segments
    result = reconstruct_building_polygon(real + noise, _PLOT)

    assert seen_lengths and seen_lengths[0] <= limit
    assert result is not None
    assert result.polygon.area == pytest.approx(40.0, abs=1e-6)


def test_reconstruction_stays_fast_with_far_more_segments_than_the_cap():
    """Wall-clock smoke test alongside the precise check above: a segment
    list far larger than the cap must not make overall reconstruction time
    scale with input size."""
    real = _rect_segments(5, 3, 15, 7)
    noise = [("0", (20.0 + 0.01 * i, 20.0), (20.0 + 0.01 * i + 0.005, 20.0)) for i in range(20000)]
    t0 = time.time()
    result = reconstruct_building_polygon(real + noise, _PLOT)
    elapsed = time.time() - t0
    assert elapsed < 15.0, f"reconstruction took {elapsed:.1f}s with 20000 input segments -- regression toward unbounded scaling"
    assert result is not None
    assert result.polygon.area == pytest.approx(40.0, abs=1e-6)
