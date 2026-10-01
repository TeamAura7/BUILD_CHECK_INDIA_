"""Tests for `backend.spatial_reasoning.building_resolution`."""

from __future__ import annotations

from backend.schemas.geometry import Point, Polygon
from backend.spatial_reasoning.building_resolution import filter_and_rank_buildings, surviving_candidates
from tests.fixtures.geometry_builders import (
    BoundingBox,
    building_candidate,
    make_extraction,
    plot_candidate,
    rect_polygon,
    text,
)


def _plot():
    return plot_candidate(rect_polygon(0, 0, 400, 480))


def test_real_footprint_survives():
    plot = _plot()
    b = building_candidate(rect_polygon(60, 60, 340, 420))
    er = make_extraction(plot_candidates=[plot], building_candidates=[b])
    filtered = filter_and_rank_buildings(er, plot)
    survivors = surviving_candidates(filtered)
    assert len(survivors) == 1
    assert survivors[0].id == b.id


def test_room_labeled_candidate_is_rejected_not_confused_with_footprint():
    plot = _plot()
    room = building_candidate(rect_polygon(70, 70, 150, 130), cid="room-1")
    er = make_extraction(
        plot_candidates=[plot],
        building_candidates=[room],
        text_evidence=[text("BEDROOM 1", bbox=BoundingBox(min_x=80, min_y=80, max_x=140, max_y=120))],
    )
    filtered = filter_and_rank_buildings(er, plot)
    assert filtered[0].kept is False
    assert "room" in filtered[0].reject_reason.lower() or "internal" in filtered[0].reject_reason.lower()


def test_compound_wall_almost_same_as_plot_is_rejected():
    plot = _plot()
    compound_wall = building_candidate(rect_polygon(2, 2, 398, 478), cid="wall")
    er = make_extraction(plot_candidates=[plot], building_candidates=[compound_wall])
    filtered = filter_and_rank_buildings(er, plot)
    assert filtered[0].kept is False


def test_tiny_furniture_sized_candidate_is_rejected():
    plot = _plot()
    tiny = building_candidate(rect_polygon(100, 100, 103, 103), cid="furniture")
    er = make_extraction(plot_candidates=[plot], building_candidates=[tiny])
    filtered = filter_and_rank_buildings(er, plot)
    assert filtered[0].kept is False


def test_plot_area_filter_uses_true_polygon_area_not_bounding_box():
    """
    Audit finding (BUILDCHECK_FORENSIC_AUDIT.md Sec 4.5): the plot-area
    fraction filter used `plot_bbox.width * plot_bbox.height` -- for a
    non-rectangular plot, the bounding box can substantially overstate the
    true plot area, silently loosening/tightening `_MIN_PLOT_FRACTION`/
    `_MAX_PLOT_FRACTION` in a way unrelated to the actual shape.

    This trapezoid's true (shoelace) polygon area is 120000 (page-points^2);
    its axis-aligned bounding box area is 192000 -- 60% larger. A building
    candidate of area 1500 sits at fraction 0.0125 of the true polygon area
    (above `_MIN_PLOT_FRACTION=0.01`, correctly kept) but only 0.0078 of the
    inflated bounding-box area (below 0.01, incorrectly rejected as
    furniture/annotation noise before this fix).
    """
    plot = plot_candidate(Polygon(points=[
        Point(x=0, y=0), Point(x=400, y=0), Point(x=250, y=480), Point(x=150, y=480),
    ]))
    building = building_candidate(rect_polygon(150, 200, 200, 230))  # 50 x 30 = 1500
    er = make_extraction(plot_candidates=[plot], building_candidates=[building])

    filtered = filter_and_rank_buildings(er, plot)

    assert filtered[0].kept is True


def test_extreme_aspect_ratio_dimension_line_artifact_is_rejected():
    plot = _plot()
    line_artifact = building_candidate(rect_polygon(50, 200, 350, 202), cid="dim-line")
    er = make_extraction(plot_candidates=[plot], building_candidates=[line_artifact])
    filtered = filter_and_rank_buildings(er, plot)
    assert filtered[0].kept is False


def test_multiple_building_blocks_all_survive():
    plot = _plot()
    block_a = building_candidate(rect_polygon(30, 30, 150, 200), cid="block-a")
    block_b = building_candidate(rect_polygon(200, 250, 350, 440), cid="block-b")
    er = make_extraction(plot_candidates=[plot], building_candidates=[block_a, block_b])
    filtered = filter_and_rank_buildings(er, plot)
    survivors = surviving_candidates(filtered)
    assert {s.id for s in survivors} == {"block-a", "block-b"}


def test_candidate_nested_inside_kept_candidate_is_rejected():
    plot = _plot()
    outer = building_candidate(rect_polygon(50, 50, 350, 430), cid="outer")
    inner_wall = building_candidate(rect_polygon(100, 100, 200, 200), cid="inner-wall")
    er = make_extraction(plot_candidates=[plot], building_candidates=[outer, inner_wall])
    filtered = filter_and_rank_buildings(er, plot)
    survivors = surviving_candidates(filtered)
    assert survivors == [outer]


def test_candidate_on_different_page_than_plot_is_excluded():
    plot = _plot()
    other_page = building_candidate(rect_polygon(60, 60, 340, 420), cid="other-page", page=1)
    er = make_extraction(plot_candidates=[plot], building_candidates=[other_page])
    filtered = filter_and_rank_buildings(er, plot)
    assert filtered[0].kept is False
