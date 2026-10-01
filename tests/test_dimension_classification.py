"""Tests for `backend.spatial_reasoning.dimension_classification`."""

from __future__ import annotations

from backend.schemas.geometry import Line, Point
from backend.spatial_reasoning import geometry_utils as geo
from backend.spatial_reasoning.dimension_classification import (
    DimensionSemanticType,
    classify_dimensions,
)
from backend.spatial_reasoning.front_side import resolve_front_side
from tests.fixtures.geometry_builders import DEFAULT_PPM, dim, rect_polygon


def _value_lookup_m(ppm):
    def _lookup(d):
        if (d.unit or "").lower() == "m":
            return d.magnitude
        return None

    return _lookup


def test_dimension_on_plot_boundary_classified_as_plot_width():
    plot = rect_polygon(0, 0, 400, 480)
    fsr = resolve_front_side(plot, road_bbox=None, access_evidence=[])
    # dimension line lying right along the top (front/rear axis) edge
    d = dim(10.0, "m", line=Line(start=Point(x=0, y=0), end=Point(x=400, y=0)))
    result = classify_dimensions(
        [d], plot, [], fsr.front_edges, fsr.rear_edges, fsr.left_edges, fsr.right_edges, [], _value_lookup_m(DEFAULT_PPM)
    )
    assert result[0].semantic_type in (DimensionSemanticType.PLOT_WIDTH, DimensionSemanticType.PLOT_DEPTH)
    assert result[0].value_metres == 10.0


def test_setback_keyword_and_geometry_agree_gives_high_confidence():
    plot = rect_polygon(0, 0, 400, 480)
    building = rect_polygon(60, 60, 340, 420)
    fsr = resolve_front_side(plot, road_bbox=None, access_evidence=[])
    # A short dimension line spanning the gap between the building's left
    # edge and the plot's left edge (perpendicular to that side).
    d = dim(1.5, "m", label="LEFT SETBACK", line=Line(start=Point(x=0, y=240), end=Point(x=60, y=240)))
    result = classify_dimensions(
        [d],
        plot,
        [building],
        fsr.front_edges,
        fsr.rear_edges,
        fsr.left_edges,
        fsr.right_edges,
        [],
        _value_lookup_m(DEFAULT_PPM),
    )
    assert result[0].semantic_type == DimensionSemanticType.LEFT_SETBACK


def test_room_dimension_inside_building_interior():
    building = rect_polygon(60, 60, 340, 420)
    d = dim(3.0, "m", label="BEDROOM", line=Line(start=Point(x=150, y=150), end=Point(x=200, y=150)))
    result = classify_dimensions(
        [d], None, [building], [], [], [], [], [], _value_lookup_m(DEFAULT_PPM)
    )
    assert result[0].semantic_type == DimensionSemanticType.ROOM_DIMENSION


def test_dimension_with_no_geometry_uses_keyword_fallback_at_low_confidence():
    d = dim(9.0, "m", label="ROAD WIDTH 9.0 M")
    result = classify_dimensions([d], None, [], [], [], [], [], [], _value_lookup_m(DEFAULT_PPM))
    assert result[0].semantic_type == DimensionSemanticType.ROAD_WIDTH
    from backend.schemas.enums import ConfidenceLevel

    assert result[0].confidence == ConfidenceLevel.LOW


def test_unclassifiable_dimension_becomes_other():
    d = dim(4.2, "m", label="XYZ 4.2")
    result = classify_dimensions([d], None, [], [], [], [], [], [], _value_lookup_m(DEFAULT_PPM))
    assert result[0].semantic_type == DimensionSemanticType.OTHER


def test_classification_is_not_pure_keyword_lookup():
    """
    A dimension with a misleading/absent label but clear geometric
    placement on the plot boundary should still be classified correctly
    from geometry alone — proving the algorithm isn't "find number ->
    classify by nearby word" as phase3.md explicitly forbids.
    """
    plot = rect_polygon(0, 0, 400, 480)
    fsr = resolve_front_side(plot, road_bbox=None, access_evidence=[])
    # No recognizable label keyword at all (a bare number), but the line
    # geometrically lies right along the plot boundary.
    d = dim(10.0, "m", label="10.0", line=Line(start=Point(x=0, y=0), end=Point(x=400, y=0)))
    result = classify_dimensions(
        [d], plot, [], fsr.front_edges, fsr.rear_edges, fsr.left_edges, fsr.right_edges, [], _value_lookup_m(DEFAULT_PPM)
    )
    assert result[0].semantic_type in (DimensionSemanticType.PLOT_WIDTH, DimensionSemanticType.PLOT_DEPTH)
