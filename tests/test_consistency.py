"""Tests for `backend.spatial_reasoning.consistency`."""

from __future__ import annotations

from backend.schemas.enums import ConfidenceLevel
from backend.schemas.evidence import Confidence, ValueField
from backend.schemas.geometry import CoordinateSpace, NormalizedGeometry
from backend.schemas.normalized_plan import BuildingSection, PlotSection, SetbackSection
from backend.spatial_reasoning.consistency import check_physical_consistency
from tests.fixtures.geometry_builders import rect_polygon


def _vf(value, level=ConfidenceLevel.HIGH):
    return ValueField[float](value=value, confidence=Confidence(level=level))


def _plot_section(width, depth, polygon=None, ppm=1.0):
    geom = None
    if polygon is not None:
        geom = NormalizedGeometry(coordinate_space=CoordinateSpace.METRIC_PLAN, polygon=polygon, bounding_box=polygon.bounding_box, points_per_metre=ppm)
    return PlotSection(geometry=geom, width=_vf(width), depth=_vf(depth), area=_vf(width * depth if width and depth else None))


def _building_section(width, depth):
    return BuildingSection(width=_vf(width), depth=_vf(depth), footprint_area=_vf((width or 0) * (depth or 0)))


def _setbacks(front, rear, left, right):
    return SetbackSection(front=_vf(front), rear=_vf(rear), left=_vf(left), right=_vf(right))


def test_building_wider_than_plot_flagged_as_impossible():
    plot = _plot_section(10.0, 12.0, polygon=rect_polygon(0, 0, 10, 12))
    building = _building_section(15.0, 5.0)  # wider than the plot
    setbacks = _setbacks(1.0, 1.0, 1.0, 1.0)
    conflicts = check_physical_consistency(plot, building, setbacks)
    assert any("building.width" in c.description and "plot.width" in c.description for c in conflicts)


def test_building_deeper_than_plot_flagged_as_impossible():
    plot = _plot_section(10.0, 12.0, polygon=rect_polygon(0, 0, 10, 12))
    building = _building_section(5.0, 20.0)  # deeper than the plot
    setbacks = _setbacks(1.0, 1.0, 1.0, 1.0)
    conflicts = check_physical_consistency(plot, building, setbacks)
    assert any("building.depth" in c.description and "plot.depth" in c.description for c in conflicts)


def test_negative_setback_is_flagged():
    plot = _plot_section(10.0, 12.0, polygon=rect_polygon(0, 0, 10, 12))
    building = _building_section(9.5, 11.0)
    setbacks = _setbacks(-0.5, 1.0, 1.0, 1.0)
    conflicts = check_physical_consistency(plot, building, setbacks)
    assert any("setbacks.front" in c.description and "negative" in c.description.lower() for c in conflicts)


def test_valid_rectangular_geometry_has_no_conflicts():
    plot = _plot_section(10.0, 12.0, polygon=rect_polygon(0, 0, 10, 12))
    building = _building_section(6.0, 8.0)
    setbacks = _setbacks(2.0, 2.0, 2.0, 2.0)
    conflicts = check_physical_consistency(plot, building, setbacks)
    assert conflicts == []


def test_rectangular_cross_check_flags_mismatched_setback_sum():
    plot = _plot_section(10.0, 12.0, polygon=rect_polygon(0, 0, 10, 12))
    building = _building_section(6.0, 8.0)
    # left+right = 2+2=4, but building.width(6)+4=10 matches; break it deliberately
    setbacks = _setbacks(1.0, 1.0, 1.0, 8.0)  # 6 + 1 + 8 = 15 != 10
    conflicts = check_physical_consistency(plot, building, setbacks)
    assert any("Rectangular cross-check failed" in c.description and "plot.width" in c.description for c in conflicts)


def test_irregular_plot_skips_rectangular_cross_check():
    """
    For irregular (non-rectangular) plots, the width/depth cross-check
    must be SKIPPED — per phase3.md, that validation only applies "for
    rectangular plots"; irregular plots should use polygon/segment
    distance logic instead, not a bounding-box-style sum-of-sides check.
    """
    from backend.schemas.geometry import Point, Polygon

    l_shape = Polygon(
        points=[
            Point(x=0, y=0),
            Point(x=10, y=0),
            Point(x=10, y=5),
            Point(x=6, y=5),
            Point(x=6, y=10),
            Point(x=0, y=10),
        ]
    )
    plot = _plot_section(10.0, 10.0, polygon=l_shape)
    building = _building_section(6.0, 8.0)
    # Deliberately mismatched sums that would fail the rectangular check.
    setbacks = _setbacks(1.0, 1.0, 1.0, 8.0)
    conflicts = check_physical_consistency(plot, building, setbacks)
    assert not any("Rectangular cross-check failed" in c.description for c in conflicts)


def test_missing_fields_do_not_raise_or_false_flag():
    plot = _plot_section(None, None)
    building = _building_section(None, None)
    setbacks = _setbacks(None, None, None, None)
    conflicts = check_physical_consistency(plot, building, setbacks)
    assert conflicts == []
