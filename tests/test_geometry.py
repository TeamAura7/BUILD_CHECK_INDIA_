from __future__ import annotations

import pytest

from backend.schemas.geometry import (
    BoundingBox,
    CoordinateSpace,
    Line,
    NormalizedGeometry,
    Point,
    Polygon,
)


def test_bounding_box_width_height():
    bb = BoundingBox(min_x=0, min_y=0, max_x=10, max_y=4)
    assert bb.width == 10
    assert bb.height == 4


def test_bounding_box_rejects_inverted_coords():
    with pytest.raises(Exception):
        BoundingBox(min_x=10, min_y=0, max_x=0, max_y=4)


def test_bounding_box_center():
    bb = BoundingBox(min_x=0, min_y=0, max_x=10, max_y=10)
    center = bb.center
    assert center.x == 5
    assert center.y == 5


def test_bounding_box_intersects():
    a = BoundingBox(min_x=0, min_y=0, max_x=5, max_y=5)
    b = BoundingBox(min_x=4, min_y=4, max_x=8, max_y=8)
    c = BoundingBox(min_x=10, min_y=10, max_x=12, max_y=12)
    assert a.intersects(b) is True
    assert a.intersects(c) is False


def test_line_length():
    line = Line(start=Point(x=0, y=0), end=Point(x=3, y=4))
    assert line.length == pytest.approx(5.0)


def test_polygon_area_square():
    square = Polygon(
        points=[Point(x=0, y=0), Point(x=4, y=0), Point(x=4, y=4), Point(x=0, y=4)]
    )
    assert square.area == pytest.approx(16.0)


def test_polygon_requires_at_least_three_points():
    with pytest.raises(Exception):
        Polygon(points=[Point(x=0, y=0), Point(x=1, y=1)])


def test_polygon_bounding_box():
    tri = Polygon(points=[Point(x=0, y=0), Point(x=4, y=0), Point(x=2, y=3)])
    bb = tri.bounding_box
    assert bb.min_x == 0 and bb.max_x == 4
    assert bb.min_y == 0 and bb.max_y == 3


def test_normalized_geometry_requires_positive_scale():
    with pytest.raises(Exception):
        NormalizedGeometry(points_per_metre=0)


def test_normalized_geometry_default_space():
    geo = NormalizedGeometry(points_per_metre=72.0)
    assert geo.coordinate_space == CoordinateSpace.METRIC_PLAN
