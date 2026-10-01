"""
Direct unit tests for `backend.cv_extraction.dxf_dimension_chain` -- the
native-DXF-DIMENSION verification chain added for Critical Requirement 5
(text -> dimension graphic -> measured span -> geometric edge, by exact
endpoint coincidence, never proximity/nearest-number matching).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from backend.cv_extraction.dxf_dimension_chain import find_dimension_edge_matches
from backend.schemas.geometry import Point, Polygon


@dataclass
class _FakeDim:
    value: float
    raw_text: str
    defpoint2: Optional[tuple[float, float]]
    defpoint3: Optional[tuple[float, float]]


def _rect(x0, y0, x1, y1) -> Polygon:
    return Polygon(points=[Point(x=x0, y=y0), Point(x=x1, y=y0), Point(x=x1, y=y1), Point(x=x0, y=y1)])


def test_matches_dimension_whose_points_coincide_with_an_edge():
    poly = _rect(0, 0, 20, 15)
    dim = _FakeDim(value=20.0, raw_text="20.00", defpoint2=(0, 0), defpoint3=(20, 0))
    matches = find_dimension_edge_matches([dim], poly)
    assert len(matches) == 1
    assert matches[0].value == 20.0
    assert matches[0].edge_length == 20.0


def test_matches_regardless_of_point_order():
    poly = _rect(0, 0, 20, 15)
    dim = _FakeDim(value=20.0, raw_text="", defpoint2=(20, 0), defpoint3=(0, 0))
    matches = find_dimension_edge_matches([dim], poly)
    assert len(matches) == 1


def test_does_not_match_a_dimension_merely_near_an_edge():
    """A dimension whose points sit close to (but not exactly at) an
    edge's endpoints -- e.g. it measures some OTHER nearby span, like a
    setback or a different wall -- must not be treated as verifying this
    edge. This is a coincidence check, not a nearest-neighbour search."""
    poly = _rect(0, 0, 20, 15)
    dim = _FakeDim(value=18.0, raw_text="", defpoint2=(1.0, 0.0), defpoint3=(19.0, 0.0))
    matches = find_dimension_edge_matches([dim], poly)
    assert matches == []


def test_ignores_dimensions_without_defpoints():
    """Radius/diameter/angular dimensions (or malformed entities) leave
    defpoint2/defpoint3 as None -- must be skipped, not crash."""
    poly = _rect(0, 0, 20, 15)
    dim = _FakeDim(value=5.0, raw_text="R5.0", defpoint2=None, defpoint3=None)
    matches = find_dimension_edge_matches([dim], poly)
    assert matches == []


def test_matches_a_rotated_polygons_edge():
    """The coincidence check is purely geometric (point-to-point
    distance), so it works identically regardless of the polygon's own
    orientation -- no page-space horizontal/vertical assumption."""
    import math

    theta = math.radians(37.0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    def rot(x, y):
        return (x * cos_t - y * sin_t, x * sin_t + y * cos_t)

    corners = [rot(0, 0), rot(20, 0), rot(20, 15), rot(0, 15)]
    poly = Polygon(points=[Point(x=x, y=y) for x, y in corners])
    dim = _FakeDim(value=20.0, raw_text="", defpoint2=corners[0], defpoint3=corners[1])
    matches = find_dimension_edge_matches([dim], poly)
    assert len(matches) == 1
    assert matches[0].edge_length == 20.0 or abs(matches[0].edge_length - 20.0) < 1e-6


def test_multiple_dimensions_can_match_different_edges():
    poly = _rect(0, 0, 20, 15)
    width_dim = _FakeDim(value=20.0, raw_text="", defpoint2=(0, 0), defpoint3=(20, 0))
    depth_dim = _FakeDim(value=15.0, raw_text="", defpoint2=(20, 0), defpoint3=(20, 15))
    matches = find_dimension_edge_matches([width_dim, depth_dim], poly)
    values = sorted(m.value for m in matches)
    assert values == [15.0, 20.0]


def test_empty_inputs_return_no_matches():
    poly = _rect(0, 0, 20, 15)
    assert find_dimension_edge_matches([], poly) == []
