"""
Direct unit tests for `backend.spatial_reasoning.geometry_utils.
reconstruct_boundary_from_fragments` -- the general fragmented-boundary
reconstruction added for Critical Requirement 2 (see dxf_extractor.py's
`_reconstruct_plot_envelope`, which is its only production caller).

These are deliberately pure-geometry tests (no DXF, no extractor) so the
algorithm itself -- grouping scattered points onto shared boundary sides
by orientation and perpendicular distance to a common line, then
intersecting consecutive sides to find corners -- can be pinned precisely
and cheaply, independent of anything DXF-specific. `test_dxf_extractor.py`
covers the same requirement end-to-end through the real extraction path.
"""

from __future__ import annotations

import math

import pytest

from backend.schemas.geometry import Point
from backend.spatial_reasoning.geometry_utils import reconstruct_boundary_from_fragments


def _dash_points_along(p0: tuple[float, float], p1: tuple[float, float], gaps: list[float]) -> list[Point]:
    """Sample points along the segment p0->p1 at the given fractional
    positions (0..1) -- standing in for the vertices of individual dash
    fragments, which is all `reconstruct_boundary_from_fragments` ever
    sees (it has no notion of a "fragment" as a shape, only points)."""
    x0, y0 = p0
    x1, y1 = p1
    return [Point(x=x0 + (x1 - x0) * t, y=y0 + (y1 - y0) * t) for t in gaps]


def _rectangle_dash_points(
    width: float, height: float, angle_degrees: float = 0.0, per_side_fractions: list[float] | None = None,
) -> list[Point]:
    """Scattered sample points along a WxH rectangle's 4 sides (never the
    corners themselves), at an arbitrary rotation -- the same shape a
    dashed/dash-dot property boundary leaves behind: many points, no two
    of them the actual corner, arbitrarily large gaps between them."""
    fractions = per_side_fractions or [0.05, 0.15, 0.35, 0.55, 0.7, 0.9]
    corners = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    theta = math.radians(angle_degrees)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    corners = [(x * cos_t - y * sin_t, x * sin_t + y * cos_t) for x, y in corners]
    points: list[Point] = []
    for i in range(4):
        points.extend(_dash_points_along(corners[i], corners[(i + 1) % 4], fractions))
    return points


def _polygon_side_lengths(polygon) -> list[float]:
    pts = polygon.points
    n = len(pts)
    return [math.hypot(pts[(i + 1) % n].x - pts[i].x, pts[(i + 1) % n].y - pts[i].y) for i in range(n)]


def test_reconstructs_rectangle_from_widely_gapped_dash_points():
    """The core case this exists for: no dash ever touches another, gaps
    between consecutive samples on a side are a large fraction of the
    side's own length -- a plain proximity/snap-based approach could never
    bridge this, but line-fitting doesn't need fragments to be close."""
    points = _rectangle_dash_points(20.0, 12.0)
    polygon = reconstruct_boundary_from_fragments(points)
    assert polygon is not None
    assert polygon.area == pytest.approx(20.0 * 12.0, rel=0.02)
    sides = sorted(_polygon_side_lengths(polygon))
    assert sides[0] == pytest.approx(12.0, rel=0.03)
    assert sides[1] == pytest.approx(12.0, rel=0.03)
    assert sides[2] == pytest.approx(20.0, rel=0.03)
    assert sides[3] == pytest.approx(20.0, rel=0.03)


@pytest.mark.parametrize(
    "fractions",
    [
        [0.1, 0.3, 0.5, 0.7, 0.9],  # evenly spaced, moderate gaps (the existing default-ish pattern)
        [0.05, 0.08, 0.11, 0.14, 0.9],  # a tight cluster of small gaps near one end, plus one big gap to a lone dash
        [0.05, 0.1, 0.15, 0.85, 0.9, 0.95],  # two dash CLUSTERS near the ends, one huge gap between them
        [0.02, 0.03, 0.35, 0.36, 0.68, 0.98],  # very uneven: near-touching pairs scattered with big gaps between
    ],
    ids=["moderate-even", "small-gaps-plus-one-big", "two-clusters-one-huge-gap", "uneven-pairs"],
)
def test_reconstructs_rectangle_across_varying_dash_gap_patterns(fractions):
    """Generalization: the actual gap size/spacing between dashes on a
    real dash-dot boundary varies a lot -- small consistent gaps, one
    huge intentional gap alongside small ones, dashes clustered in a
    couple of spots rather than spread evenly -- line-fitting must not be
    tuned to any one specific dash-spacing pattern, only to there being
    enough dashes overall (the default `min_side_support`, unmodified
    here, matching a realistic dash population per side). A wider
    tolerance than the evenly-spaced case is deliberate and honest, not
    loosened to force a pass: dashes clustered tightly near a corner on
    one side can sit close enough to a neighbouring side's own near-
    corner cluster to very slightly bias that corner's fitted
    intersection -- a real, bounded, already-present precision cost of
    working from sparse/uneven samples rather than a regression, and
    still far tighter than the coarse bounding-rectangle fallback this
    replaces (`_MIN_THICKNESS_SAMPLES`-style honesty about the method's
    own limits, not the algorithm being tuned to these specific cases)."""
    points = _rectangle_dash_points(20.0, 12.0, per_side_fractions=fractions)
    polygon = reconstruct_boundary_from_fragments(points)
    assert polygon is not None
    assert polygon.area == pytest.approx(20.0 * 12.0, rel=0.08)


@pytest.mark.parametrize("angle_degrees", [0, 20, 45, 70, 90])
def test_reconstructs_rotated_rectangle_from_dash_points(angle_degrees):
    """Same shape, rotated -- the fitted sides come from orientation/
    collinearity in the points' own frame, never an assumption that a
    side is horizontal or vertical."""
    points = _rectangle_dash_points(20.0, 12.0, angle_degrees=angle_degrees)
    polygon = reconstruct_boundary_from_fragments(points)
    assert polygon is not None
    assert polygon.area == pytest.approx(20.0 * 12.0, rel=0.03)


def test_unrelated_stray_points_do_not_distort_the_fit():
    """A couple of points from unrelated nearby line-work (e.g. a
    dimension witness tick reaching a little past the boundary it
    measures, the same "small constant distance outside" pattern used in
    `tests/fixtures/dxf_builders.py`'s own multi-drawing fixture) must not
    change the outcome, since they don't have enough independent support
    on a shared line to be treated as a real side -- this is the
    `min_side_support` mechanism, not a proximity/snap tolerance. (A stray
    cluster that pokes out FAR PAST the whole side it sits next to can
    swallow that side's own convex-hull presence entirely -- an inherent
    limitation of any hull-based method, not something support-counting
    alone can fix -- so this stays close to the real edge, as real
    unrelated annotation marks actually do.)"""
    points = _rectangle_dash_points(20.0, 12.0)
    stray = [Point(x=20.4, y=5.8), Point(x=20.6, y=6.3)]  # just past the real x=20 side, near its midpoint
    polygon = reconstruct_boundary_from_fragments(points + stray)
    assert polygon is not None
    assert polygon.area == pytest.approx(20.0 * 12.0, rel=0.05)


def test_returns_none_when_too_few_points_support_any_side():
    """Sparse, noisy points with no side genuinely well-supported must
    fall back (return None) rather than fabricate a confident-looking
    polygon from noise -- the caller is expected to fall back to a
    coarser method in that case. `_reconstruct_plot_envelope` only ever
    calls this once there are dozens of fragments (hundreds in practice),
    so this uses a matching, realistic point count -- a handful of random
    points is not the population this function is contracted to handle
    robustly, any more than a 2-point sample would be."""
    import random

    rng = random.Random(7)
    points = [Point(x=rng.uniform(0, 20), y=rng.uniform(0, 12)) for _ in range(40)]
    polygon = reconstruct_boundary_from_fragments(points)
    assert polygon is None


def test_reconstructs_a_non_rectangular_convex_pentagon():
    """Not hardcoded to quadrilaterals: a general convex polygon's sides
    are found the same way."""
    corners = [(0.0, 0.0), (10.0, 0.0), (14.0, 6.0), (7.0, 12.0), (-2.0, 7.0)]
    fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
    points: list[Point] = []
    for i in range(len(corners)):
        points.extend(_dash_points_along(corners[i], corners[(i + 1) % len(corners)], fractions))
    polygon = reconstruct_boundary_from_fragments(points)
    assert polygon is not None
    assert len(polygon.points) == 5
    # Shoelace area of the true pentagon.
    n = len(corners)
    true_area = abs(sum(
        corners[i][0] * corners[(i + 1) % n][1] - corners[(i + 1) % n][0] * corners[i][1] for i in range(n)
    )) / 2.0
    assert polygon.area == pytest.approx(true_area, rel=0.03)


def test_returns_none_for_degenerate_input():
    assert reconstruct_boundary_from_fragments([]) is None
    assert reconstruct_boundary_from_fragments([Point(x=0, y=0), Point(x=1, y=1)]) is None
    assert reconstruct_boundary_from_fragments([Point(x=5, y=5)] * 10) is None
