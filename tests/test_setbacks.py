"""Tests for `backend.spatial_reasoning.setbacks`."""

from __future__ import annotations

from backend.schemas.enums import ConfidenceLevel
from backend.schemas.evidence import Confidence, ValueField
from backend.spatial_reasoning.dimension_classification import DimensionSemanticType, classify_dimensions
from backend.spatial_reasoning.front_side import resolve_front_side
from backend.spatial_reasoning.setbacks import axis_budget_cap, compute_setbacks
from tests.fixtures.geometry_builders import rect_polygon


def _vf(value: float) -> ValueField[float]:
    return ValueField[float](value=value, confidence=Confidence(level=ConfidenceLevel.HIGH))


def test_axis_budget_cap_none_when_extents_unknown():
    assert axis_budget_cap("left", None, None, None, None) is None
    assert axis_budget_cap("front", _vf(10.0), None, _vf(7.0), None) is None


def test_axis_budget_cap_reflects_plot_minus_building_extent():
    # plot.width=10, building.width=7 -> 3m total left+right budget (plus tolerance).
    cap = axis_budget_cap("left", _vf(10.0), _vf(12.0), _vf(7.0), _vf(9.0), tol=0.05)
    assert abs(cap - 3.05) < 1e-9


def test_axis_budget_cap_catches_the_building_wider_than_plot_case():
    """Reproduces the live bug: building.width (12.2101) resolved slightly
    *larger* than plot.width (12.19) -- physically impossible, but exactly
    the kind of near-tie two independent measurements of the same edge can
    produce. The budget must come out non-positive (never a large positive
    number an unrelated setback reading could then slip under)."""
    cap = axis_budget_cap("right", _vf(12.19), _vf(9.14), _vf(12.2101), _vf(7.8668), tol=0.05)
    assert cap < 0.1
    # The implausible value actually observed in production must be rejected.
    assert 1.2581 > cap


def test_setbacks_computed_from_building_to_plot_edge_distance():
    plot = rect_polygon(0, 0, 400, 480)
    building = rect_polygon(60, 40, 340, 420)  # inset 60 left/right, 40 from top, 60 from bottom
    fsr = resolve_front_side(plot, road_bbox=None, access_evidence=[])
    setbacks = compute_setbacks(building, fsr, classified_dimensions=[], points_per_metre=1.0)
    assert setbacks.left.value == 60.0
    assert setbacks.right.value == 60.0
    # No road evidence -> front defaults to the polygon's first ring edge
    # (the top edge, y=0); rear is the farthest edge (bottom, y=480).
    assert round(setbacks.front.value) == 40
    assert round(setbacks.rear.value) == 60


def test_setback_missing_when_no_building_or_dimension_evidence():
    plot = rect_polygon(0, 0, 400, 480)
    fsr = resolve_front_side(plot, road_bbox=None, access_evidence=[])
    setbacks = compute_setbacks(None, fsr, classified_dimensions=[], points_per_metre=1.0)
    assert setbacks.front.value is None
    assert setbacks.front.status.name == "MISSING"


def test_dimension_derived_setback_used_when_geometry_absent():
    plot = rect_polygon(0, 0, 400, 480)
    fsr = resolve_front_side(plot, road_bbox=None, access_evidence=[])
    from backend.spatial_reasoning.dimension_classification import ClassifiedDimension
    from backend.schemas.enums import ConfidenceLevel
    from tests.fixtures.geometry_builders import dim

    cd = ClassifiedDimension(
        dimension=dim(3.0, "m", label="FRONT SETBACK"),
        semantic_type=DimensionSemanticType.FRONT_SETBACK,
        confidence=ConfidenceLevel.MEDIUM,
        reasoning="test",
        value_metres=3.0,
    )
    setbacks = compute_setbacks(None, fsr, classified_dimensions=[cd], points_per_metre=1.0)
    assert setbacks.front.value == 3.0


def test_setback_reconciles_geometry_and_dimension_agreement():
    plot = rect_polygon(0, 0, 400, 480)
    building = rect_polygon(60, 40, 340, 420)
    fsr = resolve_front_side(plot, road_bbox=None, access_evidence=[])
    from backend.spatial_reasoning.dimension_classification import ClassifiedDimension
    from backend.schemas.enums import ConfidenceLevel
    from tests.fixtures.geometry_builders import dim

    # front setback geometry-derived distance will be 40 (in whatever
    # units points_per_metre=1.0 means here); supply a matching dimension.
    cd = ClassifiedDimension(
        dimension=dim(40.0, "m", label="FRONT SETBACK"),
        semantic_type=DimensionSemanticType.FRONT_SETBACK,
        confidence=ConfidenceLevel.MEDIUM,
        reasoning="test",
        value_metres=40.0,
    )
    setbacks = compute_setbacks(building, fsr, classified_dimensions=[cd], points_per_metre=1.0)
    assert setbacks.front.status.name in ("HIGH", "MEDIUM")
    assert abs(setbacks.front.value - 40.0) < 1.0


# --- flag_if_setback_implausible: orientation ---------------------------------

from backend.spatial_reasoning.setbacks import flag_if_setback_implausible  # noqa: E402


def _flag(side, value, pw, pd, bw, bd, **kw):
    field = _vf(value)
    return flag_if_setback_implausible(
        side, field, f"setbacks.{side}", pw and _vf(pw), pd and _vf(pd), bw and _vf(bw), bd and _vf(bd), **kw
    )


def test_sheet_derived_values_are_not_flagged_when_the_road_is_on_a_side():
    """Real PLAN4 numbers: plot 18.288 x 16.6116, building 16.4592 x 11.2776,
    road on the left, so the 2.66 m side gaps run along the DEPTH axis (5.33 m
    of room) even though the fixed frontage-relative map looks at the width
    axis (only 1.83 m of room)."""
    args = (18.288, 16.6116, 16.4592, 11.2776)
    assert _flag("left", 2.6637, *args).conflict is not None   # classifier convention flags it
    for side, value in (("left", 2.6637), ("right", 2.6703), ("front", 0.9144), ("rear", 0.9144)):
        out = _flag(side, value, *args, frontage_relative_axes=False)
        assert out.conflict is None
        assert out.value == value


def test_sheet_derived_front_setback_on_a_rotated_sheet_is_not_flagged():
    """Real PLAN5 numbers: front 3.0 m, depth budget 2.07 m, width budget 4.5 m."""
    out = _flag("front", 3.0, 17.5895, 9.2075, 13.0895, 7.1374, frontage_relative_axes=False)
    assert out.conflict is None


def test_a_setback_that_fits_on_neither_axis_is_still_flagged_and_left_unchanged():
    out = _flag("front", 6.0, 17.5895, 9.2075, 13.0895, 7.1374, frontage_relative_axes=False)
    assert out.conflict is not None
    assert out.value == 6.0
    assert out.confidence.level == ConfidenceLevel.HIGH


def test_agnostic_mode_abstains_unless_both_axes_are_resolved():
    out = flag_if_setback_implausible(
        "front", _vf(6.0), "setbacks.front", _vf(17.5895), None, _vf(13.0895), _vf(7.1374),
        frontage_relative_axes=False,
    )
    assert out.conflict is None
