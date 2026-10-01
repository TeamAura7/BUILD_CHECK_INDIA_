"""Physical-impossibility gate (`spatial_reasoning.physical_bounds`) and the DXF row-formula fix."""

from __future__ import annotations

import pytest

from backend.cv_extraction.dxf_extractor import _without_row_formulas
from backend.schemas.enums import ConfidenceLevel
from backend.schemas.evidence import Confidence, ValueField
from backend.schemas.normalized_plan import BuildingSection, NormalizedPlan, PlotSection, RoadSection, SetbackSection
from backend.spatial_reasoning.physical_bounds import enforce_physical_bounds


def vf(x, level=ConfidenceLevel.HIGH):
    if x is None:
        return ValueField.missing("absent")
    return ValueField(value=x, confidence=Confidence(level=level, reason="test"))


def plan(plot_area=300.0, pw=15.0, pd=20.0, bw=8.0, bd=10.0, fp=80.0, sb=(3.0, 2.0, 1.5, 1.5), cov=26.7, far=1.2):
    return NormalizedPlan(
        plan_id="t", source_document_id="t",
        plot=PlotSection(width=vf(pw), depth=vf(pd), area=vf(plot_area)),
        building=BuildingSection(width=vf(bw), depth=vf(bd), footprint_area=vf(fp)),
        road=RoadSection(width=vf(9.0)),
        setbacks=SetbackSection(front=vf(sb[0]), rear=vf(sb[1]), left=vf(sb[2]), right=vf(sb[3])),
        coverage=vf(cov), far=vf(far),
    )


def test_a_plausible_plan_is_left_untouched():
    p = enforce_physical_bounds(plan())
    assert p.plot.area.value == 300.0 and p.setbacks.front.value == 3.0 and p.far.value == 1.2
    assert "physical_bounds" not in p.metadata


@pytest.mark.parametrize("area", [1.0, 0.0185, 11.9, 600_000.0])
def test_an_impossible_plot_area_withdraws_the_whole_geometry_chain(area):
    p = enforce_physical_bounds(plan(plot_area=area))
    for vfield in (p.plot.area, p.plot.width, p.building.footprint_area, p.setbacks.left, p.coverage, p.far):
        assert vfield.value is None and vfield.confidence.level == ConfidenceLevel.MISSING
    assert p.road.width.value == 9.0            # road width is not part of this geometry chain
    assert any(e["field"] == "plot.area" for e in p.metadata["physical_bounds"])
    assert "Physical-bounds gate" in p.overall_confidence_note


def test_a_building_that_is_the_plot_outline_is_withdrawn_with_its_setbacks():
    p = enforce_physical_bounds(plan(fp=290.0, sb=(3.0, 2.0, 1.5, 1.5)))
    assert p.building.footprint_area.value is None and p.setbacks.front.value is None and p.coverage.value is None
    assert p.plot.area.value is None            # which of the two is wrong is unknowable, so neither ships


def test_four_zero_setbacks_mean_the_building_is_the_plot_and_are_withdrawn():
    p = enforce_physical_bounds(plan(sb=(0.0, 0.0, 0.0, 0.0)))
    assert p.setbacks.rear.value is None and p.building.width.value is None
    assert p.plot.width.value == 15.0


def test_setbacks_of_one_wall_thickness_on_three_sides_are_withdrawn():
    p = enforce_physical_bounds(plan(sb=(0.0, 0.23, 0.23, 0.23)))
    assert p.setbacks.rear.value is None and p.coverage.value is None
    q = enforce_physical_bounds(plan(sb=(3.0, 2.0, 0.0, 0.2)))     # two closed sides (semi-detached) is legitimate
    assert q.setbacks.front.value == 3.0


def test_a_single_zero_setback_is_legitimate():
    p = enforce_physical_bounds(plan(sb=(3.0, 2.0, 0.0, 1.5)))
    assert p.setbacks.left.value == 0.0 and p.setbacks.front.value == 3.0


def test_a_setback_longer_than_the_plot_is_withdrawn_alone():
    p = enforce_physical_bounds(plan(sb=(45.0, 2.0, 1.5, 1.5)))
    assert p.setbacks.front.value is None and p.setbacks.rear.value == 2.0


def test_coverage_above_100_percent_withdraws_coverage_and_far():
    p = enforce_physical_bounds(plan(cov=140.0))
    assert p.coverage.value is None and p.far.value is None and p.setbacks.front.value == 3.0


def test_withdrawn_values_are_never_marked_conflicting():
    """CONFLICTING short-circuits the compliance engine; a withdrawn value is MISSING."""
    p = enforce_physical_bounds(plan(plot_area=1.0))
    assert p.plot.area.confidence.level == ConfidenceLevel.MISSING


@pytest.mark.parametrize("text,expected", [
    ("3.  BALANCE AREA OF PLOT (1-2) :", "3.  BALANCE AREA OF PLOT   :"),
    ("13.  TOTAL BUILT UP AREA PROPOSED (10+11+12)", "13.  TOTAL BUILT UP AREA PROPOSED  "),
    ("FAR Area (1.64)", "FAR Area (1.64)"),
    ("SITE AREA 120.5 SQM", "SITE AREA 120.5 SQM"),
])
def test_row_formula_references_are_not_values(text, expected):
    assert _without_row_formulas(text) == expected


def test_dxf_label_reader_no_longer_reads_row_numbers_as_areas():
    from backend.cv_extraction.dxf_extractor import _AREA_FIELD_PATTERNS
    plot_pat = dict((f, p) for f, p, _ in _AREA_FIELD_PATTERNS)["plot.area"]
    assert plot_pat.search(_without_row_formulas("3.  BALANCE AREA OF PLOT (1-2) :")) is None
    assert plot_pat.search(_without_row_formulas("AREA OF PLOT (1-2) : 234.5")).group("value") == "234.5"


def test_a_percentage_is_never_read_as_an_area():
    """PLAN8: 'Proposed Coverage Area (62.83 %)' shipped as a 62.83 m2 footprint at HIGH."""
    from backend.cv_extraction.dxf_extractor import _AREA_FIELD_PATTERNS, _PERCENT_TAIL_RE
    pat = dict((f, p) for f, p, _ in _AREA_FIELD_PATTERNS)["building.footprint_area"]
    text = "Proposed Coverage Area (62.83 %)"
    m = pat.search(text)
    assert m and _PERCENT_TAIL_RE.match(text, m.end("value"))
    m2 = pat.search("Proposed Coverage Area 115.42 sqm")
    assert m2 and not _PERCENT_TAIL_RE.match("Proposed Coverage Area 115.42 sqm", m2.end("value"))
