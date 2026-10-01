"""
Architecture V2, Phase 7 -- tests for the pure helper functions in
`backend.tools.run_evidence_decision_shadow`. The script's actual
extraction/comparison logic is exercised directly against the real plans
(see ARCHITECTURE_V2.md's Implementation log for that run's results); this
file covers the small, deterministic pieces in isolation.
"""

from __future__ import annotations

from backend.schemas.enums import ConfidenceLevel
from backend.schemas.evidence import Confidence, ValueField
from backend.schemas.geometry import NormalizedGeometry
from backend.schemas.normalized_plan import (
    BuildingSection,
    NormalizedPlan,
    PlotSection,
    RoadSection,
    SetbackSection,
)
from backend.tools.run_evidence_decision_shadow import _real_plan_field, _unit_for_field, _values_agree


def test_unit_for_field_picks_area_unit_for_area_like_fields():
    assert _unit_for_field("plot.area") == "sq_m"
    assert _unit_for_field("coverage") == "sq_m"
    assert _unit_for_field("far") == "sq_m"
    assert _unit_for_field("far.area") == "sq_m"


def test_unit_for_field_picks_length_unit_otherwise():
    assert _unit_for_field("plot.width") == "m"
    assert _unit_for_field("setbacks.front") == "m"
    assert _unit_for_field("road.width") == "m"


def test_values_agree_treats_both_none_as_agreement():
    assert _values_agree(None, None) is True


def test_values_agree_treats_one_none_as_disagreement():
    assert _values_agree(None, 1.0) is False
    assert _values_agree(1.0, None) is False


def test_values_agree_is_tolerant_of_tiny_float_noise_but_not_a_real_difference():
    assert _values_agree(17.59, 17.5900001) is True
    # A genuinely different value (well beyond the 0.1% relative tolerance
    # used to absorb float noise between two independent extraction paths).
    assert _values_agree(17.59, 15.35) is False


def _vf(value: float, level: ConfidenceLevel = ConfidenceLevel.HIGH) -> ValueField[float]:
    return ValueField[float](value=value, confidence=Confidence(level=level, score=0.9))


def _minimal_plan() -> NormalizedPlan:
    return NormalizedPlan(
        plan_id="p1", source_document_id="d1",
        plot=PlotSection(width=_vf(10.0), depth=_vf(20.0), area=_vf(200.0)),
        building=BuildingSection(width=_vf(5.0), depth=_vf(8.0), footprint_area=_vf(40.0)),
        road=RoadSection(width=_vf(9.0)),
        setbacks=SetbackSection(front=_vf(1.0), rear=_vf(1.0), left=_vf(1.0), right=_vf(1.0)),
        coverage=_vf(20.0), far=_vf(1.0),
    )


def test_real_plan_field_maps_every_core_field():
    plan = _minimal_plan()
    for field_name, expected_value in [
        ("plot.width", 10.0), ("plot.depth", 20.0), ("plot.area", 200.0),
        ("building.width", 5.0), ("building.depth", 8.0), ("building.footprint_area", 40.0),
        ("road.width", 9.0),
        ("setbacks.front", 1.0), ("setbacks.rear", 1.0), ("setbacks.left", 1.0), ("setbacks.right", 1.0),
        ("coverage", 20.0), ("far", 1.0),
    ]:
        field = _real_plan_field(plan, field_name)
        assert field is not None, f"{field_name} should map to a real ValueField"
        assert field.value == expected_value


def test_real_plan_field_returns_none_for_fields_with_no_normalized_plan_representation():
    plan = _minimal_plan()
    assert _real_plan_field(plan, "plot.net_area") is None
    assert _real_plan_field(plan, "far.area") is None
    assert _real_plan_field(plan, "building.gross_built_up_area") is None
