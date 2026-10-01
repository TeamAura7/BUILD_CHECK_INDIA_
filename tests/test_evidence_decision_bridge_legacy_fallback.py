"""
Phase 9 Ground Truth 2 validation, Fix 1 -- tests for the legacy-fallback and
derived-field-restoration logic added to
`backend.spatial_reasoning.evidence_decision_bridge`.

See PHASE9_REPORT.md Finding 1: the Phase 8 bridge only ever consulted
`independent_cv`/vision, so a field the OLD path resolved via the legacy
geometry/vision-semantic candidate pool (or via printed Area Statement text,
or via computing coverage/FAR from already-resolved areas) went MISSING the
moment the new engine shipped, on 28/96 real PDF field-rows. These tests
exercise the fix's own explicit test plan (TEST 1-11 in the task prompt):
independent_cv priority is preserved, the legacy fallback only fires on a
genuine gap, PLAN7-shaped false positives cannot leak back in via a LOW-
confidence legacy value, coverage/FAR are derived (not reimplemented) from
this engine's own resolved areas, and decision_id now distinguishes an
evaluated ABSTAIN/CONFLICT from a field the engine never touched at all.
"""

from __future__ import annotations

from backend.schemas.enums import ConfidenceLevel, DecisionStatus
from backend.schemas.evidence import Confidence, ValueField
from backend.schemas.independent_measurements import IndependentCVResult, IndependentMeasurement
from backend.schemas.normalized_plan import BuildingSection, NormalizedPlan, PlotSection, RoadSection, SetbackSection
from backend.schemas.vision import VisionDocumentResult
from backend.spatial_reasoning import document_evidence as doc_ev
from backend.spatial_reasoning.evidence_decision import decide_field
from backend.spatial_reasoning.evidence_decision_bridge import (
    apply_evidence_decision_fields_to_plan,
    compute_evidence_decision_fields,
)


def _empty_vision() -> VisionDocumentResult:
    return VisionDocumentResult(model_name="disabled", pages=[], enabled=False)


def _vf(value, level: ConfidenceLevel, source: str = "legacy", reason: str = "legacy value") -> ValueField[float]:
    return ValueField[float](value=value, confidence=Confidence(level=level, reason=reason), source=source)


def _plan_with(**overrides) -> NormalizedPlan:
    missing = lambda: ValueField[float].missing("test fixture")  # noqa: E731
    fields = dict(
        plot_width=missing(), plot_depth=missing(), plot_area=missing(),
        building_width=missing(), building_depth=missing(), building_footprint_area=missing(),
        road_width=missing(),
        setback_front=missing(), setback_rear=missing(), setback_left=missing(), setback_right=missing(),
        coverage=missing(), far=missing(),
    )
    fields.update(overrides)
    return NormalizedPlan(
        plan_id="p", source_document_id="doc",
        plot=PlotSection(width=fields["plot_width"], depth=fields["plot_depth"], area=fields["plot_area"]),
        building=BuildingSection(
            width=fields["building_width"], depth=fields["building_depth"],
            footprint_area=fields["building_footprint_area"],
        ),
        road=RoadSection(width=fields["road_width"]),
        setbacks=SetbackSection(
            front=fields["setback_front"], rear=fields["setback_rear"],
            left=fields["setback_left"], right=fields["setback_right"],
        ),
        coverage=fields["coverage"], far=fields["far"],
    )


# --- TEST 1: independent_cv stays authoritative over a competing legacy value ---


def test_independent_cv_remains_higher_priority_than_legacy_when_both_exist():
    cv = IndependentCVResult(document_id="p", measurements=[
        IndependentMeasurement(field="plot.width", value_m=12.19, source="NATIVE_TEXT", confidence=0.97),
    ])
    legacy_plan = _plan_with(plot_width=_vf(99.0, ConfidenceLevel.HIGH))
    field_decisions = compute_evidence_decision_fields(cv, _empty_vision(), legacy_plan=legacy_plan)
    _decision, vf = field_decisions["plot.width"]
    assert vf.value == 12.19
    assert not (vf.source or "").startswith("legacy_fallback")


# --- TEST 2: legacy fallback fires when independent_cv has nothing ---


def test_legacy_fallback_used_when_independent_cv_has_no_candidate():
    legacy_plan = _plan_with(building_width=_vf(10.59, ConfidenceLevel.HIGH, source="geometry (building bbox)"))
    field_decisions = compute_evidence_decision_fields(None, _empty_vision(), legacy_plan=legacy_plan)
    decision, vf = field_decisions["building.width"]
    assert vf.value == 10.59
    assert vf.source.startswith("legacy_fallback:")
    assert decision.status == DecisionStatus.ACCEPT
    assert decision.accepted_candidate_id == "legacy_resolved_plan"
    assert vf.decision_id == decision.id
    # Never ships at the same tier as an independently-corroborated value.
    assert vf.status in (ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM)


# --- TEST 3: absence of independent_cv is not automatically filled ---


def test_legacy_fallback_does_not_manufacture_a_value_when_legacy_is_also_empty():
    legacy_plan = _plan_with()  # every field MISSING
    field_decisions = compute_evidence_decision_fields(None, None, legacy_plan=legacy_plan)
    decision, vf = field_decisions["building.width"]
    assert vf.value is None
    assert decision.status == DecisionStatus.ABSTAIN


def test_legacy_fallback_skips_a_low_confidence_legacy_value():
    # A weak/lone legacy candidate (LOW confidence) must not leak through
    # just because independent_cv/vision found nothing either.
    legacy_plan = _plan_with(plot_width=_vf(5.0, ConfidenceLevel.LOW, reason="single weak geometric candidate"))
    field_decisions = compute_evidence_decision_fields(None, None, legacy_plan=legacy_plan)
    decision, vf = field_decisions["plot.width"]
    assert vf.value is None
    assert decision.status == DecisionStatus.ABSTAIN


# --- TEST 4: PLAN7-shaped false positive does not return generically ---


def test_low_confidence_legacy_plot_and_building_do_not_regress_into_false_positive():
    # Simulates PLAN7: the legacy geometry pipeline produced SOME spurious
    # plot/building numbers (a photograph with no real vector geometry),
    # but plot_confidence's existing usability floor already capped them to
    # LOW -- no PLAN7-specific code anywhere in this test or in the fix.
    legacy_plan = _plan_with(
        plot_width=_vf(7.3, ConfidenceLevel.LOW, reason="the only plot candidate found scored 0.59, below the usability floor"),
        plot_depth=_vf(4.1, ConfidenceLevel.LOW),
        building_width=_vf(6.0, ConfidenceLevel.LOW),
        building_depth=_vf(3.5, ConfidenceLevel.LOW),
    )
    field_decisions = compute_evidence_decision_fields(None, None, legacy_plan=legacy_plan)
    for field in ("plot.width", "plot.depth", "building.width", "building.depth"):
        decision, vf = field_decisions[field]
        assert vf.value is None, f"{field} should remain unresolved, not regress to the spurious legacy value"
        assert decision.status == DecisionStatus.ABSTAIN


# --- TEST 5/6/7: derived coverage/FAR/area use the NEW engine's resolved inputs ---


def test_coverage_is_derived_from_new_engine_resolved_footprint_and_plot_area():
    cv = IndependentCVResult(document_id="p", measurements=[
        IndependentMeasurement(field="building.footprint_area", value_m=174.52, source="NATIVE_TEXT", confidence=0.9),
        IndependentMeasurement(field="plot.area", value_m=222.83, source="NATIVE_TEXT", confidence=0.9),
    ])
    legacy_plan = _plan_with(coverage=_vf(50.0, ConfidenceLevel.HIGH, reason="stale legacy coverage"))
    field_decisions = compute_evidence_decision_fields(cv, _empty_vision(), legacy_plan=legacy_plan)
    decision, vf = field_decisions["coverage"]
    assert vf.value is not None
    # 174.52 / 222.83 * 100, from the NEW engine's own resolved areas, not the stale legacy 50.0.
    assert abs(vf.value - 78.32) < 0.5
    assert decision.accepted_candidate_id == "derived:coverage_field"


def test_far_is_derived_from_new_engine_resolved_footprint_and_plot_area():
    cv = IndependentCVResult(document_id="p", measurements=[
        IndependentMeasurement(field="building.footprint_area", value_m=174.52, source="NATIVE_TEXT", confidence=0.9),
        IndependentMeasurement(field="plot.area", value_m=222.83, source="NATIVE_TEXT", confidence=0.9),
    ])
    field_decisions = compute_evidence_decision_fields(cv, _empty_vision(), legacy_plan=_plan_with())
    decision, vf = field_decisions["far"]
    assert vf.value is not None
    assert abs(vf.value - (174.52 / 222.83)) < 0.02
    assert decision.accepted_candidate_id == "derived:far_field"


def test_derived_coverage_uses_legacy_fallback_footprint_not_stale_legacy_coverage():
    # independent_cv has plot.area but NOT footprint_area -- footprint_area
    # must come from the legacy fallback (Fix 1's first stage) BEFORE
    # coverage is derived, never from a stale pre-computed legacy coverage
    # number computed against different inputs.
    cv = IndependentCVResult(document_id="p", measurements=[
        IndependentMeasurement(field="plot.area", value_m=222.83, source="NATIVE_TEXT", confidence=0.9),
    ])
    legacy_plan = _plan_with(
        building_footprint_area=_vf(174.52, ConfidenceLevel.HIGH, reason="printed Area Statement"),
        coverage=_vf(12.34, ConfidenceLevel.HIGH, reason="stale/unrelated legacy coverage"),
    )
    field_decisions = compute_evidence_decision_fields(cv, _empty_vision(), legacy_plan=legacy_plan)
    footprint_decision, footprint_vf = field_decisions["building.footprint_area"]
    assert footprint_vf.value == 174.52
    assert footprint_vf.source.startswith("legacy_fallback:")

    coverage_decision, coverage_vf = field_decisions["coverage"]
    assert abs(coverage_vf.value - 78.32) < 0.5
    assert coverage_decision.accepted_candidate_id == "derived:coverage_field"


# --- TEST 8/9/10/11: decision_id provenance ---


def test_accept_value_fields_have_decision_id():
    decision, vf = decide_field(
        "plot.width", cv_value=17.59, vision_value=None, unit="m",
        field_measurements=[], text_index=doc_ev.DocumentTextIndex(pages=[], available=False),
    )
    assert decision.status == DecisionStatus.ACCEPT
    assert vf.decision_id == decision.id


def test_abstain_value_fields_have_decision_id_when_evaluated():
    decision, vf = decide_field(
        "plot.width", cv_value=None, vision_value=None, unit="m",
        field_measurements=[], text_index=doc_ev.DocumentTextIndex(pages=[], available=False),
    )
    assert decision.status == DecisionStatus.ABSTAIN
    assert vf.decision_id == decision.id


def test_conflict_value_fields_have_decision_id_when_evaluated():
    cv_value, vision_value = 17.59, 10.00
    measurement = IndependentMeasurement(field="", value=cv_value, value_m=cv_value, source="DERIVED", confidence=0.8)
    text_index = doc_ev.DocumentTextIndex(pages=[f"{cv_value} {cv_value} {vision_value} {vision_value}"], available=True)
    decision, vf = decide_field(
        "plot.width", cv_value=cv_value, vision_value=vision_value, unit="m",
        field_measurements=[measurement], text_index=text_index,
    )
    assert decision.status == DecisionStatus.CONFLICT
    assert vf.decision_id == decision.id


def test_untouched_field_is_distinguishable_from_an_evaluated_abstain():
    # Genuinely untouched: nothing ever called `decide_field` for this
    # ValueField (e.g. the Phase 8 shadow computation raised and
    # `pipeline.build_normalized_plan` shipped `old_plan` unmodified) --
    # `decision_id` stays None, exactly like every other never-evaluated
    # `ValueField.missing()`/`.conflicting()` call site across the codebase.
    untouched = ValueField[float].missing("no evidence decision ever ran for this field")
    assert untouched.decision_id is None

    # Explicitly evaluated and ABSTAINed: `decide_field` ran, considered
    # both sources, and had nothing to accept -- `decision_id` is set and
    # traceable back to the `Decision` record that produced it.
    decision, evaluated = decide_field(
        "plot.width", cv_value=None, vision_value=None, unit="m",
        field_measurements=[], text_index=doc_ev.DocumentTextIndex(pages=[], available=False),
    )
    assert evaluated.value is None and untouched.value is None  # same observable "no value" state
    assert evaluated.decision_id == decision.id
    assert evaluated.decision_id is not None
    assert evaluated.decision_id != untouched.decision_id
