import pytest

from backend.schemas.enums import ConfidenceLevel
from backend.schemas.evidence import Confidence, ValueField
from backend.schemas.independent_measurements import IndependentCVResult, IndependentMeasurement
from backend.schemas.normalized_plan import BuildingSection, NormalizedPlan, PlotSection, RoadSection, SetbackSection
from backend.schemas.vision import VisionDocumentResult, VisionPageResult, VisionDimension, VisionArea
from backend.spatial_reasoning.final_fusion import apply_final_agreement_to_plan, build_final_agreement, build_vision_only_plan


def test_final_fusion_agrees_on_independent_cv_and_vision():
    cv = IndependentCVResult(document_id='p', pages_analyzed=[1], measurements=[
        IndependentMeasurement(field='plot.width', value_m=12.19, source='NATIVE_TEXT', confidence=.99),
        IndependentMeasurement(field='plot.area', value=222.83, unit='m2', source='NATIVE_TEXT', confidence=.99),
    ])
    vision = VisionDocumentResult(model_name='test', pages=[VisionPageResult(page_number=1,
        dimensions=[VisionDimension(value=12.20, unit='m', type='PLOT_WIDTH', confidence=.95)],
        areas=[VisionArea(value=222.84, unit='m2', type='PLOT_AREA', confidence=.95)])])
    out = build_final_agreement(cv, vision)
    assert out['values']['plot.width']['status'] == 'AGREED'
    assert out['values']['plot.area']['status'] == 'AGREED'


def test_final_fusion_does_not_pick_a_winner_on_conflict():
    cv = IndependentCVResult(document_id='p', measurements=[
        IndependentMeasurement(field='plot.width', value_m=12.19, source='NATIVE_TEXT', confidence=.99)])
    vision = VisionDocumentResult(model_name='test', pages=[VisionPageResult(page_number=1,
        dimensions=[VisionDimension(value=10.0, unit='m', type='PLOT_WIDTH', confidence=.99)])])
    out = build_final_agreement(cv, vision)
    assert out['values']['plot.width']['status'] == 'CONFLICT'
    assert out['values']['plot.width']['value'] is None


def test_vision_only_plan_reports_low_not_medium_for_unset_confidence():
    """Same confidence-gate bug as build_normalized_plan's Vision branch,
    but in the Vision-only-mode plan builder: a VisionDimension/VisionArea
    with confidence never populated (schema default 0.0) must report LOW,
    not MEDIUM -- otherwise it silently bypasses the compliance engine's
    LOW -> REQUIRES_REVIEW safety gate."""
    vision = VisionDocumentResult(model_name='test', pages=[VisionPageResult(
        page_number=1,
        dimensions=[VisionDimension(value=12.0, unit='m', type='PLOT_WIDTH')],  # confidence omitted -> 0.0
        areas=[VisionArea(value=200.0, unit='m2', type='PLOT_AREA')],  # confidence omitted -> 0.0
    )])
    plan = build_vision_only_plan(vision, plan_id='p', document_id='d')
    assert plan.plot.width.value == 12.0
    assert plan.plot.width.confidence.level == ConfidenceLevel.LOW
    assert plan.plot.area.value == 200.0
    assert plan.plot.area.confidence.level == ConfidenceLevel.LOW


def test_build_final_agreement_cv_only_reports_low_for_low_confidence_measurement():
    """The same confidence-discard bug as the Vision-only test above, but
    in `_value_field`'s CV-only branch (`build_final_agreement`) -- this
    one was found live on real DXF extractions this session: a DXF
    measurement the extractor itself capped at 0.4 (e.g.
    `_CAPTION_OVERRIDE_CONFIDENCE_CAP`, DXF_FAILURE_TAXONOMY.md items 8/9)
    was reaching this function tagged ConfidenceLevel.MEDIUM unconditionally
    -- identical to a fully-trusted reading, and silently invisible to the
    compliance engine's LOW -> REQUIRES_REVIEW safety gate. No Vision
    result is supplied here (empty pages), matching the DXF pipeline's own
    typical fusion inputs (Vision usually disabled/unavailable)."""
    cv = IndependentCVResult(document_id="p", measurements=[
        IndependentMeasurement(field="plot.width", value_m=12.78, source="VECTOR_GEOMETRY_RECONSTRUCTED", confidence=0.4),
    ])
    vision = VisionDocumentResult(model_name="test", pages=[])
    out = build_final_agreement(cv, vision)
    assert out["values"]["plot.width"]["status"] == "CV_ONLY"
    assert out["values"]["plot.width"]["confidence"] == ConfidenceLevel.LOW.value


def test_build_final_agreement_cv_only_reports_high_for_high_confidence_measurement():
    """The flip side of the test above: a genuinely trustworthy CV-only
    reading must NOT be dragged down to LOW just because this is a
    single-source value -- confidence should track the SOURCE
    measurement's own score, not collapse to one fixed level regardless."""
    cv = IndependentCVResult(document_id="p", measurements=[
        IndependentMeasurement(field="plot.width", value_m=12.0, source="NATIVE_TEXT", confidence=0.97),
    ])
    vision = VisionDocumentResult(model_name="test", pages=[])
    out = build_final_agreement(cv, vision)
    assert out["values"]["plot.width"]["confidence"] == ConfidenceLevel.HIGH.value


def _minimal_plan(building_width_value: float) -> NormalizedPlan:
    """A plan whose building.width already carries a value from the
    'first pass' (pipeline.py's legacy candidate resolver / vision-semantic-
    frame reconstruction) -- the scenario `apply_final_agreement_to_plan`
    must not just silently trust when the independent CV+Vision layer
    later finds a conflict for the same field."""
    missing = lambda: ValueField[float].missing("test fixture")
    return NormalizedPlan(
        plan_id="p", source_document_id="doc",
        plot=PlotSection(width=missing(), depth=missing(), area=missing()),
        building=BuildingSection(
            width=ValueField[float](value=building_width_value, confidence=Confidence(level=ConfidenceLevel.MEDIUM, reason="first pass")),
            depth=missing(), footprint_area=missing(),
        ),
        road=RoadSection(width=missing()),
        setbacks=SetbackSection(front=missing(), rear=missing(), left=missing(), right=missing()),
        coverage=missing(), far=missing(),
    )


def test_apply_final_agreement_surfaces_conflict_instead_of_keeping_stale_first_pass_value():
    """
    Pins a real bug found live on PLAN5.pdf's Fusion mode: CV independently
    measured building.width=13.09 (matching the sheet's own printed
    dimension and hand-verified ground truth), but the dashboard showed
    6.2172 -- traced to `pipeline.py`'s first pass unconditionally trusting
    a Vision BUILDING_WIDTH reading that was real and grounded on the page
    (a first-floor room's width, printed and legitimate) but for the WRONG
    semantic region, with no cross-check against CV at all. The final
    agreement layer here DID correctly detect CV vs Vision disagreement
    (status=CONFLICT) but then silently kept that already-wrong first-pass
    fallback instead of surfacing the conflict -- exactly the bug this
    test pins.
    """
    plan = _minimal_plan(building_width_value=6.2172)  # the wrong first-pass value
    cv = IndependentCVResult(document_id="p", measurements=[
        IndependentMeasurement(field="building.width", value_m=13.09, source="VECTOR_GEOMETRY", confidence=0.95),
    ])
    vision = VisionDocumentResult(model_name="test", pages=[VisionPageResult(
        page_number=1,
        dimensions=[VisionDimension(value=6.59, unit="m", type="BUILDING_WIDTH", confidence=0.95)],
    )])

    result = apply_final_agreement_to_plan(plan, cv, vision)

    assert result.building.width.value is None
    assert result.building.width.status == ConfidenceLevel.CONFLICTING
    assert result.building.width.value != 6.2172  # the stale first-pass value must not survive
    assert result.building.width.conflict is not None
    assert len(result.building.width.conflict.conflicting_raw_values) == 2
    assert result.building.width.conflict in result.conflicts


def test_apply_final_agreement_ships_low_confidence_for_low_confidence_cv_only_value():
    """Pins the second, longer-lived instance of the same confidence-discard
    bug `test_build_final_agreement_cv_only_reports_low_for_low_confidence_
    measurement` already pins in `build_final_agreement`/`_value_field`.

    That earlier fix corrected `item["confidence"]` inside the fusion
    dict -- but `apply_final_agreement_to_plan`'s own `vf()` closure, the
    function that actually builds the `ValueField` attached to
    `NormalizedPlan` (what a real API response/compliance check actually
    sees), independently reconstructed
    `ConfidenceLevel.HIGH if item["status"] == "AGREED" else ConfidenceLevel.MEDIUM`
    from `status` alone, discarding the already-correct `item["confidence"]`
    a second time. This is the code path every PDF plan with a resolved
    plot/building candidate takes (DXF-only plans have empty candidate
    lists and go through `pipeline.py`'s own, already-fixed `fv()` closure
    instead) -- confirmed live on every real PDF plan tested via
    `backend/tools/run_evidence_decision_shadow.py`'s shadow-mode
    comparison (see ARCHITECTURE_V2.md's Implementation log) before this
    fix: every scored field on PLAN2/PLAN4/PLAN5/PLAN6 shipped
    ConfidenceLevel.MEDIUM regardless of whether the underlying evidence
    was strong or weak.
    """
    plan = _minimal_plan(building_width_value=999.0)
    cv = IndependentCVResult(document_id="p", measurements=[
        IndependentMeasurement(field="building.width", value_m=12.78, source="VECTOR_GEOMETRY_RECONSTRUCTED", confidence=0.4),
    ])
    vision = VisionDocumentResult(model_name="test", pages=[])
    result = apply_final_agreement_to_plan(plan, cv, vision)
    assert result.building.width.value == pytest.approx(12.78, abs=0.01)
    assert result.building.width.status == ConfidenceLevel.LOW


def test_apply_final_agreement_area_derived_from_low_confidence_dimensions_stays_low():
    """Pins the PLAN7-class bug found via a live UI run: `_reconcile_area_
    field` used to stamp a freshly-computed width*depth `plot.area`/
    `building.footprint_area` at ConfidenceLevel.HIGH unconditionally, even
    when plot.width/plot.depth themselves were correctly resolved at LOW
    (e.g. a spurious geometric candidate on a photograph with no real site
    plan) -- silently discarding that LOW a second time, the same
    "confidence computed correctly, then overwritten" bug class already
    fixed elsewhere in this module. Both plot.width and plot.depth here are
    LOW-confidence CV-only measurements; plot.area must inherit LOW, not
    HIGH, and coverage/far (derived from it) must not launder that back up
    either."""
    plan = _minimal_plan(building_width_value=999.0)
    cv = IndependentCVResult(document_id="p", measurements=[
        IndependentMeasurement(field="plot.width", value_m=19.37, source="VECTOR_GEOMETRY_RECONSTRUCTED", confidence=0.4),
        IndependentMeasurement(field="plot.depth", value_m=4.40, source="VECTOR_GEOMETRY_RECONSTRUCTED", confidence=0.4),
        IndependentMeasurement(field="building.width", value_m=5.82, source="VECTOR_GEOMETRY_RECONSTRUCTED", confidence=0.4),
        IndependentMeasurement(field="building.depth", value_m=13.83, source="VECTOR_GEOMETRY_RECONSTRUCTED", confidence=0.4),
    ])
    vision = VisionDocumentResult(model_name="test", pages=[])

    result = apply_final_agreement_to_plan(plan, cv, vision)

    assert result.plot.width.status == ConfidenceLevel.LOW
    assert result.plot.depth.status == ConfidenceLevel.LOW
    assert result.plot.area.value == pytest.approx(19.37 * 4.40, abs=0.01)
    assert result.plot.area.status == ConfidenceLevel.LOW
    assert result.building.footprint_area.status == ConfidenceLevel.LOW
    assert result.coverage.status == ConfidenceLevel.LOW
    assert result.far.status == ConfidenceLevel.LOW


def test_apply_final_agreement_ships_high_confidence_for_high_confidence_cv_only_value():
    """The flip side: a genuinely trustworthy CV-only reading must not be
    dragged down to a blanket MEDIUM just because it is single-source."""
    plan = _minimal_plan(building_width_value=999.0)
    cv = IndependentCVResult(document_id="p", measurements=[
        IndependentMeasurement(field="building.width", value_m=13.09, source="NATIVE_TEXT", confidence=0.97),
    ])
    vision = VisionDocumentResult(model_name="test", pages=[])
    result = apply_final_agreement_to_plan(plan, cv, vision)
    assert result.building.width.value == pytest.approx(13.09, abs=0.01)
    assert result.building.width.status == ConfidenceLevel.HIGH


def test_apply_final_agreement_still_uses_agreed_value_when_sources_match():
    plan = _minimal_plan(building_width_value=999.0)  # first pass would be irrelevant here
    cv = IndependentCVResult(document_id="p", measurements=[
        IndependentMeasurement(field="building.width", value_m=13.09, source="VECTOR_GEOMETRY", confidence=0.95),
    ])
    vision = VisionDocumentResult(model_name="test", pages=[VisionPageResult(
        page_number=1,
        dimensions=[VisionDimension(value=13.10, unit="m", type="BUILDING_WIDTH", confidence=0.95)],
    )])
    result = apply_final_agreement_to_plan(plan, cv, vision)
    assert result.building.width.value == pytest.approx(13.095, abs=0.01)
    assert result.building.width.status == ConfidenceLevel.HIGH


def test_apply_final_agreement_runs_finalization_when_cv_is_none():
    """Regression for the confirmed fusion-bypass bug.

    `apply_final_agreement_to_plan` used to contain `if cv is None: return
    plan`, silently shipping the pre-fusion plan untouched -- no area/
    coverage/FAR recomputation, no footprint-vs-plot sanity check, and no
    visible indication that independent CV validation never actually ran.
    Finalization must still execute: a Vision-supplied value for a field
    the pre-fusion plan didn't have must still be picked up, and the plan
    must record that CV validation did not happen.
    """
    plan = _minimal_plan(building_width_value=999.0)
    vision = VisionDocumentResult(model_name="test", pages=[VisionPageResult(
        page_number=1,
        dimensions=[VisionDimension(value=8.5, unit="m", type="ROAD_WIDTH", confidence=0.9)],
    )])

    result = apply_final_agreement_to_plan(plan, None, vision)

    # Finalization actually ran (picked up a field the pre-fusion plan
    # never had a value for at all), not just "returned plan unchanged".
    assert result.road.width.value == pytest.approx(8.5)
    assert result.metadata.get("cv_validation_status") == "NOT_RUN"
    assert "not attempted" in (result.overall_confidence_note or "").lower()
    # A pre-fusion value Vision does not compete with must still survive --
    # CV being unavailable must not blank out otherwise-good values.
    assert result.building.width.value == pytest.approx(999.0)


def test_apply_final_agreement_runs_finalization_when_cv_extraction_failed():
    """Same as above, but for the 'CV was attempted and raised' case.

    Before the fix, `backend/cv_extraction/pdf_extractor.py` left
    `ExtractionResult.independent_cv` as bare `None` on any exception from
    `extract_independent_cv`, which was indistinguishable from "never
    attempted" and hit the exact same silent-bypass bug. It must now be an
    explicit `IndependentCVResult(status="FAILED", error=...)`, which this
    function must reconcile through (not bypass) while surfacing the
    failure.
    """
    plan = _minimal_plan(building_width_value=999.0)
    cv = IndependentCVResult(document_id="p", status="FAILED", error="OpenCV contour detection raised ValueError")
    vision = VisionDocumentResult(model_name="test", pages=[VisionPageResult(
        page_number=1,
        dimensions=[VisionDimension(value=8.5, unit="m", type="ROAD_WIDTH", confidence=0.9)],
    )])

    result = apply_final_agreement_to_plan(plan, cv, vision)

    assert result.road.width.value == pytest.approx(8.5)
    assert result.metadata.get("cv_validation_status") == "FAILED"
    assert "ValueError" in result.metadata.get("cv_validation_error", "")
    assert "failed" in (result.overall_confidence_note or "").lower()
    assert result.building.width.value == pytest.approx(999.0)
