"""
Architecture V2, Phase 8 -- tests for
`backend.spatial_reasoning.evidence_decision_bridge`, the feature-flagged
wiring between `pipeline.py` and `evidence_decision.decide_field`.

These are unit tests against the bridge functions directly (not the full
`build_normalized_plan` pipeline, which is covered by the real-plan
regression suites with the flag at its default `False`); see
ARCHITECTURE_V2.md's Implementation log for the real-plan verification
this phase was checked against.
"""

from __future__ import annotations

from backend.schemas.enums import ConfidenceLevel
from backend.schemas.evidence import Confidence, ValueField
from backend.schemas.independent_measurements import IndependentCVResult, IndependentMeasurement
from backend.schemas.normalized_plan import BuildingSection, NormalizedPlan, PlotSection, RoadSection, SetbackSection
from backend.schemas.vision import VisionDocumentResult
from backend.spatial_reasoning.evidence_decision_bridge import (
    apply_evidence_decision_fields_to_plan,
    compute_evidence_decision_fields,
    log_comparison,
)


def _empty_vision() -> VisionDocumentResult:
    return VisionDocumentResult(model_name="disabled", pages=[], enabled=False)


def _minimal_plan() -> NormalizedPlan:
    missing = lambda: ValueField[float].missing("test fixture")  # noqa: E731
    return NormalizedPlan(
        plan_id="p", source_document_id="doc",
        plot=PlotSection(width=missing(), depth=missing(), area=missing()),
        building=BuildingSection(width=missing(), depth=missing(), footprint_area=missing()),
        road=RoadSection(width=missing()),
        setbacks=SetbackSection(front=missing(), rear=missing(), left=missing(), right=missing()),
        coverage=missing(), far=missing(),
    )


def test_compute_evidence_decision_fields_covers_length_and_numeric_fields():
    cv = IndependentCVResult(document_id="p", measurements=[
        IndependentMeasurement(field="plot.width", value_m=12.19, source="NATIVE_TEXT", confidence=0.97),
    ])
    field_decisions = compute_evidence_decision_fields(cv, _empty_vision())
    assert "plot.width" in field_decisions
    assert "plot.area" in field_decisions  # NUMERIC_FIELDS entry, present even with no evidence
    decision, value_field = field_decisions["plot.width"]
    assert value_field.value == 12.19
    assert decision.field_name == "plot.width"


def test_compute_evidence_decision_fields_handles_no_cv_no_vision():
    field_decisions = compute_evidence_decision_fields(None, None)
    decision, value_field = field_decisions["plot.width"]
    assert value_field.value is None
    assert value_field.status == ConfidenceLevel.MISSING


def test_apply_evidence_decision_fields_to_plan_overwrites_only_shippable_fields():
    plan = _minimal_plan()
    cv = IndependentCVResult(document_id="p", measurements=[
        IndependentMeasurement(field="plot.width", value_m=12.19, source="NATIVE_TEXT", confidence=0.97),
        IndependentMeasurement(field="building.width", value_m=10.59, source="NATIVE_TEXT", confidence=0.95),
    ])
    field_decisions = compute_evidence_decision_fields(cv, _empty_vision())
    updated = apply_evidence_decision_fields_to_plan(plan, field_decisions)

    assert updated.plot.width.value == 12.19
    assert updated.plot.width.decision_id is not None
    assert updated.building.width.value == 10.59
    # Fields with no evidence stay MISSING, not silently fabricated.
    assert updated.road.width.status == ConfidenceLevel.MISSING


def test_apply_evidence_decision_fields_to_plan_is_the_same_object_mutated():
    plan = _minimal_plan()
    field_decisions = compute_evidence_decision_fields(None, None)
    updated = apply_evidence_decision_fields_to_plan(plan, field_decisions)
    assert updated is plan


def test_log_comparison_does_not_raise_on_agreeing_or_disagreeing_fields():
    plan = _minimal_plan()
    plan.plot.width = ValueField[float](value=12.19, confidence=Confidence(level=ConfidenceLevel.HIGH, score=0.9))
    cv = IndependentCVResult(document_id="p", measurements=[
        IndependentMeasurement(field="plot.width", value_m=99.0, source="NATIVE_TEXT", confidence=0.97),
    ])
    field_decisions = compute_evidence_decision_fields(cv, _empty_vision())
    log_comparison(field_decisions, plan)  # must not raise


def test_dxf_early_return_path_also_computes_and_logs_phase8_comparison(monkeypatch):
    """Architecture V2, Phase DXF-0: `build_normalized_plan`'s DXF-relevant
    early-return branch (`winner is None`, always true for DXF since it
    never populates the legacy `plot_candidates` pool) used to `return plan`
    directly, bypassing the Phase 8 bridge entirely -- confirmed by tracing
    the control flow directly: `use_evidence_decision_engine=True` had ZERO
    effect on any DXF plan before this fix. Both flag states are checked
    against the real `build_normalized_plan` entrypoint (not the bridge
    functions in isolation, unlike this file's other tests), since the bug
    was specifically in whether the DXF branch REACHES the bridge at all.
    """
    import backend.spatial_reasoning.pipeline as pipeline_mod
    from backend.config import get_settings
    from backend.schemas.extraction import DocumentType, ExtractionResult
    from backend.spatial_reasoning.pipeline import build_normalized_plan

    def _dxf_extraction() -> ExtractionResult:
        return ExtractionResult(
            document_id="dxf-phase8", document_type=DocumentType.DXF, page_count=1,
            plot_candidates=[], building_candidates=[], road_candidates=[],
            independent_cv=IndependentCVResult(document_id="dxf-phase8", measurements=[
                IndependentMeasurement(field="plot.width", value_m=12.19, source="NATIVE_TEXT", confidence=0.97),
            ]),
        )

    # Flag at its default (False): the bridge must still be REACHED (proven
    # via a spy on the shared helper), but the shipped plan is unaffected --
    # no decision_id, same value as the old fv()-only path.
    calls: list[bool] = []
    real_bridge = pipeline_mod._apply_phase8_bridge

    def _spy(plan, extraction, vision_doc):
        calls.append(True)
        return real_bridge(plan, extraction, vision_doc)

    monkeypatch.setattr(pipeline_mod, "_apply_phase8_bridge", _spy)
    plan = build_normalized_plan(_dxf_extraction())
    assert calls, "DXF's early-return branch must call the shared Phase 8 bridge, not bypass it"
    assert plan.plot.width.value == 12.19
    assert plan.plot.width.decision_id is None

    # Flag on: the DXF plan's shippable fields must now carry a decision_id
    # from evidence_decision.decide_field, via the exact same bridge.
    base_settings = get_settings()
    monkeypatch.setattr(
        pipeline_mod, "get_settings",
        lambda: base_settings.model_copy(update={"use_evidence_decision_engine": True}),
    )
    plan_with_flag = build_normalized_plan(_dxf_extraction())
    assert plan_with_flag.plot.width.value == 12.19
    assert plan_with_flag.plot.width.decision_id is not None
