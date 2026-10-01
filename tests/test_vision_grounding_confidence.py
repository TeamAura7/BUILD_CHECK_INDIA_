"""
Regression tests for the Vision confidence-laundering fix (forensic audit
finding, BUILDCHECK_FORENSIC_AUDIT.md Section 4.4 / 5.1 / 14.1 / 15.1).

Prior behavior: a VLM's own self-reported `confidence` score was mapped
straight into `ConfidenceLevel` (via `confidence_level_from_score`) with no
independent check, in two places:
  - `backend.spatial_reasoning.final_fusion.build_vision_only_plan` (the
    dashboard's "Vision" extraction mode, where CV is never consulted).
  - `backend.spatial_reasoning.final_fusion._value_field`'s Vision-only
    branch -- confirmed BY EXECUTION (not just by reading the audit) to be
    the actual shipping location for every `LENGTH_FIELDS`/`NUMERIC_FIELDS`
    entry (plot/building width & depth, road width, all four setbacks,
    heights) whenever `apply_final_agreement_to_plan` runs, which is
    unconditional for every PDF plan with a resolved plot/building
    candidate. An earlier attempt at this fix inside
    `backend.spatial_reasoning.pipeline`'s own setback-assignment block was
    found, by direct execution, to have zero effect on the shipped plan --
    `apply_final_agreement_to_plan` always overwrites it -- so that
    intermediate attempt was reverted in favor of fixing the actual
    shipping location below.

Fix: `backend.vision_extraction.base.BaseArchitecturalPlanExtractor.
_is_item_grounded` now records, per dimension/area, whether the value was
actually confirmed present in the PDF's native text layer (`grounded=True`),
was checked but had no native text layer to check against at all --  e.g. a
scanned page (`grounded=False`), or was never checked at all, e.g. an older
cached result (`grounded=None`, unchanged/legacy behavior). Both
`build_vision_only_plan` and `_value_field`'s Vision-only branch now cap the
resulting `ConfidenceLevel` to LOW whenever `grounded is False`, regardless
of how high the model's own self-reported score is. `grounded is True`
(genuinely corroborated) or `None` (grounding did not run) are unaffected --
the self-reported score decides the level exactly as before this fix.
"""
from __future__ import annotations

from backend.schemas.enums import ConfidenceLevel
from backend.schemas.vision import VisionArea, VisionDimension, VisionDocumentResult, VisionPageResult
from backend.spatial_reasoning.final_fusion import build_vision_only_plan
from backend.spatial_reasoning.pipeline import build_normalized_plan
from backend.vision_extraction.base import BaseArchitecturalPlanExtractor
from tests.fixtures.geometry_builders import standard_rectangular_plan


# --- 1. The grounding signal itself (backend.vision_extraction.base) -------


def test_is_item_grounded_marks_scanned_page_trivial_pass_as_ungrounded():
    """No native text layer at all (native_spans empty): the item is still
    KEPT (Vision must remain usable on scanned pages), but must be marked
    `grounded=False`, not indistinguishable from a genuine match."""
    item = {"value": 3.0, "evidence": "3.0"}
    kept = BaseArchitecturalPlanExtractor._is_item_grounded(item, [], 100.0, 100.0)
    assert kept is True
    assert item["grounded"] is False


def test_is_item_grounded_marks_genuine_native_text_match_as_grounded():
    item = {"value": 9.14, "evidence": "9.14"}
    kept = BaseArchitecturalPlanExtractor._is_item_grounded(item, [("9.14", (0.0, 0.0))], 100.0, 100.0)
    assert kept is True
    assert item["grounded"] is True


def test_is_item_grounded_still_drops_unmatched_value_on_a_vector_page():
    """Existing hallucination-catch behavior (native text layer exists but
    doesn't contain this value anywhere) must be unchanged by this fix."""
    item = {"value": 3.0, "evidence": "3.0"}
    kept = BaseArchitecturalPlanExtractor._is_item_grounded(item, [("9.14", (0.0, 0.0))], 100.0, 100.0)
    assert kept is False


def test_is_item_grounded_leaves_valueless_items_ungrounded_but_not_flagged():
    item = {"value": None}
    kept = BaseArchitecturalPlanExtractor._is_item_grounded(item, [], 100.0, 100.0)
    assert kept is True
    assert item["grounded"] is None


# --- 2. Vision-only mode (build_vision_only_plan) ---------------------------


def test_vision_only_mode_caps_ungrounded_high_confidence_value_to_low():
    """High self-reported Vision confidence alone must not create
    unjustified HIGH evidence when grounding ran and found nothing to
    confirm the value against."""
    vision = VisionDocumentResult(model_name="t", pages=[VisionPageResult(
        page_number=1,
        dimensions=[VisionDimension(value=17.59, unit="m", type="PLOT_WIDTH", confidence=0.95, grounded=False)],
        areas=[VisionArea(value=200.0, unit="m2", type="PLOT_AREA", confidence=0.95, grounded=False)],
    )])
    plan = build_vision_only_plan(vision, plan_id="p", document_id="d")
    assert plan.plot.width.value == 17.59
    assert plan.plot.width.confidence.level == ConfidenceLevel.LOW
    assert plan.plot.area.value == 200.0
    assert plan.plot.area.confidence.level == ConfidenceLevel.LOW


def test_vision_only_mode_keeps_high_confidence_for_genuinely_grounded_value():
    """Genuine corroboration (found in the PDF's own native text) must
    still be usable at the model's own reported confidence."""
    vision = VisionDocumentResult(model_name="t", pages=[VisionPageResult(
        page_number=1,
        dimensions=[VisionDimension(value=17.59, unit="m", type="PLOT_WIDTH", confidence=0.95, grounded=True)],
    )])
    plan = build_vision_only_plan(vision, plan_id="p", document_id="d")
    assert plan.plot.width.value == 17.59
    assert plan.plot.width.confidence.level == ConfidenceLevel.HIGH


def test_vision_only_mode_unaffected_when_grounding_did_not_run():
    """`grounded=None` (e.g. an older cached vision result predating this
    fix, or ground_against_native_text=False) must behave exactly as
    before this fix: the self-reported score alone decides the level."""
    vision = VisionDocumentResult(model_name="t", pages=[VisionPageResult(
        page_number=1,
        dimensions=[VisionDimension(value=17.59, unit="m", type="PLOT_WIDTH", confidence=0.95)],
    )])
    plan = build_vision_only_plan(vision, plan_id="p", document_id="d")
    assert plan.plot.width.confidence.level == ConfidenceLevel.HIGH


# --- 3. Standard fusion pipeline (build_normalized_plan / apply_final_agreement_to_plan) --


def test_fusion_pipeline_caps_ungrounded_vision_setback_to_low():
    """The same laundering path, reached through the real production
    pipeline (PDFHybridExtractor -> build_normalized_plan), not just the
    Vision-only-mode builder. This is the scenario `pipeline.py`'s own
    `vision_setbacks` merge feeds into `apply_final_agreement_to_plan`,
    which is the function that actually determines the shipped
    confidence level."""
    er = standard_rectangular_plan()
    er.vision_pages = [VisionPageResult(page_number=1, dimensions=[
        VisionDimension(value=5.0, unit="m", type="RIGHT_SETBACK", evidence="5.0", confidence=0.95, grounded=False),
    ])]
    plan = build_normalized_plan(er)
    assert plan.setbacks.right.value == 5.0
    assert plan.setbacks.right.confidence.level == ConfidenceLevel.LOW
    # Existing physical-plausibility conflict flagging must still run
    # (abstention/conflict discipline preserved): a setback this large
    # doesn't fit the plot/building budget on this fixture regardless of
    # confidence, and that must still be surfaced.
    assert plan.setbacks.right.conflict is not None


def test_fusion_pipeline_keeps_high_confidence_for_grounded_vision_setback():
    """Genuine corroboration must still ship at HIGH through the real
    production pipeline, not just in the Vision-only-mode builder."""
    er = standard_rectangular_plan()
    er.vision_pages = [VisionPageResult(page_number=1, dimensions=[
        VisionDimension(value=1.5, unit="m", type="RIGHT_SETBACK", evidence="1.5", confidence=0.95, grounded=True),
    ])]
    plan = build_normalized_plan(er)
    assert plan.setbacks.right.value == 1.5
    assert plan.setbacks.right.confidence.level == ConfidenceLevel.HIGH
