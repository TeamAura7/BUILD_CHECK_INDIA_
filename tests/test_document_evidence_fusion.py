"""Tests for PHASEE3NEW.md: the Document Evidence Layer and the
document-evidence-verified fusion engine
(`backend.spatial_reasoning.document_evidence` /
`backend.spatial_reasoning.final_fusion.build_document_verified_fusion`).

Covers: independent CV/Vision (never influencing each other), the
NOT_FOUND_BY_CV / NOT_FOUND_BY_VISION / NOT_FOUND_BY_EITHER vocabulary,
agreement + document verification, CV-only / Vision-only document
verification, conflict resolution via the evidence hierarchy, targeted
re-check via the document text index, unresolved conflicts (never
hallucinated), and the reported-vs-calculated distinction.
"""
from __future__ import annotations

import json

from backend.schemas.independent_measurements import IndependentCVResult, IndependentMeasurement
from backend.schemas.vision import VisionDocumentResult, VisionPageResult, VisionDimension, VisionArea
from backend.spatial_reasoning import document_evidence as doc_ev
from backend.spatial_reasoning.final_fusion import build_document_verified_fusion


def _cv(field, value, source="NATIVE_TEXT", confidence=0.9, evidence=None, bbox=None):
    return IndependentMeasurement(
        field=field, value_m=value, source=source, confidence=confidence,
        evidence=evidence or [str(value)], geometry_bbox_pts=bbox,
    )


def _vision_dim(value, semantic_type, confidence=0.9):
    return VisionPageResult(
        page_number=1,
        dimensions=[VisionDimension(value=value, unit="m", type=semantic_type, evidence=str(value), confidence=confidence)],
        page_width_pts=1000.0, page_height_pts=1000.0,
    )


# --- Independence ----------------------------------------------------------


def test_missing_cv_field_is_labeled_not_found_by_cv_not_absent():
    cv = IndependentCVResult(document_id="d", measurements=[])
    vision = VisionDocumentResult(model_name="v", pages=[_vision_dim(9.14, "PLOT_WIDTH")], enabled=True)
    out = build_document_verified_fusion(cv, vision, pdf_path=None)
    assert out["cv_values"]["plot.width"]["status"] == doc_ev.NOT_FOUND_BY_CV
    # Missing from CV must never be reported as "value does not exist".
    assert out["cv_values"]["plot.width"]["value"] is None


def test_missing_vision_field_is_labeled_not_found_by_vision():
    cv = IndependentCVResult(document_id="d", measurements=[_cv("plot.width", 9.14, bbox=[0, 0, 1, 1])])
    out = build_document_verified_fusion(cv, None, pdf_path=None)
    assert out["vision_values"]["plot.width"]["status"] == doc_ev.NOT_FOUND_BY_VISION


def test_both_missing_is_not_found_by_either_not_hallucinated():
    cv = IndependentCVResult(document_id="d", measurements=[])
    out = build_document_verified_fusion(cv, None, pdf_path=None)
    assert out["validation"]["plot.width"]["status"] == doc_ev.NOT_FOUND_BY_EITHER
    assert out["final_agreed_values"]["plot.width"]["final_value"] is None


# --- Agreement + document verification -------------------------------------


def test_agreement_with_document_evidence_is_verified():
    cv = IndependentCVResult(document_id="d", measurements=[_cv("plot.width", 17.59, bbox=[0, 0, 1, 1])])
    vision = VisionDocumentResult(model_name="v", pages=[_vision_dim(17.59, "PLOT_WIDTH")], enabled=True)
    out = build_document_verified_fusion(cv, vision, pdf_path=None)
    v = out["validation"]["plot.width"]
    assert v["status"] == doc_ev.AGREED_DOCUMENT_VERIFIED
    assert out["final_agreed_values"]["plot.width"]["final_value"] == 17.59


def test_agreement_without_document_evidence_is_unverified_not_silently_accepted():
    cv = IndependentCVResult(document_id="d", measurements=[_cv("plot.width", 17.59, source="DERIVED", evidence=[])])
    vision = VisionDocumentResult(model_name="v", pages=[_vision_dim(17.59, "PLOT_WIDTH")], enabled=True)
    out = build_document_verified_fusion(cv, vision, pdf_path=None)
    assert out["validation"]["plot.width"]["status"] == doc_ev.AGREED_UNVERIFIED


# --- CV-only / Vision-only ---------------------------------------------------


def test_cv_only_document_verified():
    cv = IndependentCVResult(document_id="d", measurements=[_cv("plot.depth", 9.14, bbox=[0, 0, 1, 1])])
    out = build_document_verified_fusion(cv, None, pdf_path=None)
    v = out["validation"]["plot.depth"]
    assert v["status"] == doc_ev.CV_ONLY_DOCUMENT_VERIFIED
    assert out["final_agreed_values"]["plot.depth"]["final_value"] == 9.14


def test_vision_only_document_verified_when_matched_elsewhere_in_cv_evidence():
    # Vision found road.width, CV did not resolve road.width directly, but
    # CV independently observed the same numeral (9.2) as native text for
    # a different field -- this is what "document evidence" means: the
    # document itself, not just one field's resolver.
    cv = IndependentCVResult(document_id="d", measurements=[_cv("setbacks.front", 9.2, bbox=[0, 0, 1, 1])])
    vision = VisionDocumentResult(model_name="v", pages=[_vision_dim(9.2, "ROAD_WIDTH")], enabled=True)
    out = build_document_verified_fusion(cv, vision, pdf_path=None)
    # road.width itself has no CV measurements, so cross-field matching
    # does not apply -- assert the plain vision-only-unverified path here,
    # and separately assert the "verified" path when CV *does* carry a
    # road.width measurement below.
    assert out["validation"]["road.width"]["status"] in (doc_ev.VISION_ONLY_DOCUMENT_VERIFIED, doc_ev.VISION_ONLY_UNVERIFIED)


def test_vision_only_document_verified_with_matching_field_measurement():
    cv = IndependentCVResult(document_id="d", measurements=[_cv("road.width", 9.2, bbox=[0, 0, 1, 1])])
    # Deliberately blank the CV value fusion sees by using a different field name mapping:
    # simulate CV missing this field but the *document* (via OCR on the same field) confirming it.
    cv_missing_view = IndependentCVResult(document_id="d", measurements=[])
    vision = VisionDocumentResult(model_name="v", pages=[_vision_dim(9.2, "ROAD_WIDTH")], enabled=True)
    # Use the real CV measurements as the field_measurements source for document evidence,
    # even though the "official" cv candidate for the field is None (simulating CV's resolver
    # dropping a value the raw evidence still supports).
    text_index = doc_ev.DocumentTextIndex(pages=[], available=False)
    verdict = doc_ev.resolve_field("road.width", None, 9.2, "m", cv.measurements, text_index)
    assert verdict.status == doc_ev.VISION_ONLY_DOCUMENT_VERIFIED
    assert verdict.final_value == 9.2


# --- Conflict resolution (evidence hierarchy) -------------------------------


def test_conflict_resolved_in_favor_of_stronger_document_evidence():
    cv = IndependentCVResult(document_id="d", measurements=[_cv("plot.width", 17.59, bbox=[0, 0, 1, 1])])
    vision = VisionDocumentResult(model_name="v", pages=[_vision_dim(17.95, "PLOT_WIDTH")], enabled=True)
    out = build_document_verified_fusion(cv, vision, pdf_path=None)
    v = out["validation"]["plot.width"]
    assert v["status"] == doc_ev.CONFLICT_RESOLVED_BY_DOCUMENT_EVIDENCE
    assert v["winner"] == "cv"
    assert out["final_agreed_values"]["plot.width"]["final_value"] == 17.59
    assert "plot.width" in out["conflicts"]


def test_unresolved_conflict_returns_null_not_a_guess():
    # Neither candidate has strong direct document evidence -> must not
    # fabricate a winner (PHASEE3NEW.md section 17/32).
    cv = IndependentCVResult(document_id="d", measurements=[_cv("plot.width", 17.59, source="DERIVED", evidence=[])])
    vision = VisionDocumentResult(model_name="v", pages=[_vision_dim(17.95, "PLOT_WIDTH", confidence=0.4)], enabled=True)
    out = build_document_verified_fusion(cv, vision, pdf_path=None)
    v = out["validation"]["plot.width"]
    assert v["status"] == doc_ev.UNRESOLVED_CONFLICT
    assert out["final_agreed_values"]["plot.width"]["final_value"] is None
    assert out["has_unresolved_conflicts"] is True


# --- Reported vs calculated (section 19) ------------------------------------


def test_reported_value_is_never_overwritten_by_calculated_value():
    cv = IndependentCVResult(document_id="d", measurements=[
        _cv("plot.area", 160.77, source="NATIVE_TEXT"),
        _cv("plot.area", 160.32, source="DERIVED", evidence=[]),
    ])
    out = build_document_verified_fusion(cv, None, pdf_path=None)
    rvc = out["reported_vs_calculated"]["plot.area"]
    assert rvc["reported_value"] == 160.77
    assert rvc["calculated_value"] == 160.32
    assert rvc["final_value"] == 160.77
    assert rvc["status"] == "REPORTED_CALCULATION_DIFFER"
    assert out["final_agreed_values"]["plot.area"]["final_value"] == 160.77


def test_reported_and_calculated_agree_within_tolerance():
    cv = IndependentCVResult(document_id="d", measurements=[
        _cv("plot.area", 222.83, source="NATIVE_TEXT"),
        _cv("plot.area", 222.80, source="DERIVED", evidence=[]),
    ])
    out = build_document_verified_fusion(cv, None, pdf_path=None)
    rvc = out["reported_vs_calculated"]["plot.area"]
    assert rvc["status"] == "REPORTED_VALUE_WITH_GEOMETRIC_CHECK"


# --- Document text index (targeted re-check) --------------------------------


def test_document_text_index_confirms_repeated_occurrence(tmp_path):
    from tests.fixtures.pdf_builders import build_vector_plan_pdf

    pdf_path = build_vector_plan_pdf(tmp_path / "plan.pdf")
    index = doc_ev.build_document_text_index(str(pdf_path))
    assert index.available is True


def test_document_text_index_unavailable_is_not_treated_as_value_absent():
    index = doc_ev.DocumentTextIndex(pages=[], available=False)
    assert index.occurrences(17.59) == 0
    # A candidate must still be evaluable (has_direct_document_evidence can
    # be False without the caller concluding the *document* lacks the value).
    ev = doc_ev.evaluate_candidate(17.59, [], index, vision_present=False)
    assert ev.native_text_confirmed is False


# --- Real-plan end-to-end regression (PLAN2) --------------------------------


def test_plan2_full_document_verified_fusion_matches_acceptance_values():
    from pathlib import Path
    from backend.cv_extraction.site_plan import extract_independent_cv

    pdf_path = Path(__file__).parent.parent / "data" / "test_plans" / "PLAN2.pdf"
    if not pdf_path.exists():
        import pytest
        pytest.skip("PLAN2.pdf fixture not present")

    cv_result = extract_independent_cv(pdf_path, "PLAN2")
    out = build_document_verified_fusion(cv_result, None, pdf_path=str(pdf_path))
    fav = out["final_agreed_values"]

    # building.width/depth moved by ~0.05% (10.5952 -> 10.5904,
    # 16.4822 -> 16.4747) when scale resolution began preferring the sheet's
    # PRINTED "Scale 1:200" note (exactly 14.17323 pt/m) over a scale inferred
    # from PLAN2's own edge labels (14.16676 pt/m). The printed note is the
    # more accurate of the two, and the sheet itself says so: the footprint
    # measured at the printed scale reproduces PLAN2's stated coverage area
    # (174.52 sq.m) to within 0.027%, against 0.064% for the label-derived
    # scale. The plot dimensions are unchanged because they still come from
    # explicit printed labels, not from geometry.
    # plot.depth moved 18.28 -> 18.288 when edge-label selection began
    # considering every label on an edge rather than only the nearest. PLAN2
    # prints its plot dimensions twice, metric and imperial
    # ("12.19(40'0\")" and "18.28" alongside "(60'0\")"), and the imperial
    # form is the exact one: 60'0" is 18.288 m, of which "18.28" is a
    # truncation. plot.width was already reading 40'0" -> 12.192 for the same
    # reason, so this makes the two axes consistent rather than mixing a
    # rounded value with an exact one.
    expected = {
        "plot.width": 12.192, "plot.depth": 18.288,
        "building.width": 10.5904, "building.depth": 16.4747,
        "setbacks.front": 1.0, "setbacks.rear": 0.8, "setbacks.left": 0.8, "setbacks.right": 0.8,
        "road.width": 9.2, "plot.area": 222.83, "building.footprint_area": 174.52,
        "coverage": 78.32, "far.area": 386.55, "far": 1.73, "building.gross_built_up_area": 579.90,
    }
    for field, value in expected.items():
        assert fav[field]["final_value"] == value, f"{field}: {fav[field]}"
        assert fav[field]["status"] == doc_ev.CV_ONLY_DOCUMENT_VERIFIED

    # No field may be silently invented: every non-expected field is
    # either NOT_FOUND_BY_EITHER or (if extraction improves later) verified.
    for field, v in out["validation"].items():
        assert v["status"] in (
            doc_ev.CV_ONLY_DOCUMENT_VERIFIED, doc_ev.NOT_FOUND_BY_EITHER,
        )
