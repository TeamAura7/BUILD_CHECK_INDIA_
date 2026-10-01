"""
Architecture V2, Phase 7 -- tests for
`backend.spatial_reasoning.evidence_decision.decide_field`, the shadow-mode
evidence decision engine built on `document_evidence.py`'s existing
scoring/threshold design.

These exercise every branch of the DecisionStatus mapping described in
`decide_field`'s own docstring: ABSTAIN (no candidates, and weak-evidence
disagreement), ACCEPT (agreed-verified, agreed-unverified, cv-only,
vision-only, conflict-resolved-by-document-evidence), and CONFLICT (a
genuine tie between two well-evidenced, disagreeing candidates).
"""

from __future__ import annotations

from backend.schemas.enums import ConfidenceLevel, DecisionStatus
from backend.schemas.independent_measurements import IndependentMeasurement
from backend.spatial_reasoning import document_evidence as doc_ev
from backend.spatial_reasoning.evidence_decision import decide_field


def _measurement(value: float, source: str = "DERIVED", **overrides) -> IndependentMeasurement:
    defaults = dict(field="", value=value, value_m=value, source=source, confidence=0.8)
    defaults.update(overrides)
    return IndependentMeasurement(**defaults)


def _unavailable_index() -> doc_ev.DocumentTextIndex:
    return doc_ev.DocumentTextIndex(pages=[], available=False)


# --- ABSTAIN -----------------------------------------------------------------


def test_neither_cv_nor_vision_abstains():
    decision, value_field = decide_field(
        "plot.width", cv_value=None, vision_value=None, unit="m",
        field_measurements=[], text_index=_unavailable_index(),
    )
    assert decision.status == DecisionStatus.ABSTAIN
    assert decision.accepted_candidate_id is None
    assert value_field.value is None
    assert value_field.status == ConfidenceLevel.MISSING


def test_weak_disagreement_with_no_supporting_evidence_abstains_not_conflicts():
    # Both candidates present and disagreeing, but neither is well-evidenced
    # -- insufficient evidence to decide, not a confirmed disagreement.
    decision, value_field = decide_field(
        "road.width", cv_value=9.0, vision_value=10.0, unit="m",
        field_measurements=[], text_index=_unavailable_index(),
    )
    assert decision.status == DecisionStatus.ABSTAIN
    assert value_field.status == ConfidenceLevel.MISSING


# --- ACCEPT --------------------------------------------------------------------


def test_agreed_and_document_verified_accepts_with_real_confidence():
    measurement = _measurement(17.59, source="NATIVE_TEXT", field="plot.width", geometry_bbox_pts=[0, 0, 1, 1])
    decision, value_field = decide_field(
        "plot.width", cv_value=17.59, vision_value=17.59, unit="m",
        field_measurements=[measurement], text_index=_unavailable_index(),
    )
    assert decision.status == DecisionStatus.ACCEPT
    assert value_field.value == 17.59
    assert value_field.status in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM)
    assert value_field.confidence.score is not None
    assert value_field.decision_id == decision.id


def test_agreed_but_unverified_accepts_at_lower_confidence_than_verified():
    verified_measurement = _measurement(17.59, source="NATIVE_TEXT", field="plot.width", geometry_bbox_pts=[0, 0, 1, 1])
    _verified_decision, verified_vf = decide_field(
        "plot.width", cv_value=17.59, vision_value=17.59, unit="m",
        field_measurements=[verified_measurement], text_index=_unavailable_index(),
    )
    unverified_decision, unverified_vf = decide_field(
        "plot.width", cv_value=17.59, vision_value=17.59, unit="m",
        field_measurements=[], text_index=_unavailable_index(),
    )
    assert unverified_decision.status == DecisionStatus.ACCEPT
    assert unverified_vf.confidence.score < verified_vf.confidence.score


def test_cv_only_unverified_accepts_at_low_confidence_never_medium_with_no_score():
    decision, value_field = decide_field(
        "building.width", cv_value=13.09, vision_value=None, unit="m",
        field_measurements=[], text_index=_unavailable_index(),
    )
    assert decision.status == DecisionStatus.ACCEPT
    assert value_field.value == 13.09
    # No document evidence at all -- must not silently read as MEDIUM.
    assert value_field.status == ConfidenceLevel.LOW
    assert value_field.confidence.score is not None


def test_cv_only_document_verified_accepts():
    measurement = _measurement(13.09, source="NATIVE_TEXT", field="building.width", geometry_bbox_pts=[0, 0, 1, 1])
    decision, value_field = decide_field(
        "building.width", cv_value=13.09, vision_value=None, unit="m",
        field_measurements=[measurement], text_index=_unavailable_index(),
    )
    assert decision.status == DecisionStatus.ACCEPT
    assert value_field.value == 13.09


def test_vision_only_document_verified_accepts():
    measurement = _measurement(9.14, source="NATIVE_TEXT", field="plot.depth", geometry_bbox_pts=[0, 0, 1, 1])
    decision, value_field = decide_field(
        "plot.depth", cv_value=None, vision_value=9.14, unit="m",
        field_measurements=[measurement], text_index=_unavailable_index(),
    )
    assert decision.status == DecisionStatus.ACCEPT
    assert value_field.value == 9.14


def test_conflict_resolved_by_document_evidence_accepts_the_winner_and_records_the_loser():
    # CV's value is well-evidenced; Vision's value has no supporting evidence.
    measurement = _measurement(17.59, source="NATIVE_TEXT", field="plot.width", geometry_bbox_pts=[0, 0, 1, 1])
    decision, value_field = decide_field(
        "plot.width", cv_value=17.59, vision_value=15.35, unit="m",
        field_measurements=[measurement], text_index=_unavailable_index(),
    )
    assert decision.status == DecisionStatus.ACCEPT
    assert decision.accepted_candidate_id == "cv"
    assert value_field.value == 17.59
    assert len(decision.rejected) == 1
    assert "vision" in decision.rejected[0].detail


# --- CONFLICT ------------------------------------------------------------------


def test_genuine_tie_between_two_well_evidenced_candidates_conflicts():
    # Neither value has its own confirming measurement beyond text-index
    # occurrence, engineered to score identically on each side (see module
    # docstring's arithmetic walkthrough in the PR/commit, or re-derive:
    # cv_ev = native_text(+5, via occurrence) + repeated(+2) + cv_present(+1) = 8
    # vision_ev = native_text(+5, via occurrence) + repeated(+2) + vision_present(+1) = 8
    cv_value, vision_value = 17.59, 10.00
    measurement = _measurement(cv_value, source="DERIVED")  # matches cv_value only
    text_index = doc_ev.DocumentTextIndex(pages=[f"{cv_value} {cv_value} {vision_value} {vision_value}"], available=True)

    decision, value_field = decide_field(
        "plot.width", cv_value=cv_value, vision_value=vision_value, unit="m",
        field_measurements=[measurement], text_index=text_index,
    )
    assert decision.status == DecisionStatus.CONFLICT
    assert value_field.value is None
    assert value_field.status == ConfidenceLevel.CONFLICTING
    assert value_field.conflict is not None
    assert set(value_field.conflict.conflicting_candidate_ids) == {"cv", "vision"}
    assert len(value_field.conflict.conflicting_raw_values) == 2


# --- Cross-cutting ---------------------------------------------------------------


def test_every_decision_has_a_unique_id():
    d1, _ = decide_field("plot.width", 17.59, None, "m", [], _unavailable_index())
    d2, _ = decide_field("plot.width", 17.59, None, "m", [], _unavailable_index())
    assert d1.id != d2.id


def test_decision_never_produces_reject_status_for_this_cv_vision_axis():
    # document_evidence.py's model only arbitrates between existing CV/
    # Vision candidates -- it never rejects a whole hypothesis outright
    # (that is StructuralHypothesis/constraint-scoring territory, not yet
    # wired into this function). Exercise every branch and confirm none
    # produces REJECT.
    cases = [
        (None, None, [], _unavailable_index()),
        (17.59, 17.59, [], _unavailable_index()),
        (13.09, None, [], _unavailable_index()),
        (None, 9.14, [], _unavailable_index()),
        (9.0, 10.0, [], _unavailable_index()),
    ]
    for cv, vision, measurements, index in cases:
        decision, _vf = decide_field("field", cv, vision, "m", measurements, index)
        assert decision.status != DecisionStatus.REJECT
