"""
Architecture V2, Phase 2 — schema tests for the new, purely additive
Drawing Evidence Graph / StructuralHypothesis / Decision contracts, and for
the `confidence_from_source` guard that is meant to make the recurring
"MEDIUM with no score" bug (ARCHITECTURE_V2.md Deliverable B.5/B.9/D.5)
structurally impossible to reintroduce.

These are pure schema/round-trip tests — nothing here wires the new types
into `ExtractionResult`/`NormalizedPlan` yet; that is a later phase.
"""

from __future__ import annotations

from backend.schemas.decision import Decision
from backend.schemas.enums import (
    ConfidenceLevel,
    DecisionStatus,
    EntityKind,
    EvidenceKind,
    HypothesisIdentity,
    RelationType,
    cap_confidence_level,
    confidence_from_source,
)
from backend.schemas.evidence import Confidence, Conflict, ValueField
from backend.schemas.evidence_graph import DrawingNode, DrawingRelationship, EvidenceCandidate
from backend.schemas.geometry import Point, Polygon
from backend.schemas.hypothesis import ConstraintResult, StructuralHypothesis
from backend.schemas.units import UnitValue


# --- confidence_from_source: the structural fix for the recurring bug ------


def test_confidence_from_source_with_no_score_is_low_not_medium():
    conf = confidence_from_source(None, reason="vision measurement missing")
    assert conf.level == ConfidenceLevel.LOW
    assert conf.score is None
    assert "no source score" in conf.reason


def test_confidence_from_source_with_a_real_score_derives_the_level():
    high = confidence_from_source(0.95, reason="native DXF DIMENSION entity")
    assert high.level == ConfidenceLevel.HIGH
    assert high.score == 0.95

    low = confidence_from_source(0.4, reason="caption override cap")
    assert low.level == ConfidenceLevel.LOW
    assert low.score == 0.4


def test_confidence_from_source_never_returns_medium_for_a_missing_score():
    # Guards the exact bug class: a None/unset score must never surface as
    # MEDIUM, which the compliance engine treats as "trustworthy enough to
    # not require review."
    for reason in ["", "some reason", "another"]:
        conf = confidence_from_source(None, reason=reason)
        assert conf.level != ConfidenceLevel.MEDIUM


# --- cap_confidence_level: fixes "a correctly-computed low judgment never propagates" ---


def test_cap_confidence_level_lowers_a_level_that_exceeds_the_cap():
    assert cap_confidence_level(ConfidenceLevel.HIGH, ConfidenceLevel.LOW) == ConfidenceLevel.LOW
    assert cap_confidence_level(ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM) == ConfidenceLevel.MEDIUM


def test_cap_confidence_level_never_raises_a_level():
    assert cap_confidence_level(ConfidenceLevel.LOW, ConfidenceLevel.HIGH) == ConfidenceLevel.LOW
    assert cap_confidence_level(ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH) == ConfidenceLevel.MEDIUM


def test_cap_confidence_level_passes_through_conflicting_and_missing_unchanged():
    assert cap_confidence_level(ConfidenceLevel.CONFLICTING, ConfidenceLevel.LOW) == ConfidenceLevel.CONFLICTING
    assert cap_confidence_level(ConfidenceLevel.MISSING, ConfidenceLevel.LOW) == ConfidenceLevel.MISSING


# --- EntityKind.SHEET_FRAME --------------------------------------------------


def test_entity_kind_has_sheet_frame_distinct_from_unknown():
    assert EntityKind.SHEET_FRAME != EntityKind.UNKNOWN
    assert EntityKind.SHEET_FRAME.value == "SHEET_FRAME"


def test_hypothesis_identity_maps_onto_entity_kind():
    assert HypothesisIdentity.PLOT_BOUNDARY.entity_kind == EntityKind.PLOT
    assert HypothesisIdentity.BUILDING.entity_kind == EntityKind.BUILDING
    assert HypothesisIdentity.ROAD.entity_kind == EntityKind.ROAD
    assert HypothesisIdentity.SHEET_FRAME.entity_kind == EntityKind.SHEET_FRAME
    assert HypothesisIdentity.OTHER.entity_kind == EntityKind.UNKNOWN


# --- EvidenceCandidate / DrawingNode / DrawingRelationship ------------------


def test_evidence_candidate_round_trip():
    cand = EvidenceCandidate(
        id="ec-1",
        kind=EvidenceKind.NATIVE_TEXT,
        entity_kind_hint=EntityKind.PLOT,
        raw_value=UnitValue(magnitude=17.59, unit="m"),
        text="17.59",
        source_confidence=0.6,
        source_module="dxf_text_recovery.recover_dimension_and_caption_evidence",
        provenance_note="native TEXT entity near top boundary",
    )
    restored = EvidenceCandidate.model_validate(cand.model_dump())
    assert restored == cand
    assert restored.kind == EvidenceKind.NATIVE_TEXT


def test_evidence_candidate_requires_a_real_source_confidence():
    # source_confidence has no default -- a caller must supply a number,
    # never silently rely on a schema default standing in for "unknown".
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EvidenceCandidate(
            id="ec-2",
            kind=EvidenceKind.OCR_TEXT,
            text="10m WIDE ROAD",
            source_module="dxf_text_recovery",
            provenance_note="missing confidence on purpose",
        )


def test_drawing_node_and_relationship_round_trip():
    polygon = Polygon(points=[Point(x=0, y=0), Point(x=10, y=0), Point(x=10, y=5), Point(x=0, y=5)])
    node_a = DrawingNode(id="n-a", node_kind="geometry", geometry_ref=polygon, role_hint=EntityKind.PLOT)
    node_b = DrawingNode(id="n-b", node_kind="evidence_candidate", evidence_ref="ec-1")

    edge = DrawingRelationship(
        id="rel-1",
        relation=RelationType.MEASURES,
        source_node_id=node_b.id,
        target_node_id=node_a.id,
        confidence=0.82,
        derivation="association_scoring",
        note="dimension text spatially aligned with top edge",
    )
    assert edge.relation == RelationType.MEASURES
    assert edge.confidence == 0.82

    deterministic_edge = DrawingRelationship(
        id="rel-2",
        relation=RelationType.ENCLOSES,
        source_node_id=node_a.id,
        target_node_id=node_a.id,
        confidence=1.0,
        derivation="deterministic_geometry",
    )
    assert deterministic_edge.derivation == "deterministic_geometry"


# --- StructuralHypothesis ----------------------------------------------------


def test_structural_hypothesis_scoring_is_named_and_auditable():
    hyp = StructuralHypothesis(
        id="hyp-1",
        identity=HypothesisIdentity.PLOT_BOUNDARY,
        supporting_node_ids=["n-a", "n-c"],
        constraint_results=[
            ConstraintResult(name="boundary_closure", score=2.0, passed=True, detail="4 sides, closed loop"),
            ConstraintResult(
                name="dimension_agreement", score=3.5, passed=True,
                detail="printed 17.59m within 2% of reconstructed 17.4m",
            ),
        ],
        total_score=5.5,
        document_evidence_ids=["ec-1"],
        contradictions=[],
    )
    assert hyp.total_score == sum(c.score for c in hyp.constraint_results)
    assert all(c.passed for c in hyp.constraint_results)


def test_structural_hypothesis_can_record_a_losing_alternative():
    winner = StructuralHypothesis(
        id="hyp-winner",
        identity=HypothesisIdentity.PLOT_BOUNDARY,
        total_score=15.0,
        rejected_alternative_ids=["hyp-loser"],
    )
    loser = StructuralHypothesis(
        id="hyp-loser",
        identity=HypothesisIdentity.SHEET_FRAME,
        total_score=-4.9,
        constraint_results=[
            ConstraintResult(name="sheet_frame_exclusion", score=-10.0, passed=False, detail="spans 98% of sheet bbox"),
        ],
    )
    assert loser.id in winner.rejected_alternative_ids
    assert loser.total_score < winner.total_score


# --- Decision -----------------------------------------------------------------


def test_decision_accept_status_carries_the_winning_candidate():
    decision = Decision(
        id="decision-plot.width-1",
        field_name="plot.width",
        status=DecisionStatus.ACCEPT,
        accepted_candidate_id="hyp-winner",
        considered_candidate_ids=["hyp-winner", "hyp-loser"],
        rejected=[
            ConstraintResult(name="sheet_frame_exclusion", score=-10.0, passed=False, detail="spans 98% of sheet bbox"),
        ],
        resulting_value_field="plot.width",
        confidence_derivation="hypothesis total_score margin of 19.9 over runner-up",
    )
    assert decision.status == DecisionStatus.ACCEPT
    assert decision.accepted_candidate_id == "hyp-winner"


def test_decision_abstain_status_has_no_accepted_candidate():
    decision = Decision(
        id="decision-road.width-1",
        field_name="road.width",
        status=DecisionStatus.ABSTAIN,
        considered_candidate_ids=["hyp-a", "hyp-b"],
        confidence_derivation="no hypothesis cleared the minimum usable score",
    )
    assert decision.status == DecisionStatus.ABSTAIN
    assert decision.accepted_candidate_id is None


# --- ValueField/Conflict additive fields do not disturb existing behaviour --


def test_value_field_decision_id_defaults_to_none_and_is_additive():
    vf = ValueField.missing(reason="not found on sheet 2")
    assert vf.decision_id is None
    vf2 = ValueField[float](value=12.0, confidence=Confidence(level=ConfidenceLevel.HIGH, score=0.95))
    vf2.decision_id = "decision-1"
    assert vf2.decision_id == "decision-1"


def test_conflict_structured_fields_default_empty_and_are_additive():
    conflict = Conflict(
        description="Two dimension lines disagree on plot width",
        conflicting_raw_values=[UnitValue(magnitude=12.0, unit="m"), UnitValue(magnitude=12.5, unit="m")],
        conflicting_sources=["sheet 1 dimension line", "sheet 3 table"],
    )
    assert conflict.conflicting_candidate_ids == []
    assert conflict.rejection_reasons == []

    conflict_with_ids = Conflict(
        description="Two hypotheses disagree",
        conflicting_candidate_ids=["hyp-a", "hyp-b"],
        rejection_reasons=["hyp-a: geometry disagrees with printed dimension by 12%"],
    )
    assert conflict_with_ids.conflicting_candidate_ids == ["hyp-a", "hyp-b"]
