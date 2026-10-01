"""Tests for `backend.spatial_reasoning.evidence_reconciliation`."""

from __future__ import annotations

from backend.schemas.enums import ConfidenceLevel
from backend.spatial_reasoning.evidence_reconciliation import EvidenceCandidate, reconcile, to_value_field


def test_single_source_is_uncorroborated_not_agreed():
    """A single candidate has nothing to cross-check against, so it must not
    be reported as AGREED (which means independent sources concurred).
    Audit finding: this previously returned AGREED unconditionally, which
    `to_value_field` then laundered straight into ConfidenceLevel.HIGH --
    identical to genuine multi-source corroboration."""
    outcome = reconcile([EvidenceCandidate(value=5.0, source="text")])
    assert outcome.status.value == "UNCORROBORATED"
    assert outcome.value == 5.0


def test_single_unsupported_candidate_does_not_become_high_confidence():
    """Requirement 1: a lone candidate must resolve at MEDIUM, never HIGH,
    purely because it is the only one -- HIGH is reserved for values an
    independent second source actually confirmed."""
    vf = to_value_field([EvidenceCandidate(value=5.0, source="text")], field_label="plot.width")
    assert vf.value == 5.0
    assert vf.status == ConfidenceLevel.MEDIUM


def test_independently_corroborated_candidates_still_reach_high_confidence():
    """Requirement 2: genuine multi-source agreement must still resolve at
    HIGH -- the fix must not indiscriminately downgrade everything."""
    vf = to_value_field(
        [EvidenceCandidate(value=5.0, source="text"), EvidenceCandidate(value=5.02, source="geometry")],
        field_label="plot.width",
    )
    assert vf.value is not None
    assert vf.status == ConfidenceLevel.HIGH


def test_no_candidates_is_missing():
    outcome = reconcile([])
    assert outcome.status.value == "MISSING"
    assert outcome.value is None


def test_two_sources_within_tolerance_agree():
    outcome = reconcile([EvidenceCandidate(value=5.0, source="text"), EvidenceCandidate(value=5.02, source="geometry")])
    assert outcome.status.value == "AGREED"


def test_conflicting_evidence_never_silently_picks_one_value():
    """
    Directly from phase3.md: "If text says 5m and geometry says 2.2m:
    DO NOT silently choose one. Return a conflict."
    """
    result = reconcile(
        [EvidenceCandidate(value=5.0, source="text says 5m"), EvidenceCandidate(value=2.2, source="geometry says 2.2m")]
    )
    outcome, conflict = result
    assert outcome.status.value == "CONFLICTING_EVIDENCE"
    assert outcome.value is None
    assert conflict is not None
    assert "5.000" in conflict.description or "5.0" in conflict.description
    assert "2.200" in conflict.description or "2.2" in conflict.description


def test_to_value_field_conflicting_has_none_value_and_populated_conflict():
    vf = to_value_field(
        [EvidenceCandidate(value=5.0, source="text"), EvidenceCandidate(value=2.2, source="geometry")],
        field_label="plot.width",
    )
    assert vf.value is None
    assert vf.status == ConfidenceLevel.CONFLICTING
    assert vf.conflict is not None


def test_outlier_among_three_plus_sources_is_detected_not_hidden():
    outcome = reconcile(
        [
            EvidenceCandidate(value=10.0, source="a"),
            EvidenceCandidate(value=10.1, source="b"),
            EvidenceCandidate(value=9.9, source="c"),
            EvidenceCandidate(value=25.0, source="d-outlier"),
        ]
    )
    assert outcome.status.value == "OUTLIER_DETECTED"
    assert any(o.source == "d-outlier" for o in outcome.outliers)
    # resolved value should reflect the agreeing majority, not be dragged by the outlier
    assert 9.5 < outcome.value < 10.5


def test_to_value_field_missing_uses_custom_reason():
    vf = to_value_field([], field_label="road.width", missing_reason="No road evidence at all.")
    assert vf.value is None
    assert vf.status == ConfidenceLevel.MISSING
    assert vf.confidence.reason == "No road evidence at all."


def test_minor_variance_still_resolves_a_value():
    outcome = reconcile([EvidenceCandidate(value=10.0, source="a"), EvidenceCandidate(value=10.5, source="b")])
    assert outcome.status.value in ("MINOR_VARIANCE", "AGREED")
    assert outcome.value is not None
