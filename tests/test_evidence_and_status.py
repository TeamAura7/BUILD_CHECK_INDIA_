from __future__ import annotations

from backend.schemas.enums import ComplianceStatus, ConfidenceLevel
from backend.schemas.evidence import Confidence, Conflict, ValueField
from backend.schemas.units import CanonicalUnit, UnitValue


def test_value_field_missing_factory():
    vf = ValueField.missing(reason="not found on sheet 2")
    assert vf.value is None
    assert vf.status == ConfidenceLevel.MISSING
    assert vf.confidence.reason == "not found on sheet 2"


def test_value_field_conflicting_factory():
    conflict = Conflict(
        description="Two dimension lines disagree on plot width",
        conflicting_raw_values=[
            UnitValue(magnitude=12.0, unit="m"),
            UnitValue(magnitude=12.5, unit="m"),
        ],
        conflicting_sources=["sheet 1 dimension line", "sheet 3 table"],
    )
    vf = ValueField.conflicting(conflict)
    assert vf.value is None
    assert vf.status == ConfidenceLevel.CONFLICTING
    assert vf.conflict is not None
    assert len(vf.conflict.conflicting_raw_values) == 2


def test_value_field_with_full_evidence_chain():
    vf = ValueField[float](
        value=12.0,
        raw_value=UnitValue(magnitude=39.37, unit="ft"),
        normalized_value=UnitValue(magnitude=12.0, unit=CanonicalUnit.METRE.value),
        confidence=Confidence(level=ConfidenceLevel.HIGH, score=0.95),
        source="sheet 1, west boundary dimension",
    )
    assert vf.status == ConfidenceLevel.HIGH
    assert vf.confidence.score == 0.95


def test_compliance_status_values_not_collapsed():
    # Every status must be distinct — this test guards against someone
    # later collapsing uncertainty states into PASS/FAIL.
    values = {s.value for s in ComplianceStatus}
    assert values == {
        "PASS",
        "FAIL",
        "INSUFFICIENT_DATA",
        "NOT_APPLICABLE",
        "CONFLICTING_EVIDENCE",
        "REQUIRES_REVIEW",
    }
