"""
Evidence reconciliation.

Combines multiple candidate values for the same semantic field (e.g. a
text-derived dimension value and a geometry-derived value for
`plot.width`) into exactly one `ValueField`, per the possible states
listed in phase3.md:

    AGREED, MINOR_VARIANCE, OUTLIER_DETECTED, CONFLICTING_EVIDENCE, MISSING

"If text says 5m and geometry says 2.2m: DO NOT silently choose one.
Return a conflict." — when candidates disagree beyond tolerance and no
robust majority can be established, the resulting ValueField has
`value=None` and a populated `conflict`, exactly like
`ValueField.conflicting()` elsewhere in the codebase.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from backend.config import get_settings
from backend.schemas.enums import ConfidenceLevel
from backend.schemas.evidence import Confidence, Conflict, GeometryEvidence, TextEvidence, ValueField
from backend.schemas.units import CanonicalUnit, UnitValue


class ReconciliationStatus(str, Enum):
    AGREED = "AGREED"
    UNCORROBORATED = "UNCORROBORATED"
    MINOR_VARIANCE = "MINOR_VARIANCE"
    OUTLIER_DETECTED = "OUTLIER_DETECTED"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    MISSING = "MISSING"


@dataclass
class EvidenceCandidate:
    value: float
    source: str
    evidence: Optional[TextEvidence | GeometryEvidence] = None
    weight: ConfidenceLevel = ConfidenceLevel.MEDIUM


@dataclass
class ReconciliationOutcome:
    status: ReconciliationStatus
    value: Optional[float]
    note: str
    outliers: list[EvidenceCandidate] = field(default_factory=list)


def _tolerance(median_value: float) -> float:
    settings = get_settings()
    return max(settings.geometry_tolerance_m, abs(median_value) * 0.05)


def reconcile(candidates: list[EvidenceCandidate]) -> ReconciliationOutcome:
    if not candidates:
        return ReconciliationOutcome(status=ReconciliationStatus.MISSING, value=None, note="No candidate evidence.")

    if len(candidates) == 1:
        # A single candidate is a plausible reading, not an "agreement" --
        # AGREED means independent sources concurred, and there is nothing
        # here for a second source to concur WITH. Treating this the same
        # as AGREED silently promoted every single-source field (which is
        # the common case whenever only one extraction method resolves a
        # field at all) straight to ConfidenceLevel.HIGH regardless of
        # whether it was ever cross-checked -- see `to_value_field` for the
        # confidence mapping this status now gets instead.
        c = candidates[0]
        return ReconciliationOutcome(
            status=ReconciliationStatus.UNCORROBORATED,
            value=c.value,
            note=f"Single source of evidence ({c.source}); nothing to cross-check against.",
        )

    values = [c.value for c in candidates]
    med = statistics.median(values)
    tol = _tolerance(med)

    if len(candidates) == 2:
        diff = abs(values[0] - values[1])
        if diff <= tol:
            return ReconciliationOutcome(
                status=ReconciliationStatus.AGREED,
                value=statistics.mean(values),
                note=f"{candidates[0].source} and {candidates[1].source} agree within tolerance "
                f"({diff:.3f} <= {tol:.3f}).",
            )
        if diff <= tol * 3:
            return ReconciliationOutcome(
                status=ReconciliationStatus.MINOR_VARIANCE,
                value=statistics.mean(values),
                note=f"{candidates[0].source} ({values[0]:.3f}) and {candidates[1].source} "
                f"({values[1]:.3f}) differ by {diff:.3f}, within a minor-variance band.",
            )
        conflict = Conflict(
            description=(
                f"{candidates[0].source} reports {values[0]:.3f} m but {candidates[1].source} "
                f"reports {values[1]:.3f} m — a difference of {diff:.3f} m, beyond tolerance "
                f"({tol:.3f} m). Not silently resolved."
            ),
            conflicting_raw_values=[UnitValue(magnitude=v, unit=CanonicalUnit.METRE.value) for v in values],
            conflicting_sources=[c.source for c in candidates],
        )
        return ReconciliationOutcome(
            status=ReconciliationStatus.CONFLICTING_EVIDENCE,
            value=None,
            note=conflict.description,
            outliers=[],
        ), conflict  # type: ignore[return-value]

    # 3+ candidates: robust median/MAD split into an agreeing majority + outliers.
    abs_devs = [abs(v - med) for v in values]
    mad = statistics.median(abs_devs)
    agreeing, outliers = [], []
    for c, dev in zip(candidates, abs_devs):
        robust_z = 0.6745 * dev / mad if mad > 1e-9 else 0.0
        (agreeing if robust_z <= 3.5 else outliers).append(c)

    if len(agreeing) <= len(candidates) / 2:
        conflict = Conflict(
            description=f"No majority agreement among {len(candidates)} sources for this field "
            f"(values: {[round(v, 3) for v in values]}).",
            conflicting_raw_values=[UnitValue(magnitude=v, unit=CanonicalUnit.METRE.value) for v in values],
            conflicting_sources=[c.source for c in candidates],
        )
        return ReconciliationOutcome(
            status=ReconciliationStatus.CONFLICTING_EVIDENCE, value=None, note=conflict.description
        ), conflict  # type: ignore[return-value]

    agree_values = [c.value for c in agreeing]
    agree_spread = (max(agree_values) - min(agree_values)) if len(agree_values) > 1 else 0.0
    resolved_value = statistics.mean(agree_values)
    if outliers:
        return ReconciliationOutcome(
            status=ReconciliationStatus.OUTLIER_DETECTED,
            value=resolved_value,
            note=f"{len(agreeing)}/{len(candidates)} sources agree "
            f"({[c.source for c in agreeing]}); {len(outliers)} rejected as outlier(s) "
            f"({[c.source for c in outliers]}).",
            outliers=outliers,
        )
    if agree_spread <= tol:
        status = ReconciliationStatus.AGREED
    else:
        status = ReconciliationStatus.MINOR_VARIANCE
    return ReconciliationOutcome(
        status=status,
        value=resolved_value,
        note=f"{len(agreeing)} sources ({[c.source for c in agreeing]}) reconciled, spread={agree_spread:.3f}.",
    )


def to_value_field(
    candidates: list[EvidenceCandidate],
    field_label: str,
    missing_reason: Optional[str] = None,
) -> ValueField[float]:
    """Reconcile candidates and package the result as a ValueField[float] in metres."""
    outcome = reconcile(candidates)
    conflict: Optional[Conflict] = None
    if isinstance(outcome, tuple):
        outcome, conflict = outcome

    evidence_list = [c.evidence for c in candidates if c.evidence is not None]

    if outcome.status == ReconciliationStatus.MISSING:
        return ValueField[float].missing(missing_reason or f"No evidence found for {field_label}.")

    if outcome.status == ReconciliationStatus.CONFLICTING_EVIDENCE:
        vf = ValueField[float].conflicting(conflict)
        vf.evidence = evidence_list
        vf.source = f"{field_label}: conflicting evidence"
        return vf

    level = {
        ReconciliationStatus.AGREED: ConfidenceLevel.HIGH,
        ReconciliationStatus.UNCORROBORATED: ConfidenceLevel.MEDIUM,
        ReconciliationStatus.MINOR_VARIANCE: ConfidenceLevel.MEDIUM,
        ReconciliationStatus.OUTLIER_DETECTED: ConfidenceLevel.MEDIUM,
    }[outcome.status]

    return ValueField[float](
        value=outcome.value,
        normalized_value=UnitValue(magnitude=round(outcome.value, 4), unit=CanonicalUnit.METRE.value),
        confidence=Confidence(level=level, reason=outcome.note),
        source=f"{field_label}: {outcome.status.value.lower()} across {len(candidates)} source(s)",
        evidence=evidence_list,
    )


__all__ = [
    "ReconciliationStatus",
    "EvidenceCandidate",
    "ReconciliationOutcome",
    "reconcile",
    "to_value_field",
]
