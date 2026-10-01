"""
Evidence decision engine (Architecture V2, Phase 7).

See ARCHITECTURE_V2.md, Deliverable C.2 item 5, and Deliverable B.6 for the
finding that motivates this module: authority for a field's final value is
today fragmented across `plot_resolution.py`, `pipeline.py`,
`final_fusion.py`, and a fourth, best-designed-but-UNWIRED module,
`document_evidence.py`. This module does not invent a new arbitration
policy -- it takes `document_evidence.py`'s own scoring/threshold/
provenance design (weighted `EVIDENCE_SCORE_WEIGHTS`, an explicit
`UNRESOLVED_CONFLICT` refusal state) and wraps it to speak the
ACCEPT/REJECT/CONFLICT/ABSTAIN vocabulary
(`backend.schemas.enums.DecisionStatus`) plus this project's `Decision`/
`ValueField` schemas, so the SAME decision computed here can eventually
replace `final_fusion.apply_final_agreement_to_plan`'s overwrite-the-
first-pass pattern (Deliverable C.2 item 5) instead of adding a fifth
competing policy.

**Status: shadow-mode only.** Nothing in `pipeline.py`/`final_fusion.py`
calls this module yet -- see `backend/tools/run_evidence_decision_shadow.py`
for the harness that runs it alongside the real, currently-shipping
pipeline on all 7 real plans and reports where the two would disagree,
without changing what either PDF or DXF plans actually ship. Wiring this in
for real is Phase 8, gated on that comparison, per ARCHITECTURE_V2.md's
phased order.

Reused, not reimplemented, from `document_evidence.py`: `DocumentTextIndex`,
`evaluate_candidate`, `resolve_field`, `EVIDENCE_SCORE_WEIGHTS`,
`DEFAULT_SCORE_THRESHOLD`. This module's own job is exactly the translation
at the boundary: `document_evidence`'s status-string vocabulary and
`float`/`CandidateEvidence` outputs, in; `Decision`/`ValueField`, out --
with `ConfidenceLevel` derived from `document_evidence`'s own status
categories rather than a rescored numeric threshold (see
`_STATUS_TO_CONFIDENCE_LEVEL`'s docstring for why), so it is never silently
MEDIUM without a real, attached score.
"""

from __future__ import annotations

import uuid
from typing import Iterable, Optional

from backend.schemas.decision import Decision
from backend.schemas.enums import ConfidenceLevel, DecisionStatus
from backend.schemas.evidence import Confidence, Conflict, TextEvidence, ValueField
from backend.schemas.hypothesis import ConstraintResult
from backend.schemas.independent_measurements import IndependentMeasurement
from backend.schemas.units import UnitValue
from backend.spatial_reasoning import document_evidence as doc_ev

# The maximum evidence score `document_evidence.evaluate_candidate` could
# ever assign (every weight fires at once) -- computed from the SAME
# weights table, never hand-duplicated, so this stays correct if the
# weights table is ever extended. Used only as an ADVISORY normalization
# for `Confidence.score` (audit/display purposes) -- see
# `_STATUS_TO_CONFIDENCE_LEVEL` below for why `Confidence.level` is NOT
# derived from this normalized score.
_MAX_EVIDENCE_SCORE = sum(doc_ev.EVIDENCE_SCORE_WEIGHTS.values())

# `confidence_from_source`/`confidence_level_from_score` (enums.py) apply
# fixed 0.85/0.75 thresholds calibrated for a genuine 0-1 probability (a
# Vision model's own confidence output). `document_evidence.py`'s integer
# evidence score is a fundamentally different kind of signal -- an
# unbounded-ish sum of weighted boolean flags (native-text confirmation,
# geometry support, repeated occurrence, ...) where achieving anywhere near
# the theoretical maximum requires several largely-alternative evidence
# sources to ALL fire at once, something a genuinely well-evidenced real
# value rarely does (a measurement has exactly one `.source`, so
# native-text/OCR/geometry confirmation are mostly mutually exclusive in
# practice). Naively normalizing this score into [0, 1] and running it
# through the Vision-calibrated thresholds systematically under-rates good
# evidence -- confirmed directly: a value with a real NATIVE_TEXT-sourced
# measurement, dimension-line association, and semantic association
# (score 13) normalizes to 13/22 = 0.59, below the LOW/MEDIUM cutoff,
# despite being exactly the kind of evidence `document_evidence.py`'s own
# `has_direct_document_evidence()` already calls "verified". This is the
# same class of mistake this project's own audit flagged elsewhere
# (ARCHITECTURE_V2.md Deliverable B.5): applying one score-to-level mapping
# across two signals with different generative processes.
#
# The fix: derive `Confidence.level` directly from `document_evidence.py`'s
# OWN already-validated status vocabulary (not a new numeric threshold),
# and keep the normalized score purely ADVISORY (`Confidence.score`, for
# audit/display), never authoritative for branching -- exactly the contract
# `Confidence`'s own docstring already describes.
_STATUS_TO_CONFIDENCE_LEVEL = {
    doc_ev.AGREED_DOCUMENT_VERIFIED: ConfidenceLevel.HIGH,
    doc_ev.CV_ONLY_DOCUMENT_VERIFIED: ConfidenceLevel.HIGH,
    doc_ev.VISION_ONLY_DOCUMENT_VERIFIED: ConfidenceLevel.HIGH,
    doc_ev.CONFLICT_RESOLVED_BY_DOCUMENT_EVIDENCE: ConfidenceLevel.HIGH,
    # Two independent sources agreeing, even without document confirmation,
    # is treated as a stronger signal than either source's own evidence
    # alone -- this mirrors an existing, deliberate choice already made in
    # `final_fusion.py`'s CV+Vision-AGREED branch (per this project's own
    # audit, ARCHITECTURE_V2.md Deliverable B.5), not a new invented rule.
    doc_ev.AGREED_UNVERIFIED: ConfidenceLevel.MEDIUM,
    doc_ev.CV_ONLY_UNVERIFIED: ConfidenceLevel.LOW,
    doc_ev.VISION_ONLY_UNVERIFIED: ConfidenceLevel.LOW,
}


def _decision_id(field_name: str) -> str:
    return f"decision-{field_name}-{uuid.uuid4().hex[:8]}"


def _accept_confidence(status: str, score: int, reason: str) -> Confidence:
    """Build the `Confidence` for an ACCEPT decision. `level` comes from
    `_STATUS_TO_CONFIDENCE_LEVEL` (see its own docstring for why), `score`
    is the advisory normalized evidence score -- always present, never
    `None`, so this can never construct the "MEDIUM with no score" pattern
    this project's own audit found recurring elsewhere (ARCHITECTURE_V2.md
    Deliverable B.5/D.5)."""
    level = _STATUS_TO_CONFIDENCE_LEVEL[status]
    normalized = min(1.0, max(0.0, score / _MAX_EVIDENCE_SCORE)) if _MAX_EVIDENCE_SCORE > 0 else 0.0
    return Confidence(level=level, score=normalized, reason=reason)


def decide_field(
    field_name: str,
    cv_value: Optional[float],
    vision_value: Optional[float],
    unit: Optional[str],
    field_measurements: Iterable[IndependentMeasurement],
    text_index: "doc_ev.DocumentTextIndex",
    score_threshold: int = doc_ev.DEFAULT_SCORE_THRESHOLD,
) -> tuple[Decision, ValueField]:
    """
    The authoritative decision for one field, given CV/Vision candidates and
    the document's own evidence (native text/OCR/geometry) -- the axis
    `document_evidence.resolve_field` already models. Returns
    `(Decision, ValueField)`: the full provenance record and the
    ready-to-ship value.

    Status mapping from `document_evidence`'s vocabulary:
      - `NOT_FOUND_BY_EITHER`                              -> ABSTAIN
      - `AGREED_DOCUMENT_VERIFIED`/`AGREED_UNVERIFIED`      -> ACCEPT
      - `CV_ONLY_DOCUMENT_VERIFIED`/`CV_ONLY_UNVERIFIED`    -> ACCEPT
      - `VISION_ONLY_DOCUMENT_VERIFIED`/`_UNVERIFIED`       -> ACCEPT
      - `CONFLICT_RESOLVED_BY_DOCUMENT_EVIDENCE`            -> ACCEPT
      - `UNRESOLVED_CONFLICT`, both candidates well-evidenced and tied
                                                             -> CONFLICT
      - `UNRESOLVED_CONFLICT`, neither candidate well-evidenced           -> ABSTAIN
    `document_evidence.py` itself does not distinguish these last two cases
    in its own `status` string (both are `UNRESOLVED_CONFLICT`) -- this
    function re-derives the distinction from the returned `CandidateEvidence`
    objects' own scores, without modifying `document_evidence.py`.

    `REJECT` is not produced by this function: `document_evidence.py`'s
    model only ever arbitrates between CV/Vision candidates that already
    exist, it does not itself generate and reject whole competing
    hypotheses (that is `StructuralHypothesis`/constraint-scoring
    territory, Deliverable C.2 item 4, not yet wired into this function --
    see module docstring's "Status" note).
    """
    field_measurements = list(field_measurements)
    verdict = doc_ev.resolve_field(field_name, cv_value, vision_value, unit, field_measurements, text_index, score_threshold)

    decision_id = _decision_id(field_name)
    considered = [c for c in ("cv" if cv_value is not None else None, "vision" if vision_value is not None else None) if c]

    if verdict.status == doc_ev.NOT_FOUND_BY_EITHER:
        decision = Decision(
            id=decision_id, field_name=field_name, status=DecisionStatus.ABSTAIN,
            considered_candidate_ids=[], resulting_value_field=field_name,
            confidence_derivation="neither CV nor Vision produced a candidate for this field",
        )
        return decision, ValueField.missing(reason=verdict.reason, decision_id=decision_id)

    if verdict.status == doc_ev.UNRESOLVED_CONFLICT:
        cv_ev, vision_ev = verdict.cv_evidence, verdict.vision_evidence
        cv_wins = cv_ev is not None and cv_ev.score >= score_threshold and cv_ev.has_direct_document_evidence()
        vision_wins = vision_ev is not None and vision_ev.score >= score_threshold and vision_ev.has_direct_document_evidence()

        if cv_wins and vision_wins:
            # Both candidates independently cleared the bar with direct
            # document evidence and still disagree -- a genuine conflict,
            # not a lack of evidence.
            conflict = Conflict(
                description=verdict.reason,
                conflicting_raw_values=[
                    v for v in (
                        UnitValue(magnitude=cv_value, unit=unit) if unit else None,
                        UnitValue(magnitude=vision_value, unit=unit) if unit else None,
                    ) if v is not None
                ],
                conflicting_sources=["cv", "vision"],
                conflicting_candidate_ids=["cv", "vision"],
                rejection_reasons=[
                    f"cv (score {cv_ev.score}) and vision (score {vision_ev.score}) are equally well "
                    "supported by document evidence; refusing to arbitrarily pick one"
                ],
            )
            decision = Decision(
                id=decision_id, field_name=field_name, status=DecisionStatus.CONFLICT,
                considered_candidate_ids=considered,
                rejected=[
                    ConstraintResult(name="document_evidence_score", score=float(cv_ev.score), passed=False,
                                      detail=f"cv scored {cv_ev.score}, tied with vision"),
                    ConstraintResult(name="document_evidence_score", score=float(vision_ev.score), passed=False,
                                      detail=f"vision scored {vision_ev.score}, tied with cv"),
                ],
                resulting_value_field=field_name,
                confidence_derivation="genuine conflict: both candidates equally well-evidenced, refusing to guess",
            )
            return decision, ValueField.conflicting(conflict, decision_id=decision_id)

        # Neither candidate cleared the bar -- insufficient evidence to
        # decide, not a confirmed disagreement.
        decision = Decision(
            id=decision_id, field_name=field_name, status=DecisionStatus.ABSTAIN,
            considered_candidate_ids=considered,
            rejected=[
                r for r in (
                    ConstraintResult(name="document_evidence_score", score=float(cv_ev.score), passed=False,
                                      detail=f"cv scored {cv_ev.score}, below threshold {score_threshold} or lacks direct document evidence")
                    if cv_ev is not None else None,
                    ConstraintResult(name="document_evidence_score", score=float(vision_ev.score), passed=False,
                                      detail=f"vision scored {vision_ev.score}, below threshold {score_threshold} or lacks direct document evidence")
                    if vision_ev is not None else None,
                ) if r is not None
            ],
            resulting_value_field=field_name,
            confidence_derivation="disagreement present but neither candidate is well-evidenced enough to decide",
        )
        return decision, ValueField.missing(reason=verdict.reason, decision_id=decision_id)

    # Every remaining status is an ACCEPT: AGREED_*, CV_ONLY_*, VISION_ONLY_*,
    # CONFLICT_RESOLVED_BY_DOCUMENT_EVIDENCE.
    winning_evidence = verdict.cv_evidence if verdict.winner in ("cv", "cv+vision") else verdict.vision_evidence
    score = winning_evidence.score if winning_evidence is not None else 0
    confidence = _accept_confidence(verdict.status, score, reason=f"{verdict.status}: {verdict.reason}")

    evidence_list = []
    if winning_evidence is not None:
        evidence_list.append(TextEvidence(raw_text=winning_evidence.raw_value or str(verdict.final_value)))

    rejected: list[ConstraintResult] = []
    if verdict.status == doc_ev.CONFLICT_RESOLVED_BY_DOCUMENT_EVIDENCE:
        loser_evidence = verdict.vision_evidence if verdict.winner == "cv" else verdict.cv_evidence
        loser_name = "vision" if verdict.winner == "cv" else "cv"
        if loser_evidence is not None:
            rejected.append(
                ConstraintResult(
                    name="document_evidence_score", score=float(loser_evidence.score), passed=False,
                    detail=f"{loser_name} scored {loser_evidence.score}, lower than the winner",
                )
            )

    decision = Decision(
        id=decision_id, field_name=field_name, status=DecisionStatus.ACCEPT,
        accepted_candidate_id=verdict.winner, considered_candidate_ids=considered,
        rejected=rejected, resulting_value_field=field_name,
        confidence_derivation=(
            f"document-evidence score {score}/{_MAX_EVIDENCE_SCORE} -> normalized "
            f"{confidence.score:.3f} -> {confidence.level.value} ({verdict.status})"
        ),
    )
    value_field = ValueField[float](
        value=verdict.final_value,
        raw_value=UnitValue(magnitude=verdict.final_value, unit=unit) if unit and verdict.final_value is not None else None,
        confidence=confidence,
        source=f"evidence_decision:{verdict.status}",
        evidence=evidence_list,
        decision_id=decision_id,
    )
    return decision, value_field


__all__ = ["decide_field"]
