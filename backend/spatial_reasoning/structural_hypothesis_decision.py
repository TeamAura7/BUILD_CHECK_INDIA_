"""
Architecture V2, Phase DXF-2: multi-hypothesis structural arbitration.

`evidence_decision.decide_field` (Phase 7) only ever arbitrates between TWO
named scalar candidates (a CV value and a Vision value) for one field --
its own module docstring says explicitly that arbitrating between multiple
`StructuralHypothesis` records ("REJECT... that is StructuralHypothesis
territory") is not something it does. This module is that missing piece: a
generic ACCEPT/REJECT/CONFLICT/ABSTAIN decision over N competing
`StructuralHypothesis` records claiming the same role (PLOT_BOUNDARY/
BUILDING/ROAD/...), used today by `dxf_extractor.py`'s region resolution
(Phase DXF-3, not yet wired in) and, per Deliverable C.2's own intent,
reusable later by a PDF-side hypothesis generator without carrying any
DXF-specific knowledge.

Deliberately a SIBLING to `evidence_decision.py`, not an extension of it --
that module's scope stays "arbitrate CV vs. Vision for one field"; this
module's scope is "arbitrate N geometric/structural hypotheses for one
role." Keeping them separate keeps each one's contract simple and matches
how the project's own migration plan describes them as two different
Deliverable C.2 items (5 and 4, respectively).

Generalizes two things that already exist, rather than inventing a third
policy:
  1. `plot_resolution.plot_confidence`'s margin-based pattern (absolute
     floor first, then winner/runner-up score margin -> HIGH/MEDIUM/LOW),
     widened from 2 candidates to N.
  2. `dxf_extractor.py`'s own ad-hoc caption-confirmation/
     `unconfirmed_drawing_identity` confidence-capping (DXF_FAILURE_
     TAXONOMY.md items 0/9) -- read here from the `caption_confirmation`/
     `drawing_identity_confirmed` `ConstraintResult` terms Phase DXF-1's
     `dxf_evidence_projection.py` already attaches to every hypothesis, so
     this function has NO DXF-specific knowledge of captions/OCR/regions
     at all; it only knows those two term NAMES.
"""

from __future__ import annotations

import uuid
from typing import Optional

from backend.schemas.decision import Decision
from backend.schemas.enums import ConfidenceLevel, DecisionStatus, HypothesisIdentity
from backend.schemas.hypothesis import ConstraintResult, StructuralHypothesis

# Measured, not guessed -- derived by running Phase DXF-1's
# `_dxf_hypothesis_sink` hook (via a one-off research script, not committed)
# across every bundled real DXF fixture (PLAN1/2/4/5/6/7/8/9), then computing
# each plan+identity's own winner-vs-runner-up total_score margin (17 data
# points across PLOT_BOUNDARY/BUILDING/ROAD; PLAN7 excluded -- an
# intentionally empty DXF stub with zero hypotheses). Mirrors how
# `_SITE_PLAN_CAPTION_SCORE_BONUS` in `dxf_extractor.py` was itself
# calibrated: from a measured spread on the real corpus, with an explicit
# margin, not picked in isolation.
#
# The 17 real margins split into two clearly separated clusters, with a
# genuine gap between them (not an artifact of how the buckets were chosen):
#   near-zero / genuinely tied:  0.000, 0.001, 0.003, 0.003, 0.038, 0.038
#   clearly separated:           2.392, 2.392, 6.001*, 9.837, 9.837, 10.778,
#                                 10.778, 17.774, 17.774, 20.583, 20.583
#   (* 6.001 belongs to a winner that itself falls below the usability floor
#   -- ABSTAIN fires before this margin is ever compared against these
#   constants, so it doesn't constrain the choice below.)
# The near-zero cluster corresponds exactly to sheets this project's own
# taxonomy already documents as genuinely ambiguous (PLAN6's own two
# top-scoring regions, PLAN4's, and PLAN9's road candidates) -- i.e. the
# real data agrees with the already-known failure mode, not a fixture
# artifact.
#
# `MEASURED_MARGIN_FOR_MEDIUM` sits in the gap between the near-zero
# cluster's top (0.038) and the next real value (2.392) -- comfortably
# separating genuine ties from genuine (if modest) separation.
# `MEASURED_MARGIN_FOR_HIGH` sits in the gap between that modest-separation
# value (2.392) and the next real value (9.837) -- deliberately biased
# toward the conservative (higher) end of that gap rather than its
# midpoint, consistent with this project's "wrong is worse than missing"
# principle: prefer an under-confident MEDIUM over an over-confident HIGH
# when the real data doesn't clearly demand one or the other.
#
# n=8 real files (17 margins) is a small corpus -- re-measure as it grows,
# per this project's own established practice for this kind of threshold
# (see `DXF_FAILURE_TAXONOMY.md`'s "Provisional thresholds" section and
# `plot_resolution._MIN_USABLE_PLOT_SCORE`'s own docstring for the same
# discipline applied elsewhere).
MEASURED_MARGIN_FOR_MEDIUM = 2.0
MEASURED_MARGIN_FOR_HIGH = 8.0

# A confirmed drawing-type caption is strong, positive evidence of identity
# (DXF_FAILURE_TAXONOMY.md item 0) -- it outranks a merely-good structural
# margin. A FAILED `drawing_identity_confirmed` term is the opposite: a
# concrete signal that NOTHING on the sheet positively confirms this is the
# right drawing (item 9), which must cap confidence regardless of how
# comfortable the margin looks, exactly like `_CAPTION_OVERRIDE_CONFIDENCE_
# CAP`/`_RawPoly.unconfirmed_drawing_identity` already do today, ad hoc.
_CAPTION_TERM = "caption_confirmation"
_IDENTITY_TERM = "drawing_identity_confirmed"


def _decision_id(identity: HypothesisIdentity) -> str:
    return f"decision-{identity.value.lower()}-{uuid.uuid4().hex[:8]}"


def _has_passed_term(hypothesis: StructuralHypothesis, name: str) -> bool:
    return any(c.name == name and c.passed for c in hypothesis.constraint_results)


def _has_failed_term(hypothesis: StructuralHypothesis, name: str) -> bool:
    return any(c.name == name and not c.passed for c in hypothesis.constraint_results)


def _structural_confidence_level(
    accepted: StructuralHypothesis,
    margin: Optional[float],
    margin_for_high: float,
    margin_for_medium: float,
) -> tuple[ConfidenceLevel, str]:
    """Returns (level, reason). `margin` is `None` for the "only one
    hypothesis" case (mirrors `plot_confidence`'s own single-candidate
    special case: accepted on its own merits, never automatically HIGH)."""
    if _has_failed_term(accepted, _IDENTITY_TERM):
        return ConfidenceLevel.LOW, (
            f"'{_IDENTITY_TERM}' failed for the accepted hypothesis -- no positive signal anywhere "
            "confirms this is actually the right drawing, regardless of its structural margin."
        )
    if _has_passed_term(accepted, _CAPTION_TERM):
        return ConfidenceLevel.HIGH, (
            f"'{_CAPTION_TERM}' passed for the accepted hypothesis -- a confirmed drawing-type "
            "caption outranks structural margin alone."
        )
    if margin is None:
        return ConfidenceLevel.MEDIUM, "Only one hypothesis was considered for this role; accepted on its own merits."
    if margin >= margin_for_high:
        return ConfidenceLevel.HIGH, f"Clear winner (margin {margin:.2f} >= {margin_for_high:.2f} over the runner-up)."
    if margin >= margin_for_medium:
        return ConfidenceLevel.MEDIUM, f"Winner beats the runner-up by {margin:.2f} (>= {margin_for_medium:.2f})."
    return ConfidenceLevel.LOW, f"Winner only narrowly ({margin:.2f}) beats the runner-up; ambiguous."


def decide_structural_hypothesis(
    identity: HypothesisIdentity,
    hypotheses: list[StructuralHypothesis],
    *,
    min_usable_score: float,
    margin_for_high: float,
    margin_for_medium: float,
) -> tuple[Decision, ConfidenceLevel]:
    """
    Arbitrate between `hypotheses` (already filtered to a single `identity`
    by the caller -- this function has no opinion on how they were
    generated). Every non-winning hypothesis is explicitly REJECTed with
    its own named `ConstraintResult` in `Decision.rejected` -- the one thing
    `evidence_decision.decide_field` cannot do today for a whole hypothesis.

    Status:
      - `hypotheses` empty                                    -> ABSTAIN
      - winner's `total_score < min_usable_score`              -> ABSTAIN,
        every hypothesis (including the nominal "winner") individually
        REJECTed with its own score against the floor.
      - exactly one hypothesis, clears the floor               -> ACCEPT
        (mirrors `plot_confidence`'s "only one candidate" MEDIUM-not-HIGH
        treatment -- see the returned reason).
      - >=2 hypotheses, winner clears the floor, runner-up ALSO clears the
        floor, and their margin < `margin_for_medium`           -> CONFLICT
        (a genuine, well-formed tie -- neither is REJECTed as wrong, both
        are recorded as tied).
      - otherwise                                              -> ACCEPT,
        every other hypothesis REJECTed with its own score/margin.

    Returns `(Decision, ConfidenceLevel)`: `Decision` carries full
    provenance (which one accepted, which ones rejected and why);
    `ConfidenceLevel` is `MISSING` for ABSTAIN and `CONFLICTING` for
    CONFLICT (matching `ValueField.missing()`/`.conflicting()`'s own
    conventions elsewhere), or the margin/caption/identity-derived level
    from `_structural_confidence_level` for ACCEPT.
    """
    decision_id = _decision_id(identity)
    considered = [h.id for h in hypotheses]

    if not hypotheses:
        return Decision(
            id=decision_id, field_name=identity.value, status=DecisionStatus.ABSTAIN,
            considered_candidate_ids=[], resulting_value_field=None,
            confidence_derivation=f"No hypotheses were generated for {identity.value}.",
        ), ConfidenceLevel.MISSING

    ranked = sorted(hypotheses, key=lambda h: h.total_score, reverse=True)
    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None

    if winner.total_score < min_usable_score:
        rejected = [
            ConstraintResult(
                name="structural_hypothesis_score", score=h.total_score, passed=False,
                detail=f"{h.id} scored {h.total_score:.3f}, below the {min_usable_score:.3f} usability floor",
            )
            for h in ranked
        ]
        return Decision(
            id=decision_id, field_name=identity.value, status=DecisionStatus.ABSTAIN,
            considered_candidate_ids=considered, rejected=rejected, resulting_value_field=None,
            confidence_derivation=(
                f"Best hypothesis ({winner.id}) scored {winner.total_score:.3f}, below the "
                f"{min_usable_score:.3f} usability floor -- no hypothesis for {identity.value} is "
                "trustworthy enough to accept, regardless of margin."
            ),
        ), ConfidenceLevel.MISSING

    if runner_up is None:
        level, level_reason = _structural_confidence_level(winner, None, margin_for_high, margin_for_medium)
        return Decision(
            id=decision_id, field_name=identity.value, status=DecisionStatus.ACCEPT,
            accepted_candidate_id=winner.id, considered_candidate_ids=considered,
            resulting_value_field=identity.value,
            confidence_derivation=f"Only one hypothesis was considered for {identity.value}. {level_reason}",
        ), level

    margin = winner.total_score - runner_up.total_score

    # A confirmed drawing-type caption is a decisive tie-breaker, not just a
    # confidence multiplier applied after the fact -- this is the entire
    # point of `_SITE_PLAN_CAPTION_SCORE_BONUS` in the real system (per
    # DXF_FAILURE_TAXONOMY.md item 0: sized to exceed the full measured
    # structural-score spread specifically so a captioned region "always
    # outranks an uncaptioned one regardless of which end either sits at").
    # Without this check, two hypotheses whose POST-bonus totals happen to
    # land close together (an edge case the bonus's own margin was sized to
    # make rare, not impossible) would wrongly fall into CONFLICT even
    # though one of them carries independent, positive identity evidence
    # the other does not.
    winner_has_decisive_caption_advantage = (
        _has_passed_term(winner, _CAPTION_TERM) and not _has_passed_term(runner_up, _CAPTION_TERM)
    )

    if (
        runner_up.total_score >= min_usable_score
        and margin < margin_for_medium
        and not winner_has_decisive_caption_advantage
    ):
        rejected = [
            ConstraintResult(
                name="structural_hypothesis_tied", score=h.total_score, passed=False,
                detail=f"{h.id} scored {h.total_score:.3f}, tied with its top competitor (margin {margin:.3f})",
            )
            for h in (winner, runner_up)
        ]
        return Decision(
            id=decision_id, field_name=identity.value, status=DecisionStatus.CONFLICT,
            considered_candidate_ids=considered, rejected=rejected, resulting_value_field=None,
            confidence_derivation=(
                f"{winner.id} (score {winner.total_score:.3f}) and {runner_up.id} "
                f"(score {runner_up.total_score:.3f}) are both above the usability floor and within "
                f"{margin_for_medium:.3f} of each other -- refusing to arbitrarily pick one."
            ),
        ), ConfidenceLevel.CONFLICTING

    level, level_reason = _structural_confidence_level(winner, margin, margin_for_high, margin_for_medium)
    rejected = [
        ConstraintResult(
            name="structural_hypothesis_score", score=h.total_score, passed=False,
            detail=f"{h.id} scored {h.total_score:.3f}, {winner.total_score - h.total_score:.3f} below the winner",
        )
        for h in ranked[1:]
    ]
    return Decision(
        id=decision_id, field_name=identity.value, status=DecisionStatus.ACCEPT,
        accepted_candidate_id=winner.id, considered_candidate_ids=considered,
        rejected=rejected, resulting_value_field=identity.value,
        confidence_derivation=(
            f"{winner.id} accepted with score {winner.total_score:.3f} "
            f"(margin {margin:.3f} over runner-up {runner_up.id}). {level_reason}"
        ),
    ), level


__all__ = ["decide_structural_hypothesis", "MEASURED_MARGIN_FOR_MEDIUM", "MEASURED_MARGIN_FOR_HIGH"]
