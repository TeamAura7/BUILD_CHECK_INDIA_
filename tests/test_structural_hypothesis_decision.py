"""
Tests for `backend.spatial_reasoning.structural_hypothesis_decision.
decide_structural_hypothesis` (Architecture V2, Phase DXF-2) -- the new
multi-hypothesis ACCEPT/REJECT/CONFLICT/ABSTAIN arbitrator, generalizing
`plot_resolution.plot_confidence`'s margin pattern and `dxf_extractor.py`'s
own caption-confirmation/unconfirmed-drawing-identity ad-hoc mechanisms
(DXF_FAILURE_TAXONOMY.md items 0/9) into one auditable decision.

Margin constants used below (10.0 for HIGH, 4.0 for MEDIUM) are illustrative
test fixtures only, chosen to comfortably separate the test cases -- NOT the
calibrated defaults Phase DXF-3 will actually wire in (those are derived
separately, from real score-distribution data across the bundled DXF
fixtures, mirroring how `_SITE_PLAN_CAPTION_SCORE_BONUS` was calibrated).
"""

from __future__ import annotations

from backend.schemas.enums import ConfidenceLevel, DecisionStatus, HypothesisIdentity
from backend.schemas.geometry import Point, Polygon
from backend.schemas.hypothesis import ConstraintResult, StructuralHypothesis
from backend.spatial_reasoning.structural_hypothesis_decision import (
    MEASURED_MARGIN_FOR_HIGH,
    MEASURED_MARGIN_FOR_MEDIUM,
    decide_structural_hypothesis,
)

_MARGIN_HIGH = 10.0
_MARGIN_MEDIUM = 4.0
_MIN_USABLE = 2.0
_SQUARE = Polygon(points=[Point(x=0, y=0), Point(x=10, y=0), Point(x=10, y=10), Point(x=0, y=10)])


def _hyp(hyp_id: str, score: float, extra_terms: list[ConstraintResult] | None = None) -> StructuralHypothesis:
    return StructuralHypothesis(
        id=hyp_id, identity=HypothesisIdentity.PLOT_BOUNDARY, geometry=_SQUARE,
        total_score=score, constraint_results=extra_terms or [],
    )


def _decide(hypotheses):
    return decide_structural_hypothesis(
        HypothesisIdentity.PLOT_BOUNDARY, hypotheses,
        min_usable_score=_MIN_USABLE, margin_for_high=_MARGIN_HIGH, margin_for_medium=_MARGIN_MEDIUM,
    )


def test_abstain_when_no_hypotheses():
    decision, level = _decide([])
    assert decision.status == DecisionStatus.ABSTAIN
    assert level == ConfidenceLevel.MISSING
    assert decision.accepted_candidate_id is None


def test_abstain_when_winner_below_usability_floor():
    hyps = [_hyp("a", 1.0), _hyp("b", 0.5)]
    decision, level = _decide(hyps)
    assert decision.status == DecisionStatus.ABSTAIN
    assert level == ConfidenceLevel.MISSING
    assert len(decision.rejected) == 2  # every hypothesis, including the nominal "winner", is REJECTed


def test_accept_only_one_hypothesis_ships_medium_not_high():
    """Mirrors plot_confidence's own single-candidate special case: a lone
    candidate is accepted on its own merits, never automatically HIGH."""
    decision, level = _decide([_hyp("a", 10.0)])
    assert decision.status == DecisionStatus.ACCEPT
    assert decision.accepted_candidate_id == "a"
    assert level == ConfidenceLevel.MEDIUM


def test_accept_clear_winner_ships_high():
    hyps = [_hyp("a", 30.0), _hyp("b", 5.0)]  # margin 25 >= 10
    decision, level = _decide(hyps)
    assert decision.status == DecisionStatus.ACCEPT
    assert decision.accepted_candidate_id == "a"
    assert level == ConfidenceLevel.HIGH
    assert len(decision.rejected) == 1
    assert decision.rejected[0].detail.startswith("b ")


def test_accept_moderate_margin_ships_medium():
    hyps = [_hyp("a", 12.0), _hyp("b", 6.0)]  # margin 6, between 4 and 10
    decision, level = _decide(hyps)
    assert decision.status == DecisionStatus.ACCEPT
    assert level == ConfidenceLevel.MEDIUM


def test_accept_narrow_margin_but_runner_up_below_floor_ships_low():
    """The runner-up is BELOW the usability floor, so this is not a genuine
    tie (CONFLICT requires both to clear the floor) -- it's a real, if
    narrow-margin, ACCEPT at LOW confidence."""
    hyps = [_hyp("a", 3.0), _hyp("b", 1.0)]  # margin 2 < 4, but b is below floor (2.0)
    decision, level = _decide(hyps)
    assert decision.status == DecisionStatus.ACCEPT
    assert level == ConfidenceLevel.LOW


def test_conflict_when_two_well_formed_hypotheses_are_tied():
    hyps = [_hyp("a", 10.0), _hyp("b", 9.0)]  # margin 1 < 4, both clear the floor
    decision, level = _decide(hyps)
    assert decision.status == DecisionStatus.CONFLICT
    assert level == ConfidenceLevel.CONFLICTING
    assert decision.accepted_candidate_id is None
    assert len(decision.rejected) == 2


def test_caption_confirmation_overrides_a_narrow_margin_to_high():
    captioned = _hyp("captioned", 5.0, [
        ConstraintResult(name="caption_confirmation", score=20.0, passed=True, detail="caption found"),
    ])
    uncaptioned = _hyp("uncaptioned", 4.5, [
        ConstraintResult(name="caption_confirmation", score=0.0, passed=False, detail="no caption"),
    ])
    decision, level = _decide([captioned, uncaptioned])
    assert decision.status == DecisionStatus.ACCEPT
    assert decision.accepted_candidate_id == "captioned"
    assert level == ConfidenceLevel.HIGH  # NOT the margin-derived LOW this narrow a margin would otherwise give


def test_conflict_still_fires_when_both_tied_hypotheses_are_captioned():
    """The caption tie-break only applies when the WINNER has a decisive
    caption advantage over the runner-up (i.e. the runner-up is NOT also
    captioned) -- two independently captioned, comparably-scored hypotheses
    (a genuinely rarer but real scenario: two sub-drawings both captioned
    similarly) must still CONFLICT, not silently pick one."""
    both_captioned_term = [ConstraintResult(name="caption_confirmation", score=20.0, passed=True, detail="caption found")]
    hyps = [_hyp("a", 20.0, both_captioned_term), _hyp("b", 19.5, both_captioned_term)]
    decision, level = _decide(hyps)
    assert decision.status == DecisionStatus.CONFLICT
    assert level == ConfidenceLevel.CONFLICTING


def test_failed_identity_confirmation_caps_confidence_to_low_regardless_of_margin():
    hyps = [
        _hyp("winner", 30.0, [
            ConstraintResult(name="drawing_identity_confirmed", score=0.0, passed=False, detail="unconfirmed"),
        ]),
        _hyp("loser", 3.0),
    ]
    decision, level = _decide(hyps)  # margin 27, would otherwise be HIGH
    assert decision.status == DecisionStatus.ACCEPT
    assert level == ConfidenceLevel.LOW


def test_measured_margin_constants_correctly_classify_the_real_margins_they_were_derived_from():
    """Regression pin for `MEASURED_MARGIN_FOR_MEDIUM`/`_HIGH`'s own
    calibration data (see their docstring in structural_hypothesis_
    decision.py): the near-zero real margins from PLAN4/PLAN6/PLAN9-ROAD
    must classify as ambiguous (CONFLICT, since their runner-ups also clear
    the floor), the PLAN8 margin (2.392) as a modest-but-real separation
    (MEDIUM), and the larger real margins (PLAN2/PLAN5/PLAN1/PLAN9) as
    clearly decisive (HIGH). If this test ever needs to change, the
    constants' own calibration doc comment must be re-derived from new
    measured data, not just the numbers bumped."""
    def _margin_case(winner_score, runner_up_score):
        return decide_structural_hypothesis(
            HypothesisIdentity.PLOT_BOUNDARY,
            [_hyp("winner", winner_score), _hyp("runner_up", runner_up_score)],
            min_usable_score=2.0, margin_for_high=MEASURED_MARGIN_FOR_HIGH, margin_for_medium=MEASURED_MARGIN_FOR_MEDIUM,
        )

    # Near-zero real margins (PLAN4: 3.062 vs 3.059; PLAN6: 4.156 vs 4.118;
    # PLAN9 ROAD: 4.545 vs 4.545) -- genuinely tied, both clear the floor.
    for winner_score, runner_up_score in [(3.062, 3.059), (4.156, 4.118), (4.545, 4.545)]:
        decision, level = _margin_case(winner_score, runner_up_score)
        assert decision.status == DecisionStatus.CONFLICT, (winner_score, runner_up_score)
        assert level == ConfidenceLevel.CONFLICTING

    # PLAN8's real margin (5.439 vs 3.047 = 2.392) -- a modest, real
    # separation, below the HIGH threshold.
    decision, level = _margin_case(5.439, 3.047)
    assert decision.status == DecisionStatus.ACCEPT
    assert level == ConfidenceLevel.MEDIUM

    # PLAN2's real margin (4.345 vs -5.492 = 9.837, runner-up below floor)
    # and PLAN5's (15.033 vs 4.255 = 10.778) -- both clearly decisive.
    for winner_score, runner_up_score in [(4.345, -5.492), (15.033, 4.255)]:
        decision, level = _margin_case(winner_score, runner_up_score)
        assert decision.status == DecisionStatus.ACCEPT
        assert level == ConfidenceLevel.HIGH


def test_plan5_shaped_fixture_caption_confirmed_region_wins_despite_worse_raw_score():
    """Reproduces DXF_FAILURE_TAXONOMY.md item 0's exact real PLAN5.dxf
    numbers: the true site plan (region3) scored -4.967 structurally (the
    WORST of 12 regions) but is captioned, so +20 makes it 15.033; the
    winning-on-raw-structure-alone region (a wall section) scored 4.228
    with no caption. The captioned region must win AND ship HIGH, proving
    this function actually generalizes the existing ad-hoc mechanism, not
    just resembles it on paper."""
    true_site_plan = _hyp("region3", -4.967 + 20.0, [
        ConstraintResult(name="caption_confirmation", score=20.0, passed=True, detail="SITE PLAN caption found"),
    ])
    wall_section = _hyp("region0", 4.228, [
        ConstraintResult(name="caption_confirmation", score=0.0, passed=False, detail="no caption"),
    ])
    decision, level = decide_structural_hypothesis(
        HypothesisIdentity.PLOT_BOUNDARY, [true_site_plan, wall_section],
        min_usable_score=2.0, margin_for_high=10.0, margin_for_medium=4.0,
    )
    assert decision.status == DecisionStatus.ACCEPT
    assert decision.accepted_candidate_id == "region3"
    assert level == ConfidenceLevel.HIGH
