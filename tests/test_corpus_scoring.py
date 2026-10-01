"""Tests for `backend.corpus.scoring` (pure scoring; no extraction)."""

from __future__ import annotations

from backend.corpus.schema import PlanTruth, TruthField
from backend.corpus.scoring import aggregate, disagreement_analysis, score_plan, within_tolerance


def _truth(**fields) -> PlanTruth:
    return PlanTruth(plan_id="T", fields={
        name.replace("__", "."): (tf if isinstance(tf, TruthField) else TruthField(value=tf, verification="printed"))
        for name, tf in fields.items()
    })


def _p(value, level="HIGH", flagged=False):
    return {"value": value, "confidence": level, "flagged": flagged}


def _by_field(scores):
    return {s.field: s for s in scores}


def test_tolerance_is_the_same_regime_as_the_existing_eval_harness():
    from backend.corpus import scoring
    from backend.tools import eval_harness

    assert scoring.ABS_TOL == eval_harness.DEFAULT_ABS_TOL
    assert scoring.REL_TOL == eval_harness.DEFAULT_REL_TOL
    for actual, expected in ((10.0, 10.1), (10.0, 11.0), (100.0, 104.9), (100.0, 106.0), (0.0, 0.1)):
        assert within_tolerance(actual, expected) == eval_harness._within_tolerance(actual, expected)


def test_correct_wrong_and_missing_are_kept_apart():
    truth = _truth(road__width=9.2, setbacks__front=3.0, setbacks__rear=1.5)
    scores = _by_field(score_plan(truth, {"road.width": _p(9.2), "setbacks.front": _p(1.0)}))
    assert scores["road.width"].status == "CORRECT"
    assert scores["setbacks.front"].status == "WRONG"
    assert scores["setbacks.rear"].status == "MISSING"


def test_axis_pairs_are_scored_as_an_unordered_pair():
    truth = _truth(plot__width=12.19, plot__depth=18.28)
    scores = _by_field(score_plan(truth, {"plot.width": _p(18.288), "plot.depth": _p(12.192)}))
    assert scores["plot.width"].status == scores["plot.depth"].status == "CORRECT"
    assert scores["plot.width"].axis_swapped
    assert aggregate(scores.values())["axis_swapped"] == 2


def test_a_swapped_pair_with_a_wrong_value_is_still_wrong():
    truth = _truth(plot__width=12.19, plot__depth=18.28)
    scores = _by_field(score_plan(truth, {"plot.width": _p(18.288), "plot.depth": _p(9.0)}))
    assert scores["plot.width"].status == "CORRECT"
    assert scores["plot.depth"].status == "WRONG"


def test_a_lone_axis_value_matching_either_side_is_correct_but_a_stranger_is_wrong():
    truth = _truth(plot__width=12.19, plot__depth=18.28)
    assert _by_field(score_plan(truth, {"plot.width": _p(18.28)}))["plot.width"].status == "CORRECT"
    assert _by_field(score_plan(truth, {"plot.width": _p(26.0)}))["plot.width"].status == "WRONG"


def test_count_and_category_fields_match_exactly():
    truth = _truth(building__floor_count=4, building_use="residential")
    scores = _by_field(score_plan(truth, {"building.floor_count": _p(3), "building_use": _p("Residential")}))
    assert scores["building.floor_count"].status == "WRONG"
    assert scores["building_use"].status == "CORRECT"


def test_must_abstain_rewards_silence_and_punishes_an_answer():
    truth = _truth(road__width=TruthField(value=None, verification="must_abstain"))
    silent = score_plan(truth, {"road.width": _p(None)})
    assert silent[0].status == "ABSTAINED_OK"
    loud = score_plan(truth, {"road.width": _p(7.3)})
    assert loud[0].status == "SPURIOUS" and loud[0].confident
    agg = aggregate(loud)
    assert agg["confident_wrong"] == 1 and agg["confident_wrong_rate"] == 1.0


def test_confident_wrong_only_counts_unflagged_high_or_medium_answers():
    truth = _truth(setbacks__front=3.0, setbacks__rear=1.5, setbacks__left=1.0, setbacks__right=1.0)
    preds = {
        "setbacks.front": _p(9.0, "HIGH"),                 # confident and wrong
        "setbacks.rear": _p(9.0, "LOW"),                   # wrong but caught by low confidence
        "setbacks.left": _p(9.0, "HIGH", flagged=True),    # wrong but carries a conflict flag
        "setbacks.right": _p(1.0, "HIGH"),                 # confident and right
    }
    agg = aggregate(score_plan(truth, preds))
    assert agg["confident_answers"] == 2
    assert agg["confident_wrong"] == 1
    assert agg["confident_wrong_rate"] == 0.5
    assert agg["wrong_answers_caught"] == 2 / 3


def test_risk_coverage_is_reported_at_each_confidence_threshold():
    truth = _truth(setbacks__front=3.0, setbacks__rear=1.5, setbacks__left=1.0)
    preds = {"setbacks.front": _p(3.0, "HIGH"), "setbacks.rear": _p(9.0, "MEDIUM"), "setbacks.left": _p(9.0, "LOW")}
    rc = {r["confidence_at_least"]: r for r in aggregate(score_plan(truth, preds))["risk_coverage"]}
    assert rc["HIGH"]["risk"] == 0.0 and rc["HIGH"]["answers"] == 1
    assert rc["HIGH+MEDIUM"]["risk"] == 0.5
    assert rc["ALL"]["risk"] == 2 / 3 and rc["ALL"]["coverage"] == 1.0


def test_unverified_truth_is_excluded_unless_asked_for():
    truth = _truth(road__width=TruthField(value=9.2, verification="unverified"))
    assert score_plan(truth, {"road.width": _p(1.0)}) == []
    assert score_plan(truth, {"road.width": _p(1.0)}, verifications=("unverified",))[0].status == "WRONG"


def test_unannotated_and_unknown_fields_are_ignored():
    truth = _truth(road__width=TruthField())
    assert score_plan(truth, {"road.width": _p(9.2)}) == []


def test_disagreement_between_independent_pipelines_detects_a_wrong_dxf():
    truths = {"T": _truth(plot__width=10.0, plot__depth=13.0, setbacks__front=3.0, setbacks__rear=1.0)}
    pdf = {"T": {"plot.width": _p(10.0), "plot.depth": _p(13.0), "setbacks.front": _p(3.0), "setbacks.rear": _p(1.0)}}
    dxf = {"T": {"plot.width": _p(2.5), "plot.depth": _p(13.05), "setbacks.front": _p(0.3), "setbacks.rear": _p(1.0)}}
    result = disagreement_analysis(truths, pdf, dxf)
    dxf_detect = result["detects_dxf_wrong"]
    assert result["pairs"] == 4
    assert dxf_detect["flagged_and_wrong"] == 2
    assert dxf_detect["missed_wrong"] == 0
    assert dxf_detect["precision"] == 1.0 and dxf_detect["recall"] == 1.0
    assert dxf_detect["auroc"] == 1.0


def test_disagreement_ignores_fields_only_one_pipeline_answered():
    truths = {"T": _truth(plot__width=10.0)}
    result = disagreement_analysis(truths, {"T": {"plot.width": _p(10.0)}}, {"T": {"plot.width": _p(None)}})
    assert result["pairs"] == 0
    assert result["detects_dxf_wrong"]["auroc"] is None
