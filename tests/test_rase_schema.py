from __future__ import annotations

import pytest

from backend.rase.schema import (
    FieldResolutionError,
    evaluate_applies_when,
    evaluate_threshold,
    parse_condition,
    resolve_field,
)
from backend.schemas.enums import ConfidenceLevel
from tests.conftest import make_value_field


def test_resolve_field_known_path(sample_normalized_plan):
    vf = resolve_field(sample_normalized_plan, "setbacks.front")
    assert vf.value == 3.0


def test_resolve_field_unknown_path_raises(sample_normalized_plan):
    with pytest.raises(FieldResolutionError):
        resolve_field(sample_normalized_plan, "plot.nonexistent")


def test_leaf_condition_true(sample_normalized_plan):
    result, observed, desc = evaluate_threshold(
        {"field": "setbacks.front", "op": ">=", "value": 3.0}, sample_normalized_plan
    )
    assert result is True
    assert len(observed) == 1
    assert observed[0].value == 3.0
    assert desc == "setbacks.front >= 3.0"


def test_leaf_condition_false(sample_normalized_plan):
    result, _, _ = evaluate_threshold(
        {"field": "coverage", "op": "<=", "value": 50.0}, sample_normalized_plan
    )
    assert result is False


def test_empty_applies_when_always_applies(sample_normalized_plan):
    assert evaluate_applies_when({}, sample_normalized_plan) is True


def test_all_combinator(sample_normalized_plan):
    cond = {
        "all": [
            {"field": "plot.area", "op": ">", "value": 100},
            {"field": "setbacks.front", "op": ">=", "value": 3.0},
        ]
    }
    assert evaluate_applies_when(cond, sample_normalized_plan) is True


def test_all_combinator_short_circuits_on_false(sample_normalized_plan):
    cond = {
        "all": [
            {"field": "plot.area", "op": ">", "value": 100000},  # False
            {"field": "setbacks.front", "op": ">=", "value": 3.0},
        ]
    }
    assert evaluate_applies_when(cond, sample_normalized_plan) is False


def test_any_combinator(sample_normalized_plan):
    cond = {
        "any": [
            {"field": "plot.area", "op": ">", "value": 100000},  # False
            {"field": "setbacks.front", "op": ">=", "value": 3.0},  # True
        ]
    }
    assert evaluate_applies_when(cond, sample_normalized_plan) is True


def test_not_combinator(sample_normalized_plan):
    cond = {"not": {"field": "plot.area", "op": ">", "value": 100000}}
    assert evaluate_applies_when(cond, sample_normalized_plan) is True


def test_missing_field_is_indeterminate_not_false(sample_normalized_plan):
    # building.floor_count is Optional and unset on the fixture plan.
    result, observed, _ = evaluate_threshold(
        {"field": "building.floor_count", "op": "<=", "value": 4}, sample_normalized_plan
    )
    assert result is None
    # Unset fields resolve to None and are skipped rather than appended --
    # "no observed evidence for this field" is an empty list here, not None.
    assert observed == []


def test_conflicting_field_is_indeterminate(sample_normalized_plan):
    from backend.schemas.evidence import Conflict

    sample_normalized_plan.setbacks.front = make_value_field(
        0.0, level=ConfidenceLevel.CONFLICTING
    )
    sample_normalized_plan.setbacks.front.conflict = Conflict(
        description="Two dimension lines disagree", conflicting_sources=["a", "b"]
    )
    result = evaluate_applies_when(
        {"field": "setbacks.front", "op": ">=", "value": 3.0}, sample_normalized_plan
    )
    assert result is None


def test_threshold_supports_boolean_group(sample_normalized_plan):
    # threshold intentionally supports boolean groups (not just a single
    # leaf) -- e.g. a real BBMP-style "0.60 m on any one side" requirement
    # needs `any`/`all` here, not just at the applies_when level. See the
    # module docstring on evaluate_threshold.
    result, observed, desc = evaluate_threshold(
        {"all": [{"field": "setbacks.front", "op": ">=", "value": 3.0}]},
        sample_normalized_plan,
    )
    assert result is True
    assert len(observed) == 1
    assert desc == "(setbacks.front >= 3.0)"


def test_threshold_empty_raises(sample_normalized_plan):
    with pytest.raises(ValueError):
        evaluate_threshold({}, sample_normalized_plan)


def test_between_operator(sample_normalized_plan):
    result, _, desc = evaluate_threshold(
        {"field": "far", "op": "between", "value": [1.0, 2.0]}, sample_normalized_plan
    )
    assert result is True
    assert "between" in desc
