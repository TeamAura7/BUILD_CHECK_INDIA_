from __future__ import annotations

import json

import pytest

from backend.compliance.engine import DeterministicRuleEvaluator, JsonFileRuleEngine
from backend.runtime_rules.contracts import RuleContext, RuntimeRuleDefinition
from backend.schemas.enums import ComplianceStatus, ConfidenceLevel
from tests.conftest import make_value_field


def _rule(rule_id, threshold, applies_when=None, citation="Regulation X"):
    return RuntimeRuleDefinition(
        rule_id=rule_id,
        municipality="BBMP",
        citation=citation,
        description=f"Test rule {rule_id}",
        applies_when=applies_when or {},
        threshold=threshold,
        version="1.0.0",
    )


def test_pass(sample_normalized_plan):
    rule = _rule("r1", {"field": "setbacks.front", "op": ">=", "value": 3.0})
    result = DeterministicRuleEvaluator().evaluate(RuleContext(plan=sample_normalized_plan, rule=rule))
    assert result.status == ComplianceStatus.PASS
    assert result.observed_value.value == 3.0
    assert result.citation == "Regulation X"


def test_fail(sample_normalized_plan):
    rule = _rule("r2", {"field": "coverage", "op": "<=", "value": 50.0})
    result = DeterministicRuleEvaluator().evaluate(RuleContext(plan=sample_normalized_plan, rule=rule))
    assert result.status == ComplianceStatus.FAIL


def test_not_applicable(sample_normalized_plan):
    rule = _rule(
        "r3",
        {"field": "road.width", "op": ">=", "value": 12.0},
        applies_when={"field": "plot.area", "op": ">", "value": 100000},
    )
    result = DeterministicRuleEvaluator().evaluate(RuleContext(plan=sample_normalized_plan, rule=rule))
    assert result.status == ComplianceStatus.NOT_APPLICABLE


def test_insufficient_data_missing_field(sample_normalized_plan):
    rule = _rule("r4", {"field": "building.floor_count", "op": "<=", "value": 4})
    result = DeterministicRuleEvaluator().evaluate(RuleContext(plan=sample_normalized_plan, rule=rule))
    assert result.status == ComplianceStatus.INSUFFICIENT_DATA


def test_insufficient_data_indeterminate_applicability(sample_normalized_plan):
    sample_normalized_plan.building.floor_count = None
    rule = _rule(
        "r5",
        {"field": "setbacks.front", "op": ">=", "value": 3.0},
        applies_when={"field": "building.floor_count", "op": ">", "value": 2},
    )
    result = DeterministicRuleEvaluator().evaluate(RuleContext(plan=sample_normalized_plan, rule=rule))
    assert result.status == ComplianceStatus.INSUFFICIENT_DATA


def test_conflicting_evidence(sample_normalized_plan):
    from backend.schemas.evidence import Conflict

    sample_normalized_plan.setbacks.front = make_value_field(0.0, level=ConfidenceLevel.CONFLICTING)
    sample_normalized_plan.setbacks.front.conflict = Conflict(
        description="Two dimension lines disagree on the front setback."
    )
    rule = _rule("r6", {"field": "setbacks.front", "op": ">=", "value": 3.0})
    result = DeterministicRuleEvaluator().evaluate(RuleContext(plan=sample_normalized_plan, rule=rule))
    assert result.status == ComplianceStatus.CONFLICTING_EVIDENCE
    assert "disagree" in result.explanation


def test_requires_review_on_low_confidence(sample_normalized_plan):
    sample_normalized_plan.setbacks.front = make_value_field(3.0, level=ConfidenceLevel.LOW)
    rule = _rule("r7", {"field": "setbacks.front", "op": ">=", "value": 3.0})
    result = DeterministicRuleEvaluator().evaluate(RuleContext(plan=sample_normalized_plan, rule=rule))
    assert result.status == ComplianceStatus.REQUIRES_REVIEW


def test_malformed_threshold_is_insufficient_data_not_a_crash(sample_normalized_plan):
    # A boolean group is now legal for threshold expressions; this test uses
    # an unknown field to exercise malformed-rule handling.
    rule = _rule("r8", {"field": "not.a.real.field", "op": ">=", "value": 3.0})
    result = DeterministicRuleEvaluator().evaluate(RuleContext(plan=sample_normalized_plan, rule=rule))
    assert result.status == ComplianceStatus.INSUFFICIENT_DATA


def test_overall_rollup_fail_dominates(sample_normalized_plan):
    from backend.schemas.compliance import ComplianceResult

    rules = [
        _rule("pass1", {"field": "setbacks.front", "op": ">=", "value": 3.0}),
        _rule("fail1", {"field": "coverage", "op": "<=", "value": 50.0}),
    ]
    evaluator = DeterministicRuleEvaluator()
    results = [evaluator.evaluate(RuleContext(plan=sample_normalized_plan, rule=r)) for r in rules]
    cr = ComplianceResult(plan_id="p", ruleset_id="BBMP", rule_results=results)
    assert cr.overall_status == ComplianceStatus.FAIL


def test_json_file_rule_engine_end_to_end(tmp_path, sample_normalized_plan, monkeypatch):
    from backend.config import Settings

    rules_dir = tmp_path / "runtime_rules" / "BBMP"
    rules_dir.mkdir(parents=True)
    (rules_dir / "rules.json").write_text(
        json.dumps(
            [
                json.loads(_rule("front", {"field": "setbacks.front", "op": ">=", "value": 3.0}).model_dump_json()),
                json.loads(_rule("cov", {"field": "coverage", "op": "<=", "value": 50.0}).model_dump_json()),
            ]
        )
    )
    settings = Settings(runtime_rules_dir=tmp_path / "runtime_rules")
    engine = JsonFileRuleEngine(settings=settings)
    result = engine.evaluate_plan(sample_normalized_plan, "BBMP")
    assert result.overall_status == ComplianceStatus.FAIL
    assert result.count_by_status() == {"PASS": 1, "FAIL": 1}


def test_json_file_rule_engine_no_ruleset_returns_empty(tmp_path, sample_normalized_plan):
    from backend.config import Settings

    settings = Settings(runtime_rules_dir=tmp_path / "empty")
    engine = JsonFileRuleEngine(settings=settings)
    result = engine.evaluate_plan(sample_normalized_plan, "NOWHERE")
    assert result.rule_results == []
    assert result.overall_status == ComplianceStatus.NOT_APPLICABLE


def test_dynamic_threshold_from_other_plan_field(sample_normalized_plan):
    rule = _rule(
        "dynamic-front",
        {"field": "setbacks.front", "op": ">=", "value": {"field": "plot.depth", "factor": 0.12}, "unit": "m"},
        applies_when={"field": "plot.area", "op": ">", "value": 150},
    )
    sample_normalized_plan.plot.depth.value = 10.0
    sample_normalized_plan.plot.area.value = 200.0
    sample_normalized_plan.setbacks.front.value = 1.25
    result = DeterministicRuleEvaluator().evaluate(
        RuleContext(plan=sample_normalized_plan, rule=rule)
    )
    assert result.status == ComplianceStatus.PASS
    assert "1.2" in result.required_value_description


def test_any_side_threshold_does_not_require_both_sides(sample_normalized_plan):
    rule = _rule(
        "one-side",
        {"any": [
            {"field": "setbacks.left", "op": ">=", "value": 1.0, "unit": "m"},
            {"field": "setbacks.right", "op": ">=", "value": 1.0, "unit": "m"},
        ]},
    )
    sample_normalized_plan.setbacks.left = make_value_field(0.0)
    sample_normalized_plan.setbacks.right = make_value_field(1.0)
    result = DeterministicRuleEvaluator().evaluate(
        RuleContext(plan=sample_normalized_plan, rule=rule)
    )
    assert result.status == ComplianceStatus.PASS
    assert len(result.observed_values) == 2


def test_active_ruleset_ignores_drafts(tmp_path, sample_normalized_plan):
    from backend.config import Settings
    rules_dir = tmp_path / "runtime_rules" / "BBMP"
    rules_dir.mkdir(parents=True)
    payload = [
        {
            **_rule("active", {"field": "setbacks.front", "op": ">=", "value": 1.0}).model_dump(),
            "status": "ACTIVE",
        },
        {
            **_rule("draft", {"field": "setbacks.front", "op": ">=", "value": 100.0}).model_dump(),
            "status": "DRAFT",
        },
    ]
    (rules_dir / "rules.json").write_text(json.dumps(payload), encoding="utf-8")
    engine = JsonFileRuleEngine(settings=Settings(runtime_rules_dir=tmp_path / "runtime_rules"))
    result = engine.evaluate_plan(sample_normalized_plan, "BBMP")
    assert [r.rule_id for r in result.rule_results] == ["active"]


def test_normalized_plan_metadata_can_supply_development_area(sample_normalized_plan):
    sample_normalized_plan.metadata["development_area"] = "b"
    # Re-validate through the model so the validator derives the canonical field.
    from backend.schemas.normalized_plan import NormalizedPlan
    rebuilt = NormalizedPlan.model_validate(sample_normalized_plan.model_dump())
    assert rebuilt.development_area.value == "B"


def test_bbmp_table6_is_conditional_on_use_area_and_plot_band(sample_normalized_plan):
    from backend.compliance.engine import JsonFileRuleEngine
    from backend.config import Settings
    from backend.schemas.evidence import Confidence
    from backend.schemas.normalized_plan import NormalizedPlan

    sample_normalized_plan.metadata.update({"building_type": "Residential", "development_area": "A"})
    sample_normalized_plan = NormalizedPlan.model_validate(sample_normalized_plan.model_dump())
    assert sample_normalized_plan.building_use.value == "residential"
    assert sample_normalized_plan.development_area.value == "A"

    result = JsonFileRuleEngine(settings=Settings()).evaluate_plan(sample_normalized_plan, "BBMP")
    by_id = {r.rule_id: r for r in result.rule_results}

    # Plot area 216 m² is in the A-area / residential / <=240 row:
    # coverage <=65%, FAR <=0.75. Coverage passes, FAR fails for the fixture's
    # deliberately excessive FAR of 1.75.
    assert by_id["bbmp-2003-t6-residential-A-coverage-upto240"].status.value == "PASS"
    assert by_id["bbmp-2003-t6-residential-A-far-upto240"].status.value == "FAIL"

    # The commercial rule for the same band must not apply to a residential plan.
    assert by_id["bbmp-2003-t6-commercial-A-coverage-upto240"].status.value == "NOT_APPLICABLE"


def test_bbmp_table6_does_not_use_road_width_for_residential(sample_normalized_plan):
    from backend.compliance.engine import JsonFileRuleEngine
    from backend.config import Settings
    from backend.schemas.normalized_plan import NormalizedPlan

    sample_normalized_plan.metadata.update({"building_type": "Residential", "development_area": "A"})
    sample_normalized_plan.road.width.value = 4.0
    plan = NormalizedPlan.model_validate(sample_normalized_plan.model_dump())
    result = JsonFileRuleEngine(settings=Settings()).evaluate_plan(plan, "BBMP")
    by_id = {r.rule_id: r for r in result.rule_results}
    # Table 6's printed road-width bands belong to the Public/Semi-Public/T&T/PU
    # column, not the residential column. Residential coverage/FAR therefore
    # remain evaluable from use + development area + plot area alone.
    assert by_id["bbmp-2003-t6-residential-A-coverage-upto240"].status.value == "PASS"
    assert by_id["bbmp-2003-t6-residential-A-far-upto240"].status.value == "FAIL"


def test_bbmp_draft_2025_rules_are_not_runtime_authority(sample_normalized_plan):
    from backend.compliance.engine import JsonFileRuleEngine
    from backend.config import Settings

    engine = JsonFileRuleEngine(settings=Settings())
    active = engine.load_ruleset("BBMP")
    assert active
    assert all(r.status == "ACTIVE" for r in active)
    assert all(r.version == "2003.1.0" for r in active)
    assert not any(r.source_document == "Revised setback gazette copy.pdf" for r in active)


def test_any_condition_can_be_decisive_with_missing_alternative(sample_normalized_plan):
    sample_normalized_plan.setbacks.left = make_value_field(1.2)
    sample_normalized_plan.setbacks.right = make_value_field(0.0)
    sample_normalized_plan.setbacks.right = type(sample_normalized_plan.setbacks.right).missing("right side not measured")
    rule = _rule(
        "one-side-missing-alternative",
        {"any": [
            {"field": "setbacks.left", "op": ">=", "value": 1.0, "unit": "m"},
            {"field": "setbacks.right", "op": ">=", "value": 1.0, "unit": "m"},
        ]},
    )
    result = DeterministicRuleEvaluator().evaluate(RuleContext(plan=sample_normalized_plan, rule=rule))
    assert result.status == ComplianceStatus.PASS


def test_bbmp_highrise_site_and_road_rules_are_height_definition_aware(sample_normalized_plan):
    from backend.compliance.engine import JsonFileRuleEngine
    from backend.config import Settings
    from backend.schemas.normalized_plan import NormalizedPlan

    from backend.schemas.evidence import Confidence, ValueField
    sample_normalized_plan.building.floor_count = ValueField[int](value=5, confidence=Confidence(level=ConfidenceLevel.HIGH))
    sample_normalized_plan.building.width = make_value_field(20.0)
    sample_normalized_plan.building.depth = make_value_field(20.0)
    sample_normalized_plan.plot.width = make_value_field(20.0)
    sample_normalized_plan.plot.depth = make_value_field(20.0)
    sample_normalized_plan.road.width = make_value_field(10.0)
    plan = NormalizedPlan.model_validate(sample_normalized_plan.model_dump())
    result = JsonFileRuleEngine(settings=Settings()).evaluate_plan(plan, "BBMP")
    by_id = {r.rule_id: r for r in result.rule_results}
    assert by_id["bbmp-2003-highrise-site-width-21m"].status.value == "FAIL"
    assert by_id["bbmp-2003-highrise-site-depth-21m"].status.value == "FAIL"
    assert by_id["bbmp-2003-highrise-road-12m"].status.value == "FAIL"
