from __future__ import annotations

from backend.schemas.compliance import ComplianceResult, RuleResult
from backend.schemas.enums import ComplianceStatus, ConfidenceLevel
from backend.schemas.evidence import ValueField
from backend.schemas.normalized_plan import NormalizedPlan

from .conftest import make_value_field


def test_normalized_plan_round_trip_serialization(sample_normalized_plan: NormalizedPlan):
    dumped = sample_normalized_plan.model_dump_json()
    restored = NormalizedPlan.model_validate_json(dumped)
    assert restored.plan_id == sample_normalized_plan.plan_id
    assert restored.plot.width.value == sample_normalized_plan.plot.width.value
    assert restored.setbacks.front.value == 3.0


def test_normalized_plan_no_missing_fields_when_fully_populated(
    sample_normalized_plan: NormalizedPlan,
):
    assert sample_normalized_plan.missing_field_names() == []


def test_normalized_plan_reports_missing_fields():
    plan = NormalizedPlan(
        plan_id="plan-002",
        source_document_id="doc-002",
        plot=dict(
            width=make_value_field(12.0),
            depth=ValueField.missing("depth dimension not legible"),
            area=make_value_field(200.0),
        ),
        building=dict(
            width=make_value_field(9.0),
            depth=make_value_field(14.0),
            footprint_area=make_value_field(126.0),
        ),
        road=dict(width=make_value_field(9.0)),
        setbacks=dict(
            front=make_value_field(3.0),
            rear=make_value_field(2.0),
            left=make_value_field(1.5),
            right=make_value_field(1.5),
        ),
        coverage=make_value_field(58.3),
        far=make_value_field(1.75),
    )
    missing = plan.missing_field_names()
    assert missing == ["plot.depth"]


def test_setback_section_as_measurements(sample_normalized_plan: NormalizedPlan):
    measurements = sample_normalized_plan.setbacks.as_measurements()
    sides = {m.side for m in measurements}
    assert sides == {"front", "rear", "left", "right"}


def test_compliance_result_overall_status_fail_dominates(sample_normalized_plan):
    result = ComplianceResult(
        plan_id=sample_normalized_plan.plan_id,
        ruleset_id="bbmp-2026",
        rule_results=[
            RuleResult(
                rule_id="r1",
                rule_description="Front setback",
                status=ComplianceStatus.PASS,
                explanation="Front setback meets minimum",
            ),
            RuleResult(
                rule_id="r2",
                rule_description="Rear setback",
                status=ComplianceStatus.FAIL,
                explanation="Rear setback below minimum",
            ),
        ],
    )
    assert result.overall_status == ComplianceStatus.FAIL


def test_compliance_result_overall_status_uncertain_dominates_over_pass(sample_normalized_plan):
    result = ComplianceResult(
        plan_id=sample_normalized_plan.plan_id,
        ruleset_id="bbmp-2026",
        rule_results=[
            RuleResult(
                rule_id="r1",
                rule_description="Front setback",
                status=ComplianceStatus.PASS,
                explanation="ok",
            ),
            RuleResult(
                rule_id="r2",
                rule_description="FAR",
                status=ComplianceStatus.INSUFFICIENT_DATA,
                explanation="floor count not extracted",
            ),
        ],
    )
    assert result.overall_status == ComplianceStatus.INSUFFICIENT_DATA


def test_compliance_result_all_pass():
    result = ComplianceResult(
        plan_id="plan-003",
        ruleset_id="bbmp-2026",
        rule_results=[
            RuleResult(rule_id="r1", rule_description="x", status=ComplianceStatus.PASS, explanation="ok"),
            RuleResult(rule_id="r2", rule_description="y", status=ComplianceStatus.NOT_APPLICABLE, explanation="n/a"),
        ],
    )
    assert result.overall_status == ComplianceStatus.PASS
    assert result.count_by_status()["PASS"] == 1
