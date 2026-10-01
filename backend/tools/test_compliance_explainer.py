from __future__ import annotations

import json

import backend.compliance.explainer as explainer_mod
from backend.compliance.explainer import explain_compliance
from backend.config import Settings
from backend.schemas.compliance import ComplianceResult, RuleResult
from backend.schemas.enums import ComplianceStatus


def _result_with(*statuses: ComplianceStatus) -> ComplianceResult:
    rule_results = [
        RuleResult(
            rule_id=f"r{i}",
            rule_description=f"Test rule {i}",
            status=s,
            required_value_description="setbacks.front >= 3.0 m",
            explanation=f"engine explanation for r{i}",
            citation="Regulation 5",
        )
        for i, s in enumerate(statuses)
    ]
    return ComplianceResult(plan_id="plan-1", ruleset_id="BBMP", rule_results=rule_results)


FAKE_EXPLANATION_JSON = json.dumps(
    {
        "overall_summary": "One rule failed and needs a bigger setback.",
        "rule_explanations": [
            {
                "rule_id": "r0",
                "plain_explanation": "The front setback is too small.",
                "suggestion": "Increase the front setback by 0.5 m.",
            },
            {
                "rule_id": "r1",
                "plain_explanation": "This passed comfortably.",
                "suggestion": None,
            },
        ],
    }
)


def test_explain_compliance_single_llm_call(monkeypatch):
    call_count = {"n": 0}

    def fake_call_groq(system, user, settings):
        call_count["n"] += 1
        return FAKE_EXPLANATION_JSON

    monkeypatch.setattr("backend.rase.extractor._call_groq", fake_call_groq)

    result = _result_with(ComplianceStatus.FAIL, ComplianceStatus.PASS)
    settings = Settings(groq_api_key="fake-key")

    explanation = explain_compliance(result, settings=settings)

    assert call_count["n"] == 1  # exactly one LLM call regardless of rule count
    assert "failed" in explanation.overall_summary
    assert explanation.rule_explanations[0].suggestion == "Increase the front setback by 0.5 m."
    assert explanation.rule_explanations[1].suggestion is None


def test_explain_compliance_many_rules_still_one_call(monkeypatch):
    call_count = {"n": 0}

    def fake_call_groq(system, user, settings):
        call_count["n"] += 1
        return json.dumps({"overall_summary": "ok", "rule_explanations": []})

    monkeypatch.setattr("backend.rase.extractor._call_groq", fake_call_groq)

    result = _result_with(*([ComplianceStatus.PASS] * 13))
    settings = Settings(groq_api_key="fake-key")

    explain_compliance(result, settings=settings)
    assert call_count["n"] == 1


def test_explain_compliance_falls_back_without_api_key():
    result = _result_with(ComplianceStatus.FAIL)
    settings = Settings(groq_api_key="")

    explanation = explain_compliance(result, settings=settings)
    assert explanation.rule_explanations[0].plain_explanation == "engine explanation for r0"
    assert explanation.rule_explanations[0].suggestion is None


def test_explain_compliance_falls_back_on_llm_error(monkeypatch):
    def broken_call_groq(system, user, settings):
        raise RuntimeError("groq is down")

    monkeypatch.setattr("backend.rase.extractor._call_groq", broken_call_groq)

    result = _result_with(ComplianceStatus.FAIL)
    settings = Settings(groq_api_key="fake-key")

    explanation = explain_compliance(result, settings=settings)
    assert explanation.rule_explanations[0].plain_explanation == "engine explanation for r0"


def test_explain_compliance_empty_rule_results_skips_llm(monkeypatch):
    call_count = {"n": 0}

    def fake_call_groq(system, user, settings):
        call_count["n"] += 1
        return "{}"

    monkeypatch.setattr("backend.rase.extractor._call_groq", fake_call_groq)

    result = ComplianceResult(plan_id="p", ruleset_id="BBMP", rule_results=[])
    settings = Settings(groq_api_key="fake-key")

    explanation = explain_compliance(result, settings=settings)
    assert call_count["n"] == 0
    assert explanation.rule_explanations == []


def test_explain_compliance_never_overrides_status(monkeypatch):
    """Even if the LLM's JSON tried to smuggle in a status change, the
    RuleExplanation.status always comes from the already-computed
    RuleResult, never from the model."""

    def fake_call_groq(system, user, settings):
        return json.dumps(
            {
                "overall_summary": "looks fine actually",
                "rule_explanations": [
                    {"rule_id": "r0", "plain_explanation": "ignore the fail", "status": "PASS"}
                ],
            }
        )

    monkeypatch.setattr("backend.rase.extractor._call_groq", fake_call_groq)

    result = _result_with(ComplianceStatus.FAIL)
    settings = Settings(groq_api_key="fake-key")

    explanation = explain_compliance(result, settings=settings)
    assert explanation.rule_explanations[0].status == ComplianceStatus.FAIL
