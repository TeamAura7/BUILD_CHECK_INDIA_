"""
backend/compliance/explainer.py
================================
Plain-English narration of an ALREADY-COMPUTED ComplianceResult, using
exactly ONE Groq call regardless of how many rules were evaluated.

This module never sets, reads-to-decide, or overrides ComplianceStatus —
DeterministicRuleEvaluator (backend/compliance/engine.py) already decided
every status with zero LLM calls, before this module is ever invoked. The
LLM's only job here is to turn a status that's already final into a
sentence a non-technical applicant can read, plus (for anything that
isn't PASS/NOT_APPLICABLE) one concrete suggestion.

Cost model, for a report with N rules:
    OLD (backend.tools.run_full_compliance's per-field RASE drafting):
        up to N Groq calls, EVERY time a plan is checked
    NEW (this module, used after a promoted ruleset exists):
        0 Groq calls to check compliance (JsonFileRuleEngine is pure rule
        evaluation) + exactly 1 Groq call to explain the result, however
        many rules N is
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from backend.config import Settings, get_settings
from backend.schemas.compliance import ComplianceResult
from backend.schemas.enums import ComplianceStatus
from backend.tools.logging_config import get_logger

logger = get_logger(__name__)


class RuleExplanation(BaseModel):
    rule_id: str
    status: ComplianceStatus
    plain_explanation: str
    suggestion: Optional[str] = None


class ComplianceExplanation(BaseModel):
    plan_id: str
    overall_summary: str
    rule_explanations: list[RuleExplanation] = Field(default_factory=list)


_EXPLAIN_SYSTEM_PROMPT = """You explain building-compliance check results to an architect or \
applicant in plain English. Every status below (PASS/FAIL/NOT_APPLICABLE/INSUFFICIENT_DATA/ \
CONFLICTING_EVIDENCE/REQUIRES_REVIEW) was already decided by a deterministic rule engine and \
is ground truth — you are NOT deciding compliance and must not contradict, soften, or upgrade \
any status. For each rule write one or two plain-English sentences explaining what was found. \
For anything that is not PASS or NOT_APPLICABLE, add one concrete, actionable suggestion \
(e.g. "increase the front setback by at least 0.3 m" or "have a surveyor re-measure the rear \
setback since the two extraction methods disagree"). For PASS/NOT_APPLICABLE, omit the \
suggestion (use null). Only use numbers that appear in the input below — never invent one.

Respond with ONLY a single JSON object (no markdown fences, no preamble), matching exactly:
{
  "overall_summary": "<2-3 sentence plain-English summary of the whole result>",
  "rule_explanations": [
    {"rule_id": "<id>", "plain_explanation": "<1-2 sentences>", "suggestion": "<actionable fix or null>"}
  ]
}
"""


def _build_user_prompt(result: ComplianceResult) -> str:
    lines = [
        f"Plan: {result.plan_id}",
        f"Ruleset: {result.ruleset_id} (v{result.ruleset_version})",
        f"Overall status (already decided, do not change): {result.overall_status.value}",
        "",
        "Per-rule results:",
    ]
    for r in result.rule_results:
        observed = r.observed_value.value if r.observed_value else "n/a"
        lines.append(
            f"- rule_id={r.rule_id} status={r.status.value} "
            f"description={r.rule_description!r} required={r.required_value_description!r} "
            f"observed={observed} citation={r.citation!r} "
            f"engine_explanation={r.explanation!r}"
        )
    return "\n".join(lines)


def explain_compliance(
    result: ComplianceResult,
    settings: Optional[Settings] = None,
) -> ComplianceExplanation:
    """
    Narrate an already-computed ComplianceResult in exactly ONE Groq call.

    Falls back to the engine's own (already human-readable) `explanation`
    field per rule if the LLM call fails for any reason — a missing/broken
    LLM should never block the person from seeing their compliance result,
    only the friendlier prose on top of it.
    """
    settings = settings or get_settings()

    fallback = ComplianceExplanation(
        plan_id=result.plan_id,
        overall_summary=f"Overall status: {result.overall_status.value}.",
        rule_explanations=[
            RuleExplanation(
                rule_id=r.rule_id,
                status=r.status,
                plain_explanation=r.explanation,
                suggestion=None,
            )
            for r in result.rule_results
        ],
    )

    if not result.rule_results:
        return fallback

    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY not set; returning engine explanations without LLM narration.")
        return fallback

    from backend.rase.extractor import _call_groq, _extract_json

    try:
        raw = _call_groq(_EXPLAIN_SYSTEM_PROMPT, _build_user_prompt(result), settings)
        payload = _extract_json(raw)
    except Exception as exc:
        logger.error("explain_compliance: LLM call/parse failed (%s); using engine explanations.", exc)
        return fallback

    explanations_by_id = {e.get("rule_id"): e for e in payload.get("rule_explanations", [])}

    rule_explanations = []
    for r in result.rule_results:
        e = explanations_by_id.get(r.rule_id, {})
        rule_explanations.append(
            RuleExplanation(
                rule_id=r.rule_id,
                status=r.status,
                plain_explanation=e.get("plain_explanation") or r.explanation,
                suggestion=e.get("suggestion"),
            )
        )

    return ComplianceExplanation(
        plan_id=result.plan_id,
        overall_summary=payload.get("overall_summary") or fallback.overall_summary,
        rule_explanations=rule_explanations,
    )
