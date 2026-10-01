"""
backend/compliance/engine.py
=============================
Concrete deterministic RuleEngine / RuleEvaluator implementation.

This is the ONLY place in the codebase permitted to produce a
ComplianceStatus (see ARCHITECTURE.md: "The LLM is never the compliance
authority"). It reads:
  - a NormalizedPlan (from spatial reasoning / CV extraction)
  - RuntimeRuleDefinition records (authored by hand, or drafted by RASE
    from RAG-retrieved regulation text and explicitly promoted by a human
    reviewer via backend.rase.extractor.promote_draft — see that module)

and evaluates the plan's fields against each rule's applies_when/threshold
(backend/rase/schema.py) with zero network/LLM calls, per the
RuleEvaluator contract.

Status derivation for one rule:
    applies_when == False           -> NOT_APPLICABLE
    applies_when indeterminate      -> INSUFFICIENT_DATA
                                        (can't tell if the rule even applies)
    threshold's observed field
        missing / unresolvable      -> INSUFFICIENT_DATA
        CONFLICTING                 -> CONFLICTING_EVIDENCE
        LOW confidence               -> REQUIRES_REVIEW
                                        (never silently PASS/FAIL on a
                                        low-confidence measurement)
        otherwise, threshold True   -> PASS
        otherwise, threshold False  -> FAIL
"""

from __future__ import annotations

import json
from typing import Optional

from backend.config import Settings, get_settings
from backend.rase.schema import evaluate_applies_when, evaluate_threshold, parse_condition
from backend.runtime_rules.contracts import (
    RuleContext,
    RuleEngine,
    RuleEvaluator,
    RuntimeRuleDefinition,
)
from backend.schemas.compliance import ComplianceResult, RuleResult
from backend.schemas.enums import ComplianceStatus, ConfidenceLevel
from backend.schemas.normalized_plan import NormalizedPlan
from backend.tools.logging_config import get_logger

logger = get_logger(__name__)


class DeterministicRuleEvaluator(RuleEvaluator):
    """Evaluates ONE RuntimeRuleDefinition against ONE NormalizedPlan.
    No network/LLM calls — every branch here is pure function of the
    already-extracted plan and the already-promoted rule."""

    def evaluate(self, context: RuleContext) -> RuleResult:
        plan = context.plan
        rule = context.rule

        try:
            applies = evaluate_applies_when(rule.applies_when, plan)
        except Exception as exc:
            logger.error("Rule %s: malformed applies_when: %s", rule.rule_id, exc)
            return RuleResult(
                rule_id=rule.rule_id,
                rule_description=rule.description,
                source_field=rule.target,
                status=ComplianceStatus.INSUFFICIENT_DATA,
                explanation=f"Could not evaluate applicability condition: {exc}",
                source_page=rule.source_page,
                citation=rule.citation,
            )

        if applies is False:
            return RuleResult(
                rule_id=rule.rule_id,
                rule_description=rule.description,
                source_field=rule.target,
                status=ComplianceStatus.NOT_APPLICABLE,
                explanation="This plan does not meet the rule's applicability condition.",
                source_page=rule.source_page,
                citation=rule.citation,
            )

        if applies is None:
            return RuleResult(
                rule_id=rule.rule_id,
                rule_description=rule.description,
                source_field=rule.target,
                status=ComplianceStatus.INSUFFICIENT_DATA,
                explanation=(
                    "Whether this rule applies to the plan could not be determined "
                    "because a field needed for the applicability check is missing "
                    "or conflicting in the extracted plan."
                ),
                source_page=rule.source_page,
                citation=rule.citation,
            )

        try:
            result, observed_values, requirement_desc = evaluate_threshold(rule.threshold, plan)
        except Exception as exc:
            logger.error("Rule %s: malformed threshold: %s", rule.rule_id, exc)
            return RuleResult(
                rule_id=rule.rule_id,
                rule_description=rule.description,
                source_field=rule.target,
                status=ComplianceStatus.INSUFFICIENT_DATA,
                explanation=f"Could not evaluate threshold condition: {exc}",
                source_page=rule.source_page,
                citation=rule.citation,
            )

        # A threshold can inspect more than one measurement (e.g. an
        # "either left or right" setback requirement).  Preserve the old
        # observed_value field for UI compatibility and expose all evidence
        # through observed_values.
        observed = observed_values[0] if observed_values else None

        # Missing fields are not automatically fatal. Boolean conditions use
        # three-valued logic, so an `any` condition can be conclusively true
        # even if one alternative side is missing, and an `all` condition can
        # be conclusively false when one required condition fails. Only a
        # genuinely indeterminate threshold produces INSUFFICIENT_DATA.
        conflicts = [vf for vf in observed_values if vf.confidence.level == ConfidenceLevel.CONFLICTING]
        if conflicts:
            descriptions = [vf.conflict.description for vf in conflicts if vf.conflict]
            return RuleResult(
                rule_id=rule.rule_id,
                rule_description=rule.description,
                source_field=rule.target,
                status=ComplianceStatus.CONFLICTING_EVIDENCE,
                observed_value=observed,
                observed_values=observed_values,
                required_value_description=requirement_desc,
                explanation=(
                    "; ".join(descriptions) if descriptions
                    else "Conflicting extracted values for this measurement."
                ),
                source_page=rule.source_page,
                citation=rule.citation,
            )

        if any(vf.confidence.level == ConfidenceLevel.LOW for vf in observed_values if vf.value is not None):
            return RuleResult(
                rule_id=rule.rule_id,
                rule_description=rule.description,
                source_field=rule.target,
                status=ComplianceStatus.REQUIRES_REVIEW,
                observed_value=observed,
                observed_values=observed_values,
                required_value_description=requirement_desc,
                explanation=(
                    "At least one measurement used by this rule has LOW extraction "
                    "confidence; a human should verify it before treating the result "
                    "as pass or fail."
                ),
                source_page=rule.source_page,
                citation=rule.citation,
            )

        if result is None:
            return RuleResult(
                rule_id=rule.rule_id,
                rule_description=rule.description,
                source_field=rule.target,
                status=ComplianceStatus.INSUFFICIENT_DATA,
                observed_value=observed,
                observed_values=observed_values,
                required_value_description=requirement_desc,
                explanation=(
                    "The threshold is indeterminate because one or more measurements "
                    "needed to decide the condition are missing."
                ),
                source_page=rule.source_page,
                citation=rule.citation,
            )

        status = ComplianceStatus.PASS if result else ComplianceStatus.FAIL
        explanation = (
            f"Observed {observed.value} {'satisfies' if result else 'does not satisfy'} "
            f"requirement '{requirement_desc}'."
        )
        return RuleResult(
            rule_id=rule.rule_id,
            rule_description=rule.description,
            source_field=rule.target,
            status=status,
            observed_value=observed,
            observed_values=observed_values,
            required_value_description=requirement_desc,
            explanation=explanation,
            source_page=rule.source_page,
            citation=rule.citation,
        )


class JsonFileRuleEngine(RuleEngine):
    """
    RuleEngine that loads the live (promoted) ruleset for a municipality
    from data/runtime_rules/<MUNICIPALITY>/rules.json and evaluates a
    NormalizedPlan against every rule in it.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        evaluator: Optional[RuleEvaluator] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.evaluator = evaluator or DeterministicRuleEvaluator()

    def load_ruleset(
        self, municipality: str, version: Optional[str] = None
    ) -> list[RuntimeRuleDefinition]:
        path = self.settings.runtime_rules_path(municipality)
        if not path.exists():
            logger.warning("No runtime ruleset found for %s at %s", municipality, path)
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"Ruleset {path} must contain a JSON list")

        rules = [RuntimeRuleDefinition.model_validate(r) for r in raw]
        municipality = municipality.upper()
        rules = [r for r in rules if r.municipality == municipality]
        # Validate the deterministic DSL at load time. Draft/superseded rules
        # are allowed to remain in the registry for auditability but are never
        # parsed or executed by the live engine.
        for r in rules:
            if r.status == "ACTIVE":
                if not r.threshold:
                    raise ValueError(f"Active rule {r.rule_id} has an empty threshold")
                if r.applies_when:
                    parse_condition(r.applies_when)
                parse_condition(r.threshold)
        rules = [r for r in rules if r.status == "ACTIVE"]
        if version:
            rules = [r for r in rules if r.version == version]

        ids = [r.rule_id for r in rules]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate active rule_id in {path}")
        return sorted(rules, key=lambda r: (r.priority, r.rule_id))

    def evaluate_plan(self, plan: NormalizedPlan, municipality: str) -> ComplianceResult:
        rules = self.load_ruleset(municipality)
        rule_results = [
            self.evaluator.evaluate(RuleContext(plan=plan, rule=rule)) for rule in rules
        ]
        return ComplianceResult(
            plan_id=plan.plan_id,
            ruleset_id=municipality.upper(),
            ruleset_version=rules[0].version if rules else None,
            rule_results=rule_results,
        )
