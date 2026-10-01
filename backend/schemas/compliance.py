"""
Compliance result contract — the shape the (future) deterministic
RuleEngine must produce, and the shape the API/frontend consumes.

The LLM never populates ComplianceStatus. Only the deterministic rule
engine (Phase 4+) is permitted to set `status` here.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from backend.schemas.enums import ComplianceStatus
from backend.schemas.evidence import ValueField


class RuleResult(BaseModel):
    """Result of evaluating ONE runtime rule against ONE NormalizedPlan."""

    rule_id: str
    rule_description: str
    source_field: Optional[str] = Field(default=None, description="Canonical plan field/metric governed by the rule.")
    status: ComplianceStatus
    observed_value: Optional[ValueField] = None
    observed_values: list[ValueField] = Field(
        default_factory=list,
        description="All measurements used by the rule; observed_value is retained for UI compatibility.",
    )
    required_value_description: Optional[str] = Field(
        default=None,
        description="Human-readable statement of the threshold, e.g. 'front setback >= 3.0 m'",
    )
    explanation: str
    source_page: Optional[int] = Field(default=None, description="Page number in the cited source document, when known.")
    citation: Optional[str] = Field(
        default=None, description="Byelaw/section reference the rule came from"
    )


class ComplianceResult(BaseModel):
    """Aggregate compliance report for one NormalizedPlan against one ruleset."""

    plan_id: str
    ruleset_id: str
    ruleset_version: Optional[str] = None
    rule_results: list[RuleResult] = Field(default_factory=list)

    @property
    def overall_status(self) -> ComplianceStatus:
        """
        Conservative rollup: any FAIL dominates; else any
        CONFLICTING_EVIDENCE/INSUFFICIENT_DATA/REQUIRES_REVIEW dominates
        over PASS. NOT_APPLICABLE rules are ignored in the rollup.
        """
        statuses = {r.status for r in self.rule_results}
        if ComplianceStatus.FAIL in statuses:
            return ComplianceStatus.FAIL
        for uncertain in (
            ComplianceStatus.CONFLICTING_EVIDENCE,
            ComplianceStatus.INSUFFICIENT_DATA,
            ComplianceStatus.REQUIRES_REVIEW,
        ):
            if uncertain in statuses:
                return uncertain
        if ComplianceStatus.PASS in statuses:
            return ComplianceStatus.PASS
        return ComplianceStatus.NOT_APPLICABLE

    def count_by_status(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.rule_results:
            counts[r.status.value] = counts.get(r.status.value, 0) + 1
        return counts
