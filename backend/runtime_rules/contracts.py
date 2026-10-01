"""
Runtime rule contracts for the deterministic BBMP compliance engine.

Rules are legal-data records, not Python constants.  The engine loads them
from data/runtime_rules/<municipality>/rules.json and evaluates them against
a NormalizedPlan without an LLM or network call.

A rule carries provenance (source document/clause, effective date and status)
so a draft Gazette or an unverified rule cannot silently become authoritative.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from backend.schemas.compliance import ComplianceResult, RuleResult
from backend.schemas.normalized_plan import NormalizedPlan


class RuntimeRuleDefinition(BaseModel):
    rule_id: str
    municipality: str
    description: str
    target: Optional[str] = None
    applies_when: dict[str, Any] = Field(default_factory=dict)
    threshold: dict[str, Any] = Field(default_factory=dict)
    version: str = "1.0.0"

    # Provenance / lifecycle.  ACTIVE rules are the only rules evaluated by
    # default. DRAFT/SUPERSEDED/RETIRED rules remain auditable but inert.
    citation: Optional[str] = None
    source_document: Optional[str] = None
    source_clause: Optional[str] = None
    source_page: Optional[int] = None
    effective_date: Optional[date] = None
    status: Literal["ACTIVE", "DRAFT", "SUPERSEDED", "RETIRED"] = "ACTIVE"
    priority: int = 0

    @field_validator("municipality")
    @classmethod
    def _upper_municipality(cls, value: str) -> str:
        return value.strip().upper()


class RuleContext(BaseModel):
    plan: NormalizedPlan
    rule: RuntimeRuleDefinition
    extra: dict[str, Any] = Field(default_factory=dict)


class RuleEvaluator(ABC):
    """Pure deterministic evaluator for one rule. No LLM/network access."""

    @abstractmethod
    def evaluate(self, context: RuleContext) -> RuleResult:
        ...


class RuleEngine(ABC):
    """Contract implemented by the deterministic municipality rule engine."""

    @abstractmethod
    def load_ruleset(
        self, municipality: str, version: Optional[str] = None
    ) -> list[RuntimeRuleDefinition]:
        ...

    @abstractmethod
    def evaluate_plan(
        self, plan: NormalizedPlan, municipality: str
    ) -> ComplianceResult:
        ...
