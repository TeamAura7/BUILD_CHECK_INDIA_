"""
Stable import point for the compliance module.

Teammates building the deterministic RuleEngine (Phase 4) should import
from `backend.compliance.interfaces`, not reach into `backend.schemas.*`
or `backend.runtime_rules.*` directly, so this module can later add
concrete orchestration (e.g. a RuleEngine registry, caching) without
breaking their imports.
"""

from backend.compliance.engine import DeterministicRuleEvaluator, JsonFileRuleEngine
from backend.runtime_rules.contracts import (
    RuleContext,
    RuleEngine,
    RuleEvaluator,
    RuntimeRuleDefinition,
)
from backend.schemas.compliance import ComplianceResult, RuleResult

__all__ = [
    "RuleContext",
    "RuleEngine",
    "RuleEvaluator",
    "RuntimeRuleDefinition",
    "ComplianceResult",
    "RuleResult",
    # Phase 4: concrete implementations, safe to import from here now.
    "DeterministicRuleEvaluator",
    "JsonFileRuleEngine",
]
