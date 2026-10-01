"""
backend/rase/schema.py
=======================
RASE (Regulation -> Applies-when / Structured / Evaluatable rule)
condition schema.

`RuntimeRuleDefinition.applies_when` and `.threshold`
(backend/runtime_rules/contracts.py) are intentionally untyped
`dict[str, Any]` in Phase 1 — "schema owned by RASE". This module is that
schema, plus the deterministic evaluator for it.

Design goals:
- JSON-serializable (so it round-trips through data/runtime_rules/*.json
  with no custom parsing).
- Deterministic: evaluating a Condition against a NormalizedPlan NEVER
  calls an LLM and NEVER guesses. A field that is MISSING or CONFLICTING
  makes the condition indeterminate (None), not False.
- Small: one leaf shape (FieldCondition) plus boolean combinators
  (all / any / not), which is enough to express every BBMP-style
  threshold ("front setback >= 3.0 m for plots > 300 sq m") without a
  general-purpose expression language.

Condition JSON shape:
    Leaf:   {"field": "setbacks.front", "op": ">=", "value": 3.0}
    Group:  {"all": [Condition, ...]}
          | {"any": [Condition, ...]}
          | {"not": Condition}
    Empty dict {} means "always applies" (used for applies_when when a
    rule has no applicability precondition, e.g. it applies to every plot).

Field paths resolve against NormalizedPlan (backend/schemas/normalized_plan.py):
    plot.width | plot.depth | plot.area
    building.width | building.depth | building.footprint_area | building.floor_count
    road.width
    setbacks.front | setbacks.rear | setbacks.left | setbacks.right
    coverage | far
    building_height_estimated
        (NOTE: this one has NO dot -- it lives directly on NormalizedPlan,
        not nested under `building`, so its field path is the flat name
        "building_height_estimated", not "building.height". Many Indian
        DCRs tier setback/coverage/FAR thresholds by building height, so
        this path exists specifically so applies_when conditions can gate
        a rule on it, e.g. {"field": "building_height_estimated", "op":
        "<=", "value": 9.5} for a "buildings up to 9.5 m" band.)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, model_validator

from backend.schemas.enums import ConfidenceLevel
from backend.schemas.evidence import ValueField
from backend.schemas.normalized_plan import NormalizedPlan

# ─── Field path resolution ──────────────────────────────────────────────────

#: Every field path a Condition is permitted to reference. Kept as an
#: explicit allow-list (rather than getattr-chasing arbitrary attributes)
#: so a malformed/hallucinated RASE draft fails loudly instead of reaching
#: into unrelated model internals.
FIELD_PATHS: dict[str, tuple[str, ...]] = {
    "plot.width": ("plot", "width"),
    "plot.depth": ("plot", "depth"),
    "plot.area": ("plot", "area"),
    "building.width": ("building", "width"),
    "building.depth": ("building", "depth"),
    "building.footprint_area": ("building", "footprint_area"),
    "building.floor_count": ("building", "floor_count"),
    "road.width": ("road", "width"),
    "setbacks.front": ("setbacks", "front"),
    "setbacks.rear": ("setbacks", "rear"),
    "setbacks.left": ("setbacks", "left"),
    "setbacks.right": ("setbacks", "right"),
    "coverage": ("coverage",),
    "far": ("far",),
    # Flat path (no "building." prefix) -- see module docstring above.
    "building_height_estimated": ("building_height_estimated",),
    "building_height_excluding_stilt": ("building_height_excluding_stilt",),
    # Plain metadata is used only for facts that cannot be inferred safely
    # from geometry (e.g. BBMP development-area class). Missing metadata is
    # indeterminate, never guessed.
    "development_area": ("development_area",),
    "building_use": ("building_use",),
}


class FieldResolutionError(ValueError):
    """Raised when a Condition references a field path outside FIELD_PATHS."""


def resolve_field(plan: NormalizedPlan, field_path: str) -> Optional[ValueField]:
    """Resolve a dotted field path to its ValueField on `plan`, or raise
    FieldResolutionError if the path isn't one RASE is allowed to express."""
    if field_path not in FIELD_PATHS:
        raise FieldResolutionError(
            f"Unknown field path {field_path!r}. Allowed: {sorted(FIELD_PATHS)}"
        )
    obj: Any = plan
    for attr in FIELD_PATHS[field_path]:
        obj = getattr(obj, attr)
        if obj is None:
            return None
    return obj


# ─── Operators ───────────────────────────────────────────────────────────────


class Op(str, Enum):
    EQ = "=="
    NE = "!="
    LT = "<"
    LTE = "<="
    GT = ">"
    GTE = ">="
    IN = "in"
    BETWEEN = "between"  # inclusive, uses value=[low, high]


def _apply_op(op: Op, actual: Any, value: Any) -> bool:
    if op == Op.EQ:
        return actual == value
    if op == Op.NE:
        return actual != value
    if op == Op.LT:
        return actual < value
    if op == Op.LTE:
        return actual <= value
    if op == Op.GT:
        return actual > value
    if op == Op.GTE:
        return actual >= value
    if op == Op.IN:
        return actual in value
    if op == Op.BETWEEN:
        low, high = value
        return low <= actual <= high
    raise ValueError(f"Unsupported operator: {op}")  # pragma: no cover


_OP_DESCRIPTIONS = {
    Op.EQ: "==",
    Op.NE: "!=",
    Op.LT: "<",
    Op.LTE: "<=",
    Op.GT: ">",
    Op.GTE: ">=",
    Op.IN: "in",
    Op.BETWEEN: "between",
}


# ─── Condition models ───────────────────────────────────────────────────────


class FieldCondition(BaseModel):
    """A single leaf comparison against one NormalizedPlan field."""

    field: str
    op: Op
    value: Any
    unit: Optional[str] = Field(
        default=None,
        description="Informational only (e.g. 'm', 'sq_m', '%'); NormalizedPlan fields are "
        "already in canonical units, so this does not trigger conversion — it exists so a "
        "RASE draft's threshold is self-documenting and citable.",
    )

    def describe(self) -> str:
        unit_suffix = f" {self.unit}" if self.unit else ""
        if isinstance(self.value, dict) and "field" in self.value:
            factor = self.value.get("factor", 1.0)
            rhs = f"{factor:g} × {self.value['field']}"
        else:
            rhs = str(self.value)
        return f"{self.field} {_OP_DESCRIPTIONS[self.op]} {rhs}{unit_suffix}"

    def _rhs(self, plan: NormalizedPlan) -> Any:
        """Resolve a literal RHS or a deterministic reference expression.

        Supported reference expression:
            {"field": "plot.depth", "factor": 0.12}
        which means 12% of the plot depth.  This is deliberately tiny and
        closed-world: no arbitrary Python/expression evaluation is allowed.
        """
        if not isinstance(self.value, dict) or "field" not in self.value:
            return self.value
        ref = self.value["field"]
        vf = resolve_field(plan, ref)
        if vf is None or vf.value is None:
            return None
        if vf.confidence.level in (ConfidenceLevel.MISSING, ConfidenceLevel.CONFLICTING):
            return None
        factor = self.value.get("factor", 1.0)
        offset = self.value.get("offset", 0.0)
        return vf.value * factor + offset

    def resolved_value(self, plan: NormalizedPlan) -> Any:
        return self._rhs(plan)

    def evaluate(self, plan: NormalizedPlan) -> Optional[bool]:
        """Returns True/False, or None when either side is indeterminate."""
        vf = resolve_field(plan, self.field)
        if vf is None or vf.value is None:
            return None
        if vf.confidence.level in (ConfidenceLevel.MISSING, ConfidenceLevel.CONFLICTING):
            return None
        rhs = self._rhs(plan)
        if rhs is None:
            return None
        return _apply_op(self.op, vf.value, rhs)


class ConditionGroup(BaseModel):
    """Boolean combinator over child Conditions. Exactly one of all_/any_/not_ is set."""

    all_: Optional[list["Condition"]] = Field(default=None, alias="all")
    any_: Optional[list["Condition"]] = Field(default=None, alias="any")
    not_: Optional["Condition"] = Field(default=None, alias="not")

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _exactly_one_combinator(self) -> "ConditionGroup":
        set_fields = [f for f in (self.all_, self.any_, self.not_) if f is not None]
        if len(set_fields) != 1:
            raise ValueError("ConditionGroup must set exactly one of all/any/not")
        return self

    def describe(self) -> str:
        if self.all_ is not None:
            return "(" + " AND ".join(c.describe() for c in self.all_) + ")"
        if self.any_ is not None:
            return "(" + " OR ".join(c.describe() for c in self.any_) + ")"
        return f"NOT ({self.not_.describe()})"

    def evaluate(self, plan: NormalizedPlan) -> Optional[bool]:
        """Three-valued logic: an indeterminate (None) child propagates
        unless the result is already decided by other children (e.g. one
        False child makes an `all` False even if a sibling is indeterminate;
        one True child makes an `any` True even if a sibling is indeterminate)."""
        if self.all_ is not None:
            results = [c.evaluate(plan) for c in self.all_]
            if any(r is False for r in results):
                return False
            if any(r is None for r in results):
                return None
            return True
        if self.any_ is not None:
            results = [c.evaluate(plan) for c in self.any_]
            if any(r is True for r in results):
                return True
            if any(r is None for r in results):
                return None
            return False
        # not_
        inner = self.not_.evaluate(plan)
        return None if inner is None else (not inner)


Condition = Union[FieldCondition, ConditionGroup]
ConditionGroup.model_rebuild()


def parse_condition(data: dict[str, Any]) -> Condition:
    """Parse a raw dict (as stored in RuntimeRuleDefinition.applies_when /
    .threshold) into a Condition. Empty dict is not a valid Condition here —
    callers should special-case {} as "always applies" before calling this
    (see evaluate_applies_when / evaluate_threshold below)."""
    if "field" in data:
        return FieldCondition.model_validate(data)
    return ConditionGroup.model_validate(data)


def evaluate_applies_when(applies_when: dict[str, Any], plan: NormalizedPlan) -> Optional[bool]:
    """Evaluate a RuntimeRuleDefinition.applies_when dict. Empty dict ({})
    means the rule always applies."""
    if not applies_when:
        return True
    condition = parse_condition(applies_when)
    return condition.evaluate(plan)


def _collect_fields(condition: Condition) -> list[str]:
    if isinstance(condition, FieldCondition):
        return [condition.field]
    fields: list[str] = []
    for child in (condition.all_ or condition.any_ or ([condition.not_] if condition.not_ else [])):
        fields.extend(_collect_fields(child))
    return fields


def evaluate_threshold(
    threshold: dict[str, Any], plan: NormalizedPlan
) -> tuple[Optional[bool], list[ValueField], str]:
    """Evaluate a threshold condition.

    Unlike the old Phase-1 contract, threshold may be a boolean group. This
    is required for legal statements such as "0.60 m on any one side".
    The caller receives every observed ValueField participating in the rule.
    """
    if not threshold:
        raise ValueError("threshold must not be empty")
    condition = parse_condition(threshold)
    result = condition.evaluate(plan)
    observed: list[ValueField] = []
    for field in _collect_fields(condition):
        vf = resolve_field(plan, field)
        if vf is not None:
            observed.append(vf)

    def describe(c: Condition) -> str:
        if isinstance(c, FieldCondition):
            rhs = c.resolved_value(plan)
            if rhs is None:
                return c.describe()
            unit = f" {c.unit}" if c.unit else ""
            return f"{c.field} {_OP_DESCRIPTIONS[c.op]} {rhs}{unit}"
        return c.describe()

    return result, observed, describe(condition)
