"""
Evaluate a NormalizedPlan (JSON, e.g. the output of run_pipeline.py)
against a municipality's live runtime ruleset and print the
ComplianceResult.

Usage:
    python -m backend.tools.run_compliance plan.json BBMP
    python -m backend.tools.run_compliance plan.json BBMP --output result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.compliance.engine import JsonFileRuleEngine
from backend.schemas.normalized_plan import NormalizedPlan
from backend.schemas.evidence import Confidence, ValueField
from backend.schemas.enums import ConfidenceLevel
from backend.schemas.units import UnitValue


def _print_summary(result) -> None:
    print(f"Plan {result.plan_id}  vs  {result.ruleset_id} (v{result.ruleset_version})")
    print(f"Overall: {result.overall_status.value}")
    print(f"Counts: {result.count_by_status()}\n")
    for r in result.rule_results:
        observed = r.observed_value.value if r.observed_value else "n/a"
        print(f"[{r.status.value:22s}] {r.rule_id}: {r.rule_description}")
        print(f"    required: {r.required_value_description or 'n/a'}   observed: {observed}")
        print(f"    {r.explanation}")
        if r.citation:
            print(f"    citation: {r.citation}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan_json", help="Path to a NormalizedPlan JSON file")
    parser.add_argument("municipality", help="e.g. BBMP")
    parser.add_argument("--output", help="Optional path to write the ComplianceResult JSON")
    parser.add_argument("--building-use", default=None, help="Explicit building-use classification, e.g. residential")
    parser.add_argument("--development-area", choices=["A", "B", "C"], default=None)
    args = parser.parse_args()

    plan_path = Path(args.plan_json)
    if not plan_path.exists():
        print(f"Plan file not found: {plan_path}", file=sys.stderr)
        sys.exit(1)

    plan = NormalizedPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    if args.building_use:
        normalized = NormalizedPlan._normalize_building_use(args.building_use)
        if not normalized:
            print(f"Unsupported building use: {args.building_use}", file=sys.stderr)
            sys.exit(2)
        plan.building_use = ValueField[str](value=normalized, confidence=Confidence(level=ConfidenceLevel.HIGH, reason="CLI context"), source="CLI")
    if args.development_area:
        plan.development_area = ValueField[str](value=args.development_area, confidence=Confidence(level=ConfidenceLevel.HIGH, reason="CLI context"), source="CLI")

    engine = JsonFileRuleEngine()
    result = engine.evaluate_plan(plan, args.municipality)

    _print_summary(result)

    if args.output:
        Path(args.output).write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"Wrote ComplianceResult -> {args.output}")


if __name__ == "__main__":
    main()
