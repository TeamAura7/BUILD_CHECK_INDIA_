"""
Generate plain-English explanations/suggestions for an ALREADY-COMPUTED
ComplianceResult. Exactly ONE Groq call, no matter how many rules were
evaluated — this never re-decides compliance, only narrates a result
that backend.compliance.engine.JsonFileRuleEngine already produced with
zero API calls.

Typical flow:
    python -m backend.tools.run_compliance plan.json BBMP --output result.json
    python -m backend.tools.run_explain_compliance result.json

Usage:
    python -m backend.tools.run_explain_compliance result.json
    python -m backend.tools.run_explain_compliance result.json --output explained.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backend.compliance.explainer import explain_compliance
from backend.schemas.compliance import ComplianceResult


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compliance_result_json", help="Output of run_compliance.py --output")
    parser.add_argument("--output", help="Optional path to write the explanation JSON")
    args = parser.parse_args()

    path = Path(args.compliance_result_json)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    result = ComplianceResult.model_validate_json(path.read_text(encoding="utf-8"))
    explanation = explain_compliance(result)

    print(f"Plan {explanation.plan_id}")
    print(explanation.overall_summary)
    print()
    for e in explanation.rule_explanations:
        print(f"[{e.status.value}] {e.rule_id}")
        print(f"    {e.plain_explanation}")
        if e.suggestion:
            print(f"    Suggestion: {e.suggestion}")
        print()

    if args.output:
        Path(args.output).write_text(explanation.model_dump_json(indent=2), encoding="utf-8")
        print(f"Wrote explanation -> {args.output}")


if __name__ == "__main__":
    main()
