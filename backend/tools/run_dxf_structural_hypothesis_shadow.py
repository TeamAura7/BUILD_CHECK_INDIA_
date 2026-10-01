"""
Architecture V2, Phase DXF-4 shadow-mode harness.

See ARCHITECTURE_V2.md / the DXF redesign plan's phased order: "Phase DXF-4:
DXF-specific validation report + go/no-go criteria." This script runs the
REAL `DXFHybridExtractor` twice per real DXF fixture with hand-verified
ground truth -- once with `use_dxf_evidence_decision_engine=False` (the
shipping default), once with `=True` (the new structural-hypothesis
decision engine, Phase DXF-2/3) -- and scores both against ground truth via
the SAME tolerance/verdict logic `backend.tools.eval_harness` already uses,
so a row here is directly comparable to that harness's own DXF rows. It
does not modify `dxf_extractor.py`/`pipeline.py` or change what either flag
state ships; it only toggles the existing flag via the environment (the
same mechanism `ARCHITECTURE_V2.md`'s own Implementation log used to
verify `use_evidence_decision_engine`) and observes the result.

Usage:
    python -m backend.tools.run_dxf_structural_hypothesis_shadow
    python -m backend.tools.run_dxf_structural_hypothesis_shadow --plans PLAN5 PLAN6
    python -m backend.tools.run_dxf_structural_hypothesis_shadow --output dxf_shadow.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from backend.tools import eval_harness as eh


def _run_dxf_with_flag(dxf_path: Path, plan_id: str, flag: bool):
    """Toggle `use_dxf_evidence_decision_engine` via the environment (the
    same mechanism this project's own docs use to verify
    `use_evidence_decision_engine`), run the REAL `DXFHybridExtractor`, and
    restore the environment afterward regardless of outcome."""
    import backend.config as config_mod

    os.environ["USE_DXF_EVIDENCE_DECISION_ENGINE"] = "true" if flag else "false"
    config_mod.get_settings.cache_clear()
    try:
        from backend.cv_extraction.dxf_extractor import DXFHybridExtractor

        return DXFHybridExtractor().extract(dxf_path, plan_id)
    finally:
        os.environ.pop("USE_DXF_EVIDENCE_DECISION_ENGINE", None)
        config_mod.get_settings.cache_clear()


def _score_dxf_result(result, expected: dict[str, Optional[float]]) -> dict[str, "eh.FieldResult"]:
    if result is None or result.independent_cv is None:
        return {fr.field: fr for fr in eh._score(expected, {})}
    cv_values = eh._cv_only_values(result.independent_cv)
    na_fields = eh._TEXT_ONLY_DXF_FIELDS if not result.text_evidence else set()
    return {fr.field: fr for fr in eh._score(expected, cv_values, na_fields=na_fields)}


def run_shadow(plan_filter: Optional[list[str]] = None) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for plan_id, doc_type, doc_path, expected_path in eh.discover_plans():
        if doc_type != "dxf":
            continue
        if plan_filter and plan_id not in plan_filter:
            continue

        expected: dict[str, Optional[float]] = json.loads(expected_path.read_text(encoding="utf-8"))

        t0 = time.time()
        try:
            result_off = _run_dxf_with_flag(doc_path, plan_id, False)
        except Exception as exc:  # noqa: BLE001 -- one plan's crash must not abort the whole shadow run
            reports[plan_id] = {"error": f"flag=False: {type(exc).__name__}: {exc}"}
            continue
        elapsed_off = time.time() - t0

        t1 = time.time()
        try:
            result_on = _run_dxf_with_flag(doc_path, plan_id, True)
        except Exception as exc:  # noqa: BLE001
            reports[plan_id] = {"error": f"flag=True: {type(exc).__name__}: {exc}"}
            continue
        elapsed_on = time.time() - t1

        scored_off = _score_dxf_result(result_off, expected)
        scored_on = _score_dxf_result(result_on, expected)

        fields = []
        for field_name in expected:
            fr_off = scored_off.get(field_name)
            fr_on = scored_on.get(field_name)
            fields.append({
                "field": field_name,
                "expected": expected[field_name],
                "old_value": fr_off.actual if fr_off else None,
                "old_verdict": fr_off.verdict if fr_off else None,
                "new_value": fr_on.actual if fr_on else None,
                "new_verdict": fr_on.verdict if fr_on else None,
                "transition": f"{fr_off.verdict if fr_off else '?'}->{fr_on.verdict if fr_on else '?'}",
            })
        reports[plan_id] = {
            "elapsed_seconds_flag_off": round(elapsed_off, 1),
            "elapsed_seconds_flag_on": round(elapsed_on, 1),
            "fields": fields,
        }
    return reports


def _print_report(reports: dict[str, dict[str, Any]]) -> None:
    print(f"\n{'Plan':<10}{'Field':<28}{'Expected':<12}{'Old':<24}{'New':<24}{'Transition':<20}")
    print("-" * 118)
    transition_counts: dict[str, int] = {}
    for plan_id, report in reports.items():
        if "error" in report:
            print(f"{plan_id:<10}ERROR: {report['error']}")
            continue
        for f in report["fields"]:
            old_str = f"{f['old_value']!s} [{f['old_verdict']}]" if f["old_verdict"] else "n/a"
            new_str = f"{f['new_value']!s} [{f['new_verdict']}]" if f["new_verdict"] else "n/a"
            transition = f["transition"]
            transition_counts[transition] = transition_counts.get(transition, 0) + 1
            marker = "" if f["old_verdict"] == f["new_verdict"] else "  <-- CHANGED"
            print(f"{plan_id:<10}{f['field']:<28}{f['expected']!s:<12}{old_str:<24}{new_str:<24}{marker}")

    print("\n--- Verdict transitions (old->new), all scored fields ---")
    for transition, count in sorted(transition_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {transition:<24} {count}")

    print("\n--- Elapsed time per plan (flag off / flag on) ---")
    for plan_id, report in reports.items():
        if "error" in report:
            continue
        print(f"  {plan_id:<10} {report['elapsed_seconds_flag_off']:.1f}s / {report['elapsed_seconds_flag_on']:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plans", nargs="*", default=None, help="Restrict to these plan ids (e.g. PLAN5 PLAN6)")
    parser.add_argument("--output", type=Path, default=None, help="Write the full JSON report to this path")
    args = parser.parse_args()

    reports = run_shadow(plan_filter=args.plans)
    _print_report(reports)
    if args.output:
        args.output.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
        print(f"\nFull report written to {args.output}")


if __name__ == "__main__":
    main()
