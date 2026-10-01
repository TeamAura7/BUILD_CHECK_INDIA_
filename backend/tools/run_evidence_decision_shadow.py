"""
Architecture V2, Phase 7 shadow-mode harness.

See ARCHITECTURE_V2.md, Deliverable C.2 item 5 and the phased-order section:
"Phase 7: build `evidence_decision.py` on top of `document_evidence.py`'s
existing design; run it in shadow mode (computed, logged, not yet shipped)
against all 7 real plans."

This script runs `backend.spatial_reasoning.evidence_decision.decide_field`
side by side with whatever the pipeline ALREADY ships for the same field on
each real plan in `data/test_plans/`, and reports where the two agree or
disagree -- on VALUE, on CONFIDENCE LEVEL, and against the hand-verified
ground truth (`PLANn.expected.json`). It does not call, import as a
dependency of, or modify `pipeline.py`/`final_fusion.py` in any way that
would change what either currently ships; it only reuses
`backend.tools.eval_harness`'s existing plan-loading helpers to avoid a
second, independently-drifting way of extracting the same real fixtures.

Two separate "shipped" comparisons are reported, deliberately not
conflated:
  - `shipped_value`/`shipped_status`: `document_evidence.
    build_document_verified_fusion`'s own analysis (via `eval_harness`'s
    existing helpers) -- a good, already-correct engine, per this
    project's own audit (ARCHITECTURE_V2.md Deliverable B.6), but NOT
    actually wired into the live pipeline today.
  - `real_shipped_confidence_level`/`_score`: what `pipeline.
    build_normalized_plan` (the code path `analyze.py` actually calls in
    production) puts on the real `NormalizedPlan`, via a genuine second
    extraction through `PDFHybridExtractor`/`DXFHybridExtractor` +
    `build_normalized_plan` -- this is the TRUE "what ships today" signal,
    and is what first surfaced the live confidence bug documented below.

Usage:
    python -m backend.tools.run_evidence_decision_shadow
    python -m backend.tools.run_evidence_decision_shadow --plans PLAN5 PLAN6
    python -m backend.tools.run_evidence_decision_shadow --output shadow_report.json

Known scoping limitation of this shadow run (documented, not silently
assumed): DXF plans are compared using an unavailable `DocumentTextIndex`
(no native-PDF-style text layer to build one from -- DXF's own native TEXT/
MTEXT entities are a different evidence channel, not yet wired into
`document_evidence.DocumentTextIndex`), so a DXF field's new-engine decision
will never reach `*_DOCUMENT_VERIFIED` status through this path even when
the DXF extractor's own confidence is high from other evidence. This is a
known gap, not a claim that DXF fields have no document evidence -- see
ARCHITECTURE_V2.md's "What was deliberately NOT attempted" section.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from backend.schemas.evidence import ValueField
from backend.schemas.normalized_plan import NormalizedPlan
from backend.spatial_reasoning import document_evidence as doc_ev
from backend.spatial_reasoning.evidence_decision import decide_field
from backend.spatial_reasoning.pipeline import build_normalized_plan
from backend.tools import eval_harness as eh


def _unit_for_field(field_name: str) -> str:
    if any(token in field_name for token in ("area", "coverage", "far")):
        return "sq_m"
    return "m"


def _real_plan_field(plan: NormalizedPlan, field_name: str) -> Optional[ValueField]:
    """Map an `expected.json`-style field path onto the actual `ValueField`
    an `analyze.py`-style live extraction would produce on `NormalizedPlan`.
    Returns None for fields with no direct `NormalizedPlan` representation
    (e.g. `plot.net_area`, `far.area`, `building.gross_built_up_area` --
    computed/tracked only inside the eval harness's own PDF fusion report,
    not on the core schema) rather than guessing at a mapping."""
    mapping = {
        "plot.width": plan.plot.width, "plot.depth": plan.plot.depth, "plot.area": plan.plot.area,
        "building.width": plan.building.width, "building.depth": plan.building.depth,
        "building.footprint_area": plan.building.footprint_area,
        "road.width": plan.road.width,
        "setbacks.front": plan.setbacks.front, "setbacks.rear": plan.setbacks.rear,
        "setbacks.left": plan.setbacks.left, "setbacks.right": plan.setbacks.right,
        "coverage": plan.coverage, "far": plan.far,
    }
    return mapping.get(field_name)


def _real_shipped_plan(doc_type: str, doc_path: Path, plan_id: str) -> Optional[NormalizedPlan]:
    """Run the ACTUAL production extraction path (the same classes
    `analyze.py` instantiates) and build the real `NormalizedPlan` -- the
    true "what ships today" signal, as opposed to `document_evidence.
    build_document_verified_fusion`'s own (currently unwired) analysis."""
    try:
        if doc_type == "pdf":
            from backend.cv_extraction.pdf_extractor import PDFHybridExtractor

            extraction = PDFHybridExtractor().extract(doc_path, plan_id)
        else:
            from backend.cv_extraction.dxf_extractor import DXFHybridExtractor

            extraction = DXFHybridExtractor().extract(doc_path, plan_id)
        return build_normalized_plan(extraction, plan_id=plan_id)
    except Exception as exc:  # noqa: BLE001 -- one plan's crash must not abort the whole shadow run
        print(f"  (real-plan comparison unavailable for {plan_id}[{doc_type}]: {type(exc).__name__}: {exc})")
        return None


def _shadow_for_pdf_plan(plan_id: str, pdf_path: Path, expected_path: Path) -> dict[str, Any]:
    expected: dict[str, Optional[float]] = json.loads(expected_path.read_text(encoding="utf-8"))
    try:
        run = eh._run_fusion_for_plan(pdf_path, plan_id)
    except Exception as exc:  # noqa: BLE001 -- one plan's crash must not abort the whole shadow run
        return {"plan_id": plan_id, "doc_type": "pdf", "error": f"{type(exc).__name__}: {exc}"}

    cv_result = run["cv"]
    cv_values = eh._cv_only_values(cv_result)
    vision_values = eh._vision_only_values(run["vision"], list(expected.keys()))
    shipped_values = eh._fusion_only_values(run["fusion"])
    shipped_statuses = eh._fusion_statuses(run["fusion"])
    text_index = doc_ev.build_document_text_index(str(pdf_path))
    real_plan = _real_shipped_plan("pdf", pdf_path, plan_id)

    fields = []
    for field_name, expected_value in expected.items():
        cv_value = cv_values.get(field_name)
        vision_value = vision_values.get(field_name)
        field_measurements = [m for m in cv_result.measurements if m.field == field_name]
        decision, value_field = decide_field(
            field_name, cv_value, vision_value, _unit_for_field(field_name), field_measurements, text_index,
        )
        real_field = _real_plan_field(real_plan, field_name) if real_plan is not None else None
        fields.append({
            "field": field_name,
            "expected": expected_value,
            "shipped_value": shipped_values.get(field_name),
            "shipped_status": shipped_statuses.get(field_name),
            "real_shipped_confidence_level": real_field.confidence.level.value if real_field is not None else None,
            "real_shipped_confidence_score": real_field.confidence.score if real_field is not None else None,
            "real_shipped_source": real_field.source if real_field is not None else None,
            "new_value": value_field.value,
            "new_status": decision.status.value,
            "new_confidence_level": value_field.confidence.level.value,
            "new_confidence_score": value_field.confidence.score,
            "agrees_with_shipped": _values_agree(shipped_values.get(field_name), value_field.value),
        })
    return {"plan_id": plan_id, "doc_type": "pdf", "fields": fields}


def _shadow_for_dxf_plan(plan_id: str, dxf_path: Path, expected_path: Path) -> dict[str, Any]:
    expected: dict[str, Optional[float]] = json.loads(expected_path.read_text(encoding="utf-8"))
    try:
        result = eh._run_dxf_for_plan(dxf_path, plan_id)
    except Exception as exc:  # noqa: BLE001
        return {"plan_id": plan_id, "doc_type": "dxf", "error": f"{type(exc).__name__}: {exc}"}

    if result.independent_cv is None:
        return {"plan_id": plan_id, "doc_type": "dxf", "error": "no independent_cv result"}

    cv_result = result.independent_cv
    cv_values = eh._cv_only_values(cv_result)
    # DXF has no native-PDF-style document text index -- see module docstring's
    # "Known scoping limitation" note.
    text_index = doc_ev.DocumentTextIndex(pages=[], available=False)
    # DXF's `result` here IS already the full `ExtractionResult` -- no
    # second extraction needed, unlike the PDF path.
    real_plan: Optional[NormalizedPlan]
    try:
        real_plan = build_normalized_plan(result, plan_id=plan_id)
    except Exception as exc:  # noqa: BLE001
        print(f"  (real-plan comparison unavailable for {plan_id}[dxf]: {type(exc).__name__}: {exc})")
        real_plan = None

    fields = []
    for field_name, expected_value in expected.items():
        cv_value = cv_values.get(field_name)
        field_measurements = [m for m in cv_result.measurements if m.field == field_name]
        shipped_confidence = next((m.confidence for m in field_measurements if m.value_m == cv_value or m.value == cv_value), None)
        decision, value_field = decide_field(
            field_name, cv_value, None, _unit_for_field(field_name), field_measurements, text_index,
        )
        real_field = _real_plan_field(real_plan, field_name) if real_plan is not None else None
        fields.append({
            "field": field_name,
            "expected": expected_value,
            "shipped_value": cv_value,
            "shipped_confidence_raw": shipped_confidence,
            "real_shipped_confidence_level": real_field.confidence.level.value if real_field is not None else None,
            "real_shipped_confidence_score": real_field.confidence.score if real_field is not None else None,
            "real_shipped_source": real_field.source if real_field is not None else None,
            "new_value": value_field.value,
            "new_status": decision.status.value,
            "new_confidence_level": value_field.confidence.level.value,
            "new_confidence_score": value_field.confidence.score,
            "agrees_with_shipped": _values_agree(cv_value, value_field.value),
        })
    return {"plan_id": plan_id, "doc_type": "dxf", "fields": fields}


def _values_agree(a: Optional[float], b: Optional[float], tol: float = 1e-6) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) <= max(tol, abs(a) * 0.001)


def run_shadow(plan_filter: Optional[list[str]] = None) -> list[dict[str, Any]]:
    reports = []
    for plan_id, doc_type, doc_path, expected_path in eh.discover_plans():
        if plan_filter and plan_id not in plan_filter:
            continue
        if doc_type == "pdf":
            reports.append(_shadow_for_pdf_plan(plan_id, doc_path, expected_path))
        else:
            reports.append(_shadow_for_dxf_plan(plan_id, doc_path, expected_path))
    return reports


def _print_report(reports: list[dict[str, Any]]) -> None:
    total_fields = 0
    total_agree = 0
    total_disagree_status = 0
    medium_with_no_score: list[str] = []
    confidence_level_mismatches: list[str] = []
    for report in reports:
        label = f"{report['plan_id']}[{report['doc_type']}]"
        if "error" in report:
            print(f"\n=== {label}: ERROR: {report['error']} ===")
            continue
        print(f"\n=== {label} ===")
        for f in report["fields"]:
            total_fields += 1
            agree_mark = "=" if f["agrees_with_shipped"] else "!="
            if f["agrees_with_shipped"]:
                total_agree += 1
            shipped_status = f.get("shipped_status", "-")
            real_level = f.get("real_shipped_confidence_level")
            real_score = f.get("real_shipped_confidence_score")
            real_note = f" real={real_level}/{real_score}" if real_level is not None else ""
            print(
                f"  {f['field']:<32} expected={f['expected']!s:<10} "
                f"shipped={f['shipped_value']!s:<10}({shipped_status!s:<28}) "
                f"{agree_mark} new={f['new_value']!s:<10}[{f['new_status']}/{f['new_confidence_level']}]{real_note}"
            )
            if f["new_status"] not in ("ACCEPT",) and f["shipped_value"] is not None:
                total_disagree_status += 1
            if real_level == "MEDIUM" and real_score is None:
                medium_with_no_score.append(f"{label}.{f['field']}")
            if real_level is not None and real_level != f["new_confidence_level"]:
                confidence_level_mismatches.append(
                    f"{label}.{f['field']}: real={real_level} vs new={f['new_confidence_level']}"
                )
    print(
        f"\n--- Summary: {total_agree}/{total_fields} fields' shipped value agrees with the new engine's value "
        f"({total_disagree_status} fields where the new engine would ABSTAIN/CONFLICT but something is currently shipped) ---"
    )
    if medium_with_no_score:
        print(
            f"\n*** LIVE BUG CONFIRMED ({len(medium_with_no_score)} field(s)): real-shipped ConfidenceLevel.MEDIUM "
            "with score=None -- exactly the pattern ARCHITECTURE_V2.md Deliverable B.5 flagged as a live bug in "
            "final_fusion.py's vf() closure ***"
        )
        for entry in medium_with_no_score:
            print(f"    {entry}")
    if confidence_level_mismatches:
        print(f"\n--- {len(confidence_level_mismatches)} field(s) where the new engine's confidence level differs from what's really shipped ---")
        for entry in confidence_level_mismatches:
            print(f"    {entry}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
