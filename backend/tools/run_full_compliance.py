"""
Deterministic end-to-end architectural-plan compliance pipeline.

Runtime compliance NEVER asks an LLM to generate a rule. The authoritative
rules come from data/runtime_rules/<municipality>/rules.json and are evaluated
by JsonFileRuleEngine.

Pipeline:
    PDF -> CV/OCR/Vision extraction -> NormalizedPlan
        -> deterministic BBMP rule selection/applicability
        -> deterministic evaluation -> PASS/FAIL/INSUFFICIENT_DATA/REVIEW

RAG may still be used elsewhere in the application for regulatory search and
explanations, but it is not part of the compliance decision path.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from backend.compliance.engine import JsonFileRuleEngine
from backend.config import get_settings
from backend.cv_extraction.pdf_extractor import PDFHybridExtractor
from backend.schemas.enums import ConfidenceLevel
from backend.schemas.evidence import Confidence, ValueField
from backend.schemas.units import UnitValue
from backend.schemas.normalized_plan import NormalizedPlan
from backend.spatial_reasoning.pipeline import build_normalized_plan


def _fmt(vf: Any) -> str:
    if vf is None:
        return "n/a"
    if getattr(vf, "value", None) is None:
        return f"MISSING ({vf.confidence.level.value})"
    return f"{vf.value} ({vf.confidence.level.value})"


def _plan_summary(plan: Any) -> str:
    use = getattr(plan, "building_use", None)
    area = getattr(plan, "development_area", None)
    height = getattr(plan, "building_height_estimated", None)
    floors = getattr(plan.building, "floor_count", None)
    return "\n".join([
        f"building use: {_fmt(use)}   development area: {_fmt(area)}",
        f"height: {_fmt(height)}   floors: {_fmt(floors)}",
        f"plot: width={_fmt(plan.plot.width)}  depth={_fmt(plan.plot.depth)}  area={_fmt(plan.plot.area)}",
        f"building: width={_fmt(plan.building.width)}  depth={_fmt(plan.building.depth)}  footprint={_fmt(plan.building.footprint_area)}",
        f"road: width={_fmt(plan.road.width)}",
        f"setbacks: front={_fmt(plan.setbacks.front)} rear={_fmt(plan.setbacks.rear)} left={_fmt(plan.setbacks.left)} right={_fmt(plan.setbacks.right)}",
        f"coverage: {_fmt(plan.coverage)}   FAR: {_fmt(plan.far)}",
    ])


def _run_extraction(pdf: Path, vision: bool, backend: str | None, max_new_tokens: int | None):
    settings = get_settings()
    if vision:
        settings.vision_enabled = True
    if backend:
        settings.vision_backend = backend
    if max_new_tokens:
        settings.vision_max_new_tokens = max_new_tokens

    print(f"vision_enabled={settings.vision_enabled}", flush=True)
    if settings.vision_enabled:
        from backend.vision_extraction import get_vision_extractor
        extractor = get_vision_extractor()
        print(f"vision backend: {type(extractor).__name__} ({extractor.model_name})", flush=True)

    print("\n[1/4] Extracting architectural parameters...", flush=True)
    extraction = PDFHybridExtractor().extract(pdf, pdf.stem)
    print(
        f"  Extracted {len(extraction.dimensions)} native/CV dimension(s), "
        f"{len(extraction.plot_candidates)} plot candidate(s), "
        f"{len(extraction.building_candidates)} building candidate(s), "
        f"{len(extraction.vision_pages)} vision page result(s).",
        flush=True,
    )
    for warning in extraction.warnings:
        print(f"  WARNING: {warning}", flush=True)

    plan = build_normalized_plan(extraction, plan_id=f"plan-{pdf.stem}")
    print("\nResolved NormalizedPlan:", flush=True)
    for line in _plan_summary(plan).splitlines():
        print(f"  {line}", flush=True)
    return settings, extraction, plan


def _apply_regulatory_context(plan: Any, building_use: str | None, development_area: str | None, height_excluding_stilt: float | None) -> None:
    if building_use:
        normalized = NormalizedPlan._normalize_building_use(building_use)
        if not normalized:
            raise ValueError(f"Unsupported building use: {building_use}")
        plan.building_use = ValueField[str](value=normalized, confidence=Confidence(level=ConfidenceLevel.HIGH, reason="CLI regulatory context."), source="CLI")
    if development_area:
        value = development_area.strip().upper()
        if value not in {"A", "B", "C"}:
            raise ValueError("development area must be A, B, or C")
        plan.development_area = ValueField[str](value=value, confidence=Confidence(level=ConfidenceLevel.HIGH, reason="CLI regulatory context."), source="CLI")
    if height_excluding_stilt is not None:
        plan.building_height_excluding_stilt = ValueField[float](value=float(height_excluding_stilt), normalized_value=UnitValue(magnitude=float(height_excluding_stilt), unit="m"), confidence=Confidence(level=ConfidenceLevel.HIGH, reason="CLI regulatory context."), source="CLI")


def main() -> None:
    parser = argparse.ArgumentParser(description="Architectural PDF -> deterministic municipality compliance")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--municipality", required=True)
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--backend", choices=["smolvlm", "qwen", "api"], default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--building-use", default=None, help="Explicit regulatory building-use classification, e.g. residential")
    parser.add_argument("--development-area", choices=["A", "B", "C"], default=None, help="BBMP Table 6 development-area class")
    parser.add_argument("--height-excluding-stilt", type=float, default=None, help="Regulatory height excluding stilt floor (for draft Table 8 only)")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"ERROR: PDF not found: {args.pdf}", file=sys.stderr)
        sys.exit(2)

    municipality = args.municipality.upper()
    try:
        settings, extraction, plan = _run_extraction(args.pdf, args.vision, args.backend, args.max_new_tokens)
    except Exception:
        print("\nExtraction/resolution failed:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    try:
        _apply_regulatory_context(plan, args.building_use, args.development_area, args.height_excluding_stilt)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    print("\n[2/4] Loading authoritative deterministic ruleset...", flush=True)
    engine = JsonFileRuleEngine(settings=settings)
    try:
        rules = engine.load_ruleset(municipality)
    except Exception as exc:
        print(f"ERROR: invalid {municipality} deterministic ruleset: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"  Active rules loaded: {len(rules)}", flush=True)
    print(f"  Ruleset version: {rules[0].version if rules else 'none'}", flush=True)
    print("  Source: data/runtime_rules/%s/rules.json" % municipality, flush=True)
    print("  LLM rule generation: DISABLED", flush=True)

    print("\n[3/4] Resolving applicable rules and evaluating deterministically...", flush=True)
    compliance = engine.evaluate_plan(plan, municipality)
    results = []
    for rr in compliance.rule_results:
        payload = json.loads(rr.model_dump_json())
        results.append(payload)
        if rr.status.value != "NOT_APPLICABLE":
            print(
                f"  {rr.status.value}: {rr.source_field or rr.rule_id} | "
                f"{rr.required_value_description or rr.rule_description}",
                flush=True,
            )

    print("\n[4/4] Deterministic compliance result", flush=True)
    print("=" * 72)
    print(f"  Municipality : {municipality}")
    print(f"  Plan         : {args.pdf.name}")
    print(f"  Active rules : {len(rules)}")
    print(f"  Evaluations  : {len(results)}")
    print(f"  OVERALL      : {compliance.overall_status.value}")
    print("=" * 72)

    report = {
        "pipeline": {
            "name": "architectural_pdf_to_deterministic_compliance",
            "municipality": municipality,
            "source_pdf": str(args.pdf),
            "rule_source": f"data/runtime_rules/{municipality}/rules.json",
            "rule_generation": "none_at_runtime",
            "rag_role": "not_authoritative; optional explanation/retrieval only",
            "final_decision": "JsonFileRuleEngine + DeterministicRuleEvaluator",
            "live_ruleset_modified": False,
        },
        "plan": plan.model_dump(mode="json"),
        "ruleset": {
            "active_rule_count": len(rules),
            "version": rules[0].version if rules else None,
        },
        "compliance": {
            "overall_status": compliance.overall_status.value,
            "rule_results": results,
            "counts": compliance.count_by_status(),
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nComplete report written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
