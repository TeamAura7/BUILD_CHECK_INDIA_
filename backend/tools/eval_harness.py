"""
Eval harness: the missing piece identified after a long round of manual
bug-hunting on individual plans -- there was no repeatable way to tell
whether a change actually improved extraction accuracy across the whole
test-plan set, only whether one specific number changed on one specific
plan. This fixes that.

For every `data/test_plans/PLANn.pdf` AND/OR `data/test_plans/PLANn.dxf`
that has a matching `PLANn.expected.json` (hand-verified ground truth,
field -> value, same keys as `final_fusion.LENGTH_FIELDS`/`NUMERIC_FIELDS`
for either format), this:

  - PDF: runs the independent CV pipeline (`extract_independent_cv`), the
    independent Vision pipeline (`get_vision_extractor()`) if
    `VISION_ENABLED`, and the document-evidence fusion engine
    (`build_document_verified_fusion`); scores CV alone, Vision alone, and
    the fusion's `final_agreed_values` against `expected.json`.
  - DXF: runs `DXFHybridExtractor` (a single deterministic pipeline, no
    separate Vision/fusion layer) and scores its measurements against the
    SAME `expected.json` -- the true dimensions of a real plan don't
    depend on which file format was scanned/exported. When a plan has
    BOTH a `.pdf` and a `.dxf` (as PLAN5/PLAN6 do, kept specifically as
    DXF-pipeline regression fixtures), both are evaluated as separate
    rows against the one shared ground truth.

Adding a new plan to the eval set is ALWAYS just dropping fixture files
into `data/test_plans/` (a `.pdf` and/or `.dxf` plus a matching
`.expected.json`) -- never a change to this harness's own code
(`discover_plans` walks the directory for whatever `.expected.json` files
exist, so this stays true going forward, not just for the plans present
today).

PLAN5 and PLAN6 (both formats) are validation fixtures ONLY: this harness
reports their numbers honestly, but no threshold, weight, or heuristic
anywhere in the extraction pipeline should ever be tuned to make PLAN5/
PLAN6 specifically pass -- that is overfitting to two known plans, not a
generalizable fix, and it would stop this harness from meaning anything.

A field counts CORRECT when a value was produced and it's within
tolerance of expected (absolute or relative, whichever is looser -- see
`_within_tolerance`); WRONG when a value was produced but outside
tolerance; MISSING when no value was produced. This distinction matters:
a MISSING field is honest uncertainty, a WRONG field is a confident
mistake -- the two should never be conflated into one "not correct"
bucket, since a fusion engine that never guesses will show more MISSING
and less WRONG than one that does, and that trade-off should be visible,
not hidden behind a single pass/fail number.

An `expected.json` field may also be explicitly `null`. That means the
ground truth itself is "this field must remain unresolved" (e.g. PLAN7,
a photograph of a blueprint with no vector geometry and OCR that loses
decimal points -- see tests/test_real_plan_regression.py). A produced
value there is a FALSE_POSITIVE: a confident answer where the honest
answer is "I don't know", strictly worse than an ordinary WRONG value
against a real expected number, because there the pipeline had a reason
to think a value existed at all. `null` is distinct from a field simply
being ABSENT from expected.json (no assertion either way, not scored).

Beyond per-field CORRECT/WRONG/MISSING/FALSE_POSITIVE counts, this also
reports, aggregated per plan and overall, per pipeline:
  - MAE / MAPE for numeric fields where both an expected and an actual
    value exist (i.e. over CORRECT + WRONG fields only).
  - Completeness rate = fraction of "should have a value" fields
    (expected is not null) that were not MISSING.
  - Conflict rate = fraction of scored fields whose fusion status came
    back CONFLICT/UNRESOLVED_CONFLICT (fusion pipeline only -- CV/Vision
    alone have no notion of "conflict", only "found"/"not found").
  - False-positive rate = FALSE_POSITIVE count / count of must-be-missing
    (expected == null) fields.

Usage:
    python -m backend.tools.eval_harness
    python -m backend.tools.eval_harness --plans PLAN2 PLAN5
    python -m backend.tools.eval_harness --output eval_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Optional, Sequence

from backend.config import get_settings

TEST_PLANS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "test_plans"

# Looser of (absolute, relative) -- small values (setbacks) need an
# absolute floor; large values (areas) need a relative allowance, or a
# fixed absolute tolerance would be simultaneously too strict for areas
# and too loose for setbacks.
DEFAULT_ABS_TOL = 0.15
DEFAULT_REL_TOL = 0.05


def _within_tolerance(actual: float, expected: float, abs_tol: float = DEFAULT_ABS_TOL, rel_tol: float = DEFAULT_REL_TOL) -> bool:
    tol = max(abs_tol, abs(expected) * rel_tol)
    return abs(actual - expected) <= tol


@dataclass
class FieldResult:
    field: str
    expected: Optional[float]  # None means "ground truth: must stay MISSING"
    actual: Optional[float]
    verdict: str  # "CORRECT" | "WRONG" | "MISSING" | "FALSE_POSITIVE" | "N/A"
    status: Optional[str] = None  # fusion-only: AGREED/CV_ONLY/VISION_ONLY/CONFLICT/... when known
    abs_error: Optional[float] = None  # |actual - expected|, only when both exist and expected is not null
    pct_error: Optional[float] = None  # abs_error / |expected| * 100, only when expected != 0


@dataclass
class PlanResult:
    plan_id: str
    pdf_path: str
    field_results: list[FieldResult] = dc_field(default_factory=list)
    error: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    doc_type: str = "pdf"  # "pdf" | "dxf" -- which extractor produced this result

    def counts(self) -> dict[str, int]:
        out = {"CORRECT": 0, "WRONG": 0, "MISSING": 0, "FALSE_POSITIVE": 0, "N/A": 0}
        for fr in self.field_results:
            out[fr.verdict] += 1
        return out

    def accuracy(self) -> Optional[float]:
        # N/A fields (structurally unreachable given this document's own
        # content, e.g. a text-only field on a DXF with zero native text)
        # are excluded from the denominator -- they were never scoreable,
        # not missed.
        scoreable = [fr for fr in self.field_results if fr.verdict != "N/A"]
        if not scoreable:
            return None
        correct = sum(1 for fr in scoreable if fr.verdict == "CORRECT")
        return correct / len(scoreable)

    def mae_mape(self) -> tuple[Optional[float], Optional[float]]:
        """Mean absolute error / mean absolute percentage error, over fields
        where both an expected (non-null) and an actual value exist."""
        errors: list[float] = []
        pct_errors: list[float] = []
        for fr in self.field_results:
            if fr.expected is None or fr.actual is None:
                continue
            err = abs(fr.actual - fr.expected)
            errors.append(err)
            if fr.expected != 0:
                pct_errors.append(err / abs(fr.expected) * 100.0)
        mae = sum(errors) / len(errors) if errors else None
        mape = sum(pct_errors) / len(pct_errors) if pct_errors else None
        return mae, mape

    def completeness(self) -> Optional[float]:
        """Fraction of 'should have a value' fields (expected is not null) that
        were not MISSING. Fields where the ground truth itself is null are
        excluded -- abstaining there is correct, not incomplete. Fields marked
        N/A are excluded too -- they were never scoreable for this document."""
        scoreable = [fr for fr in self.field_results if fr.expected is not None and fr.verdict != "N/A"]
        if not scoreable:
            return None
        resolved = [fr for fr in scoreable if fr.verdict != "MISSING"]
        return len(resolved) / len(scoreable)

    def conflict_rate(self) -> Optional[float]:
        scoreable = [fr for fr in self.field_results if fr.status is not None]
        if not scoreable:
            return None
        conflicted = [fr for fr in scoreable if "CONFLICT" in fr.status.upper()]
        return len(conflicted) / len(scoreable)

    def false_positive_rate(self) -> Optional[float]:
        must_be_missing = [fr for fr in self.field_results if fr.expected is None]
        if not must_be_missing:
            return None
        false_positives = [fr for fr in must_be_missing if fr.verdict == "FALSE_POSITIVE"]
        return len(false_positives) / len(must_be_missing)

    def boundary_resolved(self) -> bool:
        """True when every boundary-primitive field this plan's ground
        truth actually asks for (`_BOUNDARY_FIELDS`) has a produced,
        non-conflicting value -- i.e. neither MISSING/N/A nor a fusion
        CONFLICT status.

        Deliberately does NOT require the value to be numerically
        CORRECT: a resolved-but-wrong boundary is still "resolved" for
        this flag's purpose. It exists to separate "the boundary itself
        never produced a usable answer" (setbacks/coverage/FAR downstream
        of it are not meaningfully comparable) from "an ordinary wrong
        value downstream of a perfectly fine boundary" -- a different
        failure with a different fix. A plan whose ground truth doesn't
        even include any of these fields (nothing to check) is NOT
        considered resolved -- there is nothing to have resolved.
        """
        present = [fr for fr in self.field_results if fr.field in _BOUNDARY_FIELDS]
        if not present:
            return False
        for fr in present:
            if fr.actual is None:
                return False
            if fr.status and "CONFLICT" in fr.status.upper():
                return False
        return True


# The plot/building geometry every downstream derived field (setbacks,
# coverage, FAR) is computed from or compared against. Chosen from the
# fields actually present across every real `data/test_plans/*.expected.json`
# today (see PLAN2/4/5/6) rather than invented names.
_BOUNDARY_FIELDS = frozenset({"plot.width", "plot.depth", "building.footprint_area"})


def aggregate_plan_results(results: Sequence[PlanResult]) -> dict[str, Any]:
    """Pool per-field metrics across many `PlanResult`s into one summary.

    Callers decide which results to include (all plans, or only
    `boundary_resolved()` ones) -- this function only pools whatever it's
    given, so the same code produces both the ALL-PLANS and
    BOUNDARY-RESOLVED-ONLY aggregates (see `boundary_gated_aggregates`).
    Plans that errored out (crash/timeout) are excluded from every metric
    here, same as the existing per-plan totals in `_print_report`.
    """
    counts = {"CORRECT": 0, "WRONG": 0, "MISSING": 0, "FALSE_POSITIVE": 0, "N/A": 0}
    errors: list[float] = []
    pct_errors: list[float] = []
    completeness_scoreable = 0
    completeness_resolved = 0
    conflict_scoreable = 0
    conflict_hits = 0
    fp_denominator = 0
    fp_hits = 0

    included = 0
    for r in results:
        if r.error:
            continue
        included += 1
        c = r.counts()
        for k in counts:
            counts[k] += c[k]
        for fr in r.field_results:
            if fr.expected is not None and fr.actual is not None:
                errors.append(fr.abs_error if fr.abs_error is not None else abs(fr.actual - fr.expected))
                if fr.pct_error is not None:
                    pct_errors.append(fr.pct_error)
            if fr.expected is not None and fr.verdict != "N/A":
                completeness_scoreable += 1
                if fr.verdict != "MISSING":
                    completeness_resolved += 1
            if fr.status is not None:
                conflict_scoreable += 1
                if "CONFLICT" in fr.status.upper():
                    conflict_hits += 1
            if fr.expected is None:
                fp_denominator += 1
                if fr.verdict == "FALSE_POSITIVE":
                    fp_hits += 1

    scoreable_total = counts["CORRECT"] + counts["WRONG"] + counts["MISSING"] + counts["FALSE_POSITIVE"]
    return {
        "plan_count": included,
        "counts": counts,
        "accuracy": (counts["CORRECT"] / scoreable_total) if scoreable_total else None,
        "mae": (sum(errors) / len(errors)) if errors else None,
        "mape": (sum(pct_errors) / len(pct_errors)) if pct_errors else None,
        "completeness": (completeness_resolved / completeness_scoreable) if completeness_scoreable else None,
        "conflict_rate": (conflict_hits / conflict_scoreable) if conflict_scoreable else None,
        "false_positive_rate": (fp_hits / fp_denominator) if fp_denominator else None,
    }


def boundary_gated_aggregates(results: Sequence[PlanResult]) -> dict[str, dict[str, Any]]:
    """The two required aggregation views (Part O.2): every plan pooled
    together, and the same metrics recomputed over only the plans whose
    boundary primitives actually resolved. Never replaces the all-plans
    view with the gated one, and never drops boundary-field failures from
    either view -- the gate filters which PLANS are included, not which
    FIELDS are scored within a plan, so a failed boundary still shows up
    as a boundary failure in both aggregates.
    """
    resolved = [r for r in results if not r.error and r.boundary_resolved()]
    return {
        "all_plans": aggregate_plan_results(results),
        "boundary_resolved_only": aggregate_plan_results(resolved),
    }


def discover_plans() -> list[tuple[str, str, Path, Path]]:
    """Return (plan_id, doc_type, document_path, expected_json_path) for
    every plan with a matching ground-truth file.

    Phase 0 validation requirement: adding a new plan to the eval set must
    never require a production-code change, only new fixture files -- drop
    `data/test_plans/PLANn.pdf` and/or `data/test_plans/PLANn.dxf` next to
    a `data/test_plans/PLANn.expected.json` (same field-name schema either
    way; see PLAN2.expected.json) and it is picked up automatically. When
    BOTH a `.pdf` and a `.dxf` exist for the same plan_id (as for PLAN5/
    PLAN6, kept as DXF-pipeline validation fixtures alongside their
    original PDF versions), both are evaluated as separate entries against
    the SAME ground truth -- the true dimensions of a real plan don't
    depend on which file format happened to be scanned/exported, and
    seeing both pipelines scored against one shared expectation is the
    point, not a duplication to collapse.
    """
    out: list[tuple[str, str, Path, Path]] = []
    if not TEST_PLANS_DIR.exists():
        return out
    for expected_path in sorted(TEST_PLANS_DIR.glob("*.expected.json")):
        plan_id = expected_path.name.removesuffix(".expected.json")
        pdf_path = TEST_PLANS_DIR / f"{plan_id}.pdf"
        if pdf_path.exists():
            out.append((plan_id, "pdf", pdf_path, expected_path))
        dxf_path = TEST_PLANS_DIR / f"{plan_id}.dxf"
        if dxf_path.exists():
            out.append((plan_id, "dxf", dxf_path, expected_path))
    return out


def _run_fusion_for_plan(pdf_path: Path, plan_id: str) -> dict[str, Any]:
    from backend.cv_extraction.site_plan import extract_independent_cv
    from backend.spatial_reasoning.final_fusion import build_document_verified_fusion
    from backend.vision_extraction import get_vision_extractor

    settings = get_settings()
    cv_result = extract_independent_cv(pdf_path, plan_id)
    vision_result = None
    vision_error: Optional[str] = None
    if settings.vision_enabled:
        try:
            vision_result = get_vision_extractor().analyze_pdf(pdf_path, ground_against_native_text=False)
        except Exception as exc:  # noqa: BLE001 -- eval harness must not crash on one plan's vision failure
            vision_error = str(exc)
    fusion = build_document_verified_fusion(cv_result, vision_result, pdf_path=str(pdf_path))
    return {"cv": cv_result, "vision": vision_result, "vision_error": vision_error, "fusion": fusion}


def _cv_only_values(cv_result) -> dict[str, Optional[float]]:
    from backend.spatial_reasoning.final_fusion import _cv_map

    cm = _cv_map(cv_result)
    out: dict[str, Optional[float]] = {}
    for field_name, m in cm.items():
        out[field_name] = m.value_m if m.value_m is not None else m.value
    return out


def _vision_only_values(vision_result, expected_fields: list[str]) -> dict[str, Optional[float]]:
    from backend.spatial_reasoning.final_fusion import LENGTH_FIELDS, NUMERIC_FIELDS, _vision_length, _vision_numeric

    out: dict[str, Optional[float]] = {}
    if vision_result is None:
        return out
    for field_name in expected_fields:
        if field_name in LENGTH_FIELDS:
            res = _vision_length(vision_result, LENGTH_FIELDS[field_name])
            out[field_name] = res[0] if res else None
        elif field_name in NUMERIC_FIELDS:
            res = _vision_numeric(vision_result, NUMERIC_FIELDS[field_name])
            out[field_name] = res[0] if res else None
    return out


def _fusion_only_values(fusion: dict[str, Any]) -> dict[str, Optional[float]]:
    return {
        field_name: entry.get("final_value")
        for field_name, entry in fusion.get("final_agreed_values", {}).items()
    }


def _fusion_statuses(fusion: dict[str, Any]) -> dict[str, str]:
    return {
        field_name: str(entry.get("status"))
        for field_name, entry in fusion.get("final_agreed_values", {}).items()
        if entry.get("status") is not None
    }


def _score(
    expected: dict[str, Optional[float]],
    actual: dict[str, Optional[float]],
    statuses: Optional[dict[str, str]] = None,
    na_fields: Optional[set[str]] = None,
) -> list[FieldResult]:
    na_fields = na_fields or set()
    results = []
    for field_name, expected_value in expected.items():
        actual_value = actual.get(field_name)
        status = statuses.get(field_name) if statuses else None
        abs_error: Optional[float] = None
        pct_error: Optional[float] = None
        if expected_value is None:
            # Ground truth says this field must remain unresolved. A
            # produced value here -- of any magnitude -- is a confident
            # wrong answer where "I don't know" was the correct one.
            verdict = "CORRECT" if actual_value is None else "FALSE_POSITIVE"
        elif actual_value is None:
            # Distinguish "the pipeline failed to find this" (MISSING) from
            # "this document has no way to produce this field at all" (N/A)
            # -- e.g. a field only ever populated from native DXF TEXT/MTEXT
            # entities, on a document that has none. Leaving those scored as
            # MISSING would count a structural non-goal against every future
            # measurement on this document class.
            verdict = "N/A" if field_name in na_fields else "MISSING"
        else:
            abs_error = abs(float(actual_value) - float(expected_value))
            if expected_value != 0:
                pct_error = abs_error / abs(float(expected_value)) * 100.0
            verdict = "CORRECT" if _within_tolerance(float(actual_value), float(expected_value)) else "WRONG"
        results.append(FieldResult(
            field=field_name, expected=expected_value, actual=actual_value, verdict=verdict, status=status,
            abs_error=abs_error, pct_error=pct_error,
        ))
    return results


# Fields whose ONLY production code path in `dxf_extractor.py` is
# `_AREA_FIELD_PATTERNS`, matched against native TEXT/MTEXT entities -- there
# is no geometry-only fallback for any of these (verified by inspection: they
# never appear as an assignment target anywhere else in that module). On a
# DXF with zero native text entities (e.g. one where every glyph was exploded
# to vector curves before export), these are unreachable by construction, not
# a pipeline weakness -- scoring them as MISSING would count the same
# structural non-goal against every future run on this document class.
#
# `road.width` is deliberately NOT in this set even though it was also
# unreachable on the specific real-world files that prompted this fix: unlike
# the fields below, `_pick_road_polygon` can resolve it purely from layer
# name / geometric adjacency, with no text involved at all. Those files
# happened to mark the road via a colour-coded legend rather than a named
# layer or DXF text, so both paths failed together -- but that is a property
# of how those particular files were authored, not a fact about text-less
# DXFs in general. Blanket-marking road.width as N/A here would hide genuine
# future improvements to the geometry-only path on a differently-authored
# text-less file.
_TEXT_ONLY_DXF_FIELDS: frozenset[str] = frozenset({
    "plot.net_area", "coverage", "far", "far.area", "building.gross_built_up_area",
})


def _run_dxf_for_plan(dxf_path: Path, plan_id: str):
    from backend.cv_extraction.dxf_extractor import DXFHybridExtractor

    return DXFHybridExtractor().extract(dxf_path, plan_id)


def evaluate_pdf_plan(plan_id: str, pdf_path: Path, expected_path: Path) -> dict[str, PlanResult]:
    """Returns {'cv': PlanResult, 'vision': PlanResult, 'fusion': PlanResult} for one PDF plan."""
    import time

    expected: dict[str, Optional[float]] = json.loads(expected_path.read_text(encoding="utf-8"))

    t0 = time.time()
    try:
        run = _run_fusion_for_plan(pdf_path, plan_id)
    except Exception as exc:  # noqa: BLE001 -- one plan's crash must not abort the whole eval run
        err = f"{type(exc).__name__}: {exc}"
        empty = PlanResult(plan_id=plan_id, pdf_path=str(pdf_path), error=err, elapsed_seconds=time.time() - t0)
        return {"cv": empty, "vision": empty, "fusion": empty}
    elapsed = time.time() - t0

    cv_values = _cv_only_values(run["cv"])
    vision_values = _vision_only_values(run["vision"], list(expected.keys()))
    fusion_values = _fusion_only_values(run["fusion"])
    fusion_statuses = _fusion_statuses(run["fusion"])

    return {
        "cv": PlanResult(plan_id, str(pdf_path), _score(expected, cv_values), elapsed_seconds=elapsed),
        "vision": PlanResult(
            plan_id, str(pdf_path), _score(expected, vision_values),
            error=run["vision_error"], elapsed_seconds=elapsed,
        ),
        "fusion": PlanResult(
            plan_id, str(pdf_path), _score(expected, fusion_values, fusion_statuses), elapsed_seconds=elapsed,
        ),
    }


def evaluate_dxf_plan(plan_id: str, dxf_path: Path, expected_path: Path) -> dict[str, PlanResult]:
    """Returns {'cv': PlanResult} for one DXF plan -- `DXFHybridExtractor`
    is a single deterministic pipeline with no separate Vision/fusion
    layer (see its own module docstring), unlike the PDF path."""
    import time

    expected: dict[str, Optional[float]] = json.loads(expected_path.read_text(encoding="utf-8"))

    t0 = time.time()
    try:
        result = _run_dxf_for_plan(dxf_path, plan_id)
    except Exception as exc:  # noqa: BLE001 -- one plan's crash must not abort the whole eval run
        err = f"{type(exc).__name__}: {exc}"
        empty = PlanResult(
            plan_id=plan_id, pdf_path=str(dxf_path), error=err, elapsed_seconds=time.time() - t0, doc_type="dxf",
        )
        return {"cv": empty}
    elapsed = time.time() - t0

    if result.independent_cv is None:
        empty = PlanResult(
            plan_id=plan_id, pdf_path=str(dxf_path),
            error="No independent_cv result (see result.warnings, e.g. a timeout)",
            elapsed_seconds=elapsed, doc_type="dxf",
        )
        return {"cv": empty}

    cv_values = _cv_only_values(result.independent_cv)
    na_fields = _TEXT_ONLY_DXF_FIELDS if not result.text_evidence else set()
    return {
        "cv": PlanResult(
            plan_id, str(dxf_path), _score(expected, cv_values, na_fields=na_fields),
            elapsed_seconds=elapsed, doc_type="dxf",
        )
    }


def evaluate_plan(plan_id: str, doc_type: str, doc_path: Path, expected_path: Path) -> dict[str, PlanResult]:
    """Dispatches to the right pipeline(s) for `doc_type` ('pdf' or 'dxf')."""
    if doc_type == "dxf":
        return evaluate_dxf_plan(plan_id, doc_path, expected_path)
    return evaluate_pdf_plan(plan_id, doc_path, expected_path)


def _fmt_pct(x: Optional[float]) -> str:
    return f"{x:.0%}" if x is not None else "n/a"


def _fmt_num(x: Optional[float]) -> str:
    return f"{x:.3f}" if x is not None else "n/a"


def _fmt_secs(x: Optional[float]) -> str:
    return f"{x:.1f}s" if x is not None else "n/a"


# Which pipeline within a plan's results is its "primary" one for the
# per-field detail section -- the final, most-fused answer when one
# exists (PDF: fusion), otherwise the only pipeline there is (DXF: cv).
_PRIMARY_PIPELINE_PREFERENCE = ("fusion", "cv")


def _print_report(all_results: dict[str, dict[str, PlanResult]]) -> None:
    all_pipelines: list[str] = []
    for per_pipeline in all_results.values():
        for pipeline in per_pipeline:
            if pipeline not in all_pipelines:
                all_pipelines.append(pipeline)

    print(f"\n{'Plan':<14}{'Pipeline':<10}{'Correct':<9}{'Wrong':<7}{'Missing':<9}{'FalsePos':<10}{'N/A':<6}{'Accuracy':<10}"
          f"{'Complete':<10}{'Conflict':<10}{'MAE':<9}{'MAPE':<8}{'Runtime':<8}")
    print("-" * 122)
    totals = {p: {"CORRECT": 0, "WRONG": 0, "MISSING": 0, "FALSE_POSITIVE": 0, "N/A": 0} for p in all_pipelines}
    for display_key, per_pipeline in all_results.items():
        for pipeline, result in per_pipeline.items():
            if result.error:
                print(f"{display_key:<14}{pipeline:<10}ERROR: {result.error}")
                continue
            counts = result.counts()
            for k in totals[pipeline]:
                totals[pipeline][k] += counts[k]
            acc = result.accuracy()
            mae, mape = result.mae_mape()
            print(
                f"{display_key:<14}{pipeline:<10}{counts['CORRECT']:<9}{counts['WRONG']:<7}{counts['MISSING']:<9}"
                f"{counts['FALSE_POSITIVE']:<10}{counts['N/A']:<6}{_fmt_pct(acc):<10}{_fmt_pct(result.completeness()):<10}"
                f"{_fmt_pct(result.conflict_rate()):<10}{_fmt_num(mae):<9}{_fmt_num(mape):<8}"
                f"{_fmt_secs(result.elapsed_seconds):<8}"
            )
    print("-" * 122)
    for pipeline in all_pipelines:
        c = totals[pipeline]
        # N/A fields were never scoreable -- excluded from accuracy's denominator,
        # same as `PlanResult.accuracy()` does per-plan.
        scoreable_total = c["CORRECT"] + c["WRONG"] + c["MISSING"] + c["FALSE_POSITIVE"]
        acc_str = f"{c['CORRECT'] / scoreable_total:.0%}" if scoreable_total else "n/a"
        print(
            f"{'TOTAL':<14}{pipeline:<10}{c['CORRECT']:<9}{c['WRONG']:<7}{c['MISSING']:<9}"
            f"{c['FALSE_POSITIVE']:<10}{c['N/A']:<6}{acc_str:<10}"
        )

    # Boundary-gated aggregates (Part O.2): every downstream field
    # (setbacks/coverage/FAR) is only meaningfully comparable when the
    # plot/building boundary itself resolved to a usable value. Report
    # BOTH views, never silently replace the all-plans one, and never
    # exclude the boundary fields themselves from either -- a failed
    # boundary must still show up as its own failure in ALL-PLANS.
    print("\n=== ALL-PLANS AGGREGATE vs BOUNDARY-RESOLVED-ONLY AGGREGATE (per pipeline) ===")
    print(f"{'Pipeline':<10}{'View':<24}{'Plans':<7}{'Correct':<9}{'Wrong':<7}{'Missing':<9}{'FalsePos':<10}"
          f"{'Accuracy':<10}{'Complete':<10}{'Conflict':<10}{'MAE':<9}{'MAPE':<8}")
    print("-" * 113)
    for pipeline in all_pipelines:
        pipeline_results = [
            per_pipeline[pipeline] for per_pipeline in all_results.values() if pipeline in per_pipeline
        ]
        gated = boundary_gated_aggregates(pipeline_results)
        for view_name, view_label in (("all_plans", "ALL-PLANS"), ("boundary_resolved_only", "BOUNDARY-RESOLVED-ONLY")):
            agg = gated[view_name]
            c = agg["counts"]
            print(
                f"{pipeline:<10}{view_label:<24}{agg['plan_count']:<7}{c['CORRECT']:<9}{c['WRONG']:<7}"
                f"{c['MISSING']:<9}{c['FALSE_POSITIVE']:<10}{_fmt_pct(agg['accuracy']):<10}"
                f"{_fmt_pct(agg['completeness']):<10}{_fmt_pct(agg['conflict_rate']):<10}"
                f"{_fmt_num(agg['mae']):<9}{_fmt_num(agg['mape']):<8}"
            )

    print("\nPer-field detail (primary pipeline: fusion for PDF plans, cv for DXF plans):")
    for display_key, per_pipeline in all_results.items():
        primary = next((per_pipeline[p] for p in _PRIMARY_PIPELINE_PREFERENCE if p in per_pipeline), None)
        if primary is None or primary.error:
            if primary is not None:
                print(f"\n  {display_key}: ERROR: {primary.error}")
            continue
        print(f"\n  {display_key} (runtime {_fmt_secs(primary.elapsed_seconds)}):")
        for fr in primary.field_results:
            expected_str = _fmt_num(fr.expected) if fr.expected is not None else "MUST_BE_MISSING"
            if fr.actual is not None:
                actual_str = _fmt_num(fr.actual)
            elif fr.verdict == "N/A":
                actual_str = "N/A (no native text in source)"
            else:
                actual_str = "MISSING"
            status_str = f" [{fr.status}]" if fr.status else ""
            error_str = ""
            if fr.abs_error is not None:
                pct_str = f", {fr.pct_error:.1f}%" if fr.pct_error is not None else ""
                error_str = f" (error={fr.abs_error:.3f}{pct_str})"
            print(f"    {fr.field:<30} expected={expected_str:<16} actual={actual_str:<12} {fr.verdict}{error_str}{status_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plans", nargs="*", default=None, help="Limit to specific plan ids (e.g. PLAN2 PLAN5).")
    parser.add_argument("--output", type=Path, default=None, help="Write full results as JSON here.")
    args = parser.parse_args()

    plans = discover_plans()
    if args.plans:
        wanted = set(args.plans)
        plans = [p for p in plans if p[0] in wanted]

    if not plans:
        print(
            f"No plans found with a .pdf or .dxf plus a matching .expected.json in {TEST_PLANS_DIR}. "
            "Add a data/test_plans/PLANn.expected.json (see PLAN2.expected.json for the field "
            "schema) plus a PLANn.pdf and/or PLANn.dxf for any plan you want scored -- no code "
            "changes needed.",
            file=sys.stderr,
        )
        sys.exit(1)

    settings = get_settings()
    plan_labels = ", ".join(f"{plan_id}[{doc_type}]" for plan_id, doc_type, _, _ in plans)
    print(f"vision_enabled={settings.vision_enabled}  |  {len(plans)} plan(s) with ground truth: {plan_labels}")

    all_results: dict[str, dict[str, PlanResult]] = {}
    for plan_id, doc_type, doc_path, expected_path in plans:
        display_key = f"{plan_id}[{doc_type}]"
        print(f"\nEvaluating {display_key}...", flush=True)
        all_results[display_key] = evaluate_plan(plan_id, doc_type, doc_path, expected_path)

    _print_report(all_results)

    if args.output:
        serializable = {
            display_key: {
                pipeline: {
                    "doc_type": result.doc_type,
                    "error": result.error,
                    "elapsed_seconds": result.elapsed_seconds,
                    "counts": result.counts(),
                    "accuracy": result.accuracy(),
                    "completeness": result.completeness(),
                    "conflict_rate": result.conflict_rate(),
                    "false_positive_rate": result.false_positive_rate(),
                    "mae": result.mae_mape()[0],
                    "mape": result.mae_mape()[1],
                    "boundary_resolved": result.boundary_resolved(),
                    "fields": [
                        {
                            "field": fr.field, "expected": fr.expected, "actual": fr.actual, "verdict": fr.verdict,
                            "status": fr.status, "abs_error": fr.abs_error, "pct_error": fr.pct_error,
                        }
                        for fr in result.field_results
                    ],
                }
                for pipeline, result in per_pipeline.items()
            }
            for display_key, per_pipeline in all_results.items()
        }
        all_pipelines: list[str] = []
        for per_pipeline in all_results.values():
            for pipeline in per_pipeline:
                if pipeline not in all_pipelines:
                    all_pipelines.append(pipeline)
        aggregates = {
            pipeline: boundary_gated_aggregates(
                [per_pipeline[pipeline] for per_pipeline in all_results.values() if pipeline in per_pipeline]
            )
            for pipeline in all_pipelines
        }
        args.output.write_text(
            json.dumps({"plans": serializable, "aggregates": aggregates}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
