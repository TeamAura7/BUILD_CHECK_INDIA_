"""
HTTP API layer for the BUILDCheck India compliance pipeline.

ADDITIVE ONLY. This file does not modify or reimplement the logic in
backend/tools/run_full_compliance.py -- it imports and calls the exact same
underlying functions/classes, in the exact same order:

    PDFHybridExtractor -> build_normalized_plan -> JsonFileRuleEngine

The only difference from the CLI script is that progress is appended to an
in-memory job log instead of printed to stdout, and results are returned as
JSON over HTTP instead of written to a file, so the frontend can run the
pipeline and poll for status.
"""

from __future__ import annotations

import json
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Response
from pydantic import BaseModel

from backend.config import get_settings
from backend.compliance.engine import JsonFileRuleEngine
from backend.schemas.enums import ConfidenceLevel
from backend.cv_extraction.dxf_extractor import DXFHybridExtractor
from backend.cv_extraction.pdf_extractor import PDFHybridExtractor
from backend.spatial_reasoning.pipeline import build_normalized_plan
from backend.schemas.vision import VisionDocumentResult
from backend.schemas.evidence import Confidence, ValueField
from backend.schemas.units import UnitValue
from backend.spatial_reasoning.final_fusion import build_final_agreement
from backend.tools.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["compliance"])

# Numeric plan fields that the frontend may edit and re-check. Regulatory
# context fields are handled separately below.
EDITABLE_NUMERIC_FIELDS = {
    "plot.width", "plot.depth", "plot.area",
    "building.width", "building.depth", "building.footprint_area", "building.floor_count",
    "road.width", "setbacks.front", "setbacks.rear",
    "setbacks.left", "setbacks.right", "coverage", "far",
    "building_height_estimated", "building_height_excluding_stilt",
    # Frontend uses the canonical nested display paths for these root-level
    # NormalizedPlan fields. Keep both spellings editable for compatibility.
    "building.height_estimated", "building.height_excluding_stilt",
}

# --- in-memory job store -----------------------------------------------
# Single-process, in-memory. Sufficient for a capstone demo/local
# deployment. Swap for Redis/DB if this ever needs multiple workers.
_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def _new_job() -> str:
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "id": job_id,
            "status": "pending",  # pending | running | done | error
            "created_at": datetime.now(timezone.utc).isoformat(),
            "log": [],
            "result": None,
            "error": None,
            "source_paths": {},
        }
    return job_id


def _log(job_id: str, message: str) -> None:
    with _JOBS_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id]["log"].append(
                {"t": datetime.now(timezone.utc).isoformat(), "message": message}
            )
    logger.info(message)


def _set_status(job_id: str, status: str) -> None:
    with _JOBS_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id]["status"] = status


def _fmt(vf: Any) -> dict[str, Any]:
    if vf is None:
        return {"value": None, "unit": None, "confidence": None, "source": None, "edited": False, "conflict": None}
    conf = getattr(vf, "confidence", None)
    normalized = getattr(vf, "normalized_value", None)
    source = getattr(vf, "source", None)
    conflict = getattr(vf, "conflict", None)
    return {
        "value": getattr(vf, "value", None),
        "unit": getattr(vf, "unit", None) or (getattr(normalized, "unit", None) if normalized is not None else None),
        "confidence": conf.level.value if conf is not None else None,
        "source": source,
        # Surfaced separately from `source` so the frontend can show a
        # simple "edited" badge without string-matching on provenance text
        # (Section 13: manually edited values must be distinguishable from
        # extracted ones in the evidence model).
        "edited": source == "USER_EDIT",
        # Populated only for a PDF<->DXF cross-check conflict (see
        # backend/spatial_reasoning/pdf_dxf_reconciliation.py) -- lets the
        # frontend flag a field regardless of its confidence level, since a
        # PDF-wins conflict deliberately does NOT set confidence.level to
        # CONFLICTING (see that module's own docstring for why).
        "conflict": None if conflict is None else {
            "description": conflict.description,
            "conflicting_raw_values": [v.model_dump(mode="json") for v in conflict.conflicting_raw_values],
            "conflicting_sources": conflict.conflicting_sources,
        },
    }


def _plan_summary_dict(plan: Any) -> dict[str, Any]:
    return {
        "plot": {
            "width": _fmt(plan.plot.width),
            "depth": _fmt(plan.plot.depth),
            "area": _fmt(plan.plot.area),
        },
        "building": {
            "width": _fmt(plan.building.width),
            "depth": _fmt(plan.building.depth),
            "footprint_area": _fmt(plan.building.footprint_area),
            "floor_count": _fmt(getattr(plan.building, "floor_count", None)),
            "use": _fmt(getattr(plan, "building_use", None)),
        },
        "road": {"width": _fmt(plan.road.width)},
        "setbacks": {
            "front": _fmt(plan.setbacks.front),
            "rear": _fmt(plan.setbacks.rear),
            "left": _fmt(plan.setbacks.left),
            "right": _fmt(plan.setbacks.right),
        },
        "coverage": _fmt(plan.coverage),
        "far": _fmt(plan.far),
        "floor_areas": {k: _fmt(v) for k, v in getattr(plan, "floor_areas", {}).items()},
        "building_height_estimated": _fmt(getattr(plan, "building_height_estimated", None)),
        "building_height_excluding_stilt": _fmt(getattr(plan, "building_height_excluding_stilt", None)),
        "metadata": getattr(plan, "metadata", {}),
    }




def _suggestions_for_results(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    suggestions = []
    for r in results:
        if r.get("status") != "FAIL":
            continue
        field = r.get("source_field", "")
        msg = {
            "setbacks.front": "Increase the front setback or move the building footprint inward until the cited minimum is satisfied.",
            "setbacks.rear": "Increase the rear setback or reduce the building depth while keeping the plot boundary unchanged.",
            "setbacks.left": "Increase the left-side setback by shifting/narrowing the building footprint.",
            "setbacks.right": "Increase the right-side setback by shifting/narrowing the building footprint.",
            "coverage": "Reduce ground-floor footprint or increase compliant plot area to bring coverage below the cited maximum.",
            "far": "Reduce total built-up floor area or revise the floor schedule to satisfy the cited FAR limit.",
            "road.width": "Verify the road/access dimension and revise the access arrangement if the cited minimum is not met.",
            "plot.width": "Verify the surveyed plot width and revise the site layout if the cited minimum is not met.",
            "plot.depth": "Verify the surveyed plot depth and revise the site layout if the cited minimum is not met.",
            "plot.area": "Verify the surveyed plot area against title/survey information before redesigning the building.",
        }.get(field, "Review the cited regulation and revise the affected plan measurement/design until the requirement is satisfied.")
        suggestions.append({"field": field, "suggestion": msg, "rule": r.get("rule_description", "")})
    return suggestions

def _finish_job(
    job_id: str,
    pdf_path: Path,
    municipality: str,
    mode: str,
    plan: Any,
    fusion_report: dict[str, Any],
    settings: Any,
    *,
    dxf_path: Optional[Path] = None,
    pdf_plan: Optional[Any] = None,
    dxf_plan: Optional[Any] = None,
    reconciliation_report: Optional[Any] = None,
) -> None:
    """
    Shared tail of the pipeline: deterministic rule evaluation + report
    assembly + job store write. Identical for every extraction path (PDF
    cv/vision/fusion modes AND the DXF path) -- compliance checking never
    depends on which extractor produced the NormalizedPlan.

    The four keyword-only params are additive, for the PDF+DXF dual-source
    cross-check (backend/spatial_reasoning/pdf_dxf_reconciliation.py) --
    all default to `None`, so both existing single-file call sites are
    completely unaffected. `plan` is still the plan compliance is actually
    evaluated against (the reconciled plan in dual-source mode); `pdf_plan`/
    `dxf_plan`/`reconciliation_report`, when given, are recorded in full so
    neither source's own independent reading is ever silently discarded.
    """
    _log(job_id, "[2/4] Loading authoritative BBMP deterministic ruleset...")
    engine = JsonFileRuleEngine(settings=settings)
    rules = engine.load_ruleset(municipality)
    _log(job_id, f"  Active deterministic rules loaded: {len(rules)}")
    _log(job_id, f"  Ruleset version: {rules[0].version if rules else 'none'}")
    _log(job_id, "  Runtime LLM rule generation: DISABLED")

    _log(job_id, "[3/4] Resolving applicable rules and evaluating deterministically...")
    compliance = engine.evaluate_plan(plan, municipality)
    results: list[dict[str, Any]] = []
    for rr in compliance.rule_results:
        payload = json.loads(rr.model_dump_json())
        results.append(payload)
        if rr.status.value != "NOT_APPLICABLE":
            _log(
                job_id,
                f"  {rr.status.value}: {rr.source_field or rr.rule_id} | "
                f"{rr.required_value_description or rr.rule_description}",
            )

    _log(
        job_id,
        f"[4/4] Done. Municipality={municipality}  Active rules={len(rules)} "
        f"Checks={len(results)}  OVERALL={compliance.overall_status.value}",
    )

    report = {
        "pipeline": {
            "name": "architectural_plan_to_deterministic_compliance",
            "municipality": municipality,
            "source_pdf": pdf_path.name,
            "source_format": pdf_path.suffix.lower().lstrip("."),
            "extraction_mode": mode,
            "rule_source": f"data/runtime_rules/{municipality}/rules.json",
            "rule_generation": "none_at_runtime",
            "rag_role": "not_authoritative; optional explanation/retrieval only",
            "final_decision": "JsonFileRuleEngine + DeterministicRuleEvaluator",
            "live_ruleset_modified": False,
        },
        "plan": plan.model_dump(mode="json"),
        "plan_summary": _plan_summary_dict(plan),
        "ruleset": {"active_rule_count": len(rules), "version": rules[0].version if rules else None},
        "retrieval": {},
        "draft_rules": [],
        "fusion": fusion_report,
        "suggestions": _suggestions_for_results(results),
        "compliance": {"overall_status": compliance.overall_status.value, "rule_results": results, "counts": compliance.count_by_status()},
    }

    if dxf_path is not None:
        report["pipeline"]["source_dxf"] = dxf_path.name
        report["pipeline"]["source_format"] = "pdf+dxf"
        report["pipeline"]["cross_check"] = "pdf_vs_dxf_post_hoc_reconciliation"
        if reconciliation_report is not None:
            report["pdf_dxf_reconciliation"] = reconciliation_report.model_dump(mode="json")
        if pdf_plan is not None:
            report["pdf_plan"] = pdf_plan.model_dump(mode="json")
            report["pdf_plan_summary"] = _plan_summary_dict(pdf_plan)
        if dxf_plan is not None:
            report["dxf_plan"] = dxf_plan.model_dump(mode="json")
            report["dxf_plan_summary"] = _plan_summary_dict(dxf_plan)

    with _JOBS_LOCK:
        _JOBS[job_id]["result"] = report
    _set_status(job_id, "done")


def _build_dxf_plan(job_id: str, dxf_path: Path, settings: Any) -> tuple[Any, dict[str, Any]]:
    """DXF's own extraction path -- extracted verbatim from `_run_pipeline`
    (behavior-preserving refactor, Phase 2 of the PDF+DXF cross-check) so
    it can be called either as the sole path (single-file DXF upload) or
    alongside `_build_pdf_plan` (dual-source upload), unchanged either way.

    DXF vector geometry is exact (real-world coordinates, no
    rasterization/OCR/scale-calibration), so it never goes through the
    CV/Vision/fusion mode selection `_build_pdf_plan` implements -- it
    always uses `DXFHybridExtractor` and the same `build_normalized_plan()`
    used by the PDF path, so compliance checking is identical regardless of
    source format. `DXFHybridExtractor` still consults
    `settings.vision_enabled` itself for its own Vision-render fallback --
    used only when the deterministic vector-geometry search finds no
    plausible building/road polygon at all.
    """
    document_id = dxf_path.stem
    _log(
        job_id,
        "Extraction mode: DXF (deterministic vector geometry"
        + (", with a Vision-render fallback for regions no closed polygon was found for"
           if settings.vision_enabled else "; Vision not enabled for this run")
        + ")",
    )
    _log(job_id, "[1/4] Extracting plot/building/road geometry directly from DXF vector data...")
    extraction = DXFHybridExtractor().extract(dxf_path, document_id)
    measurement_count = len(extraction.independent_cv.measurements) if extraction.independent_cv else 0
    _log(
        job_id,
        f"Extracted {measurement_count} measurement(s) directly from DXF vector geometry "
        f"({len(extraction.dimensions)} DIMENSION entity value(s), {len(extraction.text_evidence)} text label(s)).",
    )
    for warning in extraction.warnings:
        _log(job_id, f"WARNING: {warning}")
    plan = build_normalized_plan(extraction, plan_id=f"plan-{document_id}")
    _log(job_id, "Resolved NormalizedPlan directly from DXF vector geometry (independent CV path).")
    fusion_report = {
        "mode": "dxf",
        "note": (
            "DXF vector geometry is exact -- no rasterization, OCR, or scale calibration was "
            "performed. Values were read directly from drawing coordinates and native DXF "
            "text/DIMENSION entities."
        ),
        "measurement_count": measurement_count,
    }
    return plan, fusion_report


def _build_pdf_plan(
    job_id: str, pdf_path: Path, document_id: str, mode: Optional[str], vision: bool, settings: Any,
) -> tuple[Any, str, dict[str, Any]]:
    """PDF's own cv/vision/fusion extraction path -- extracted verbatim
    from `_run_pipeline` (behavior-preserving refactor, Phase 2 of the
    PDF+DXF cross-check). Returns `(plan, resolved_mode, fusion_report)` --
    callers must use the RETURNED `resolved_mode` for `_finish_job`, not
    their own input `mode`, since this function normalizes/defaults it
    exactly as `_run_pipeline` used to inline.

    `mode` is the dashboard's explicit "Extraction Mode" (Section 7):
      cv     -> native/OpenCV only, no Vision call at all.
      vision -> the selected Vision backend only; CV must not influence
                the result (see build_vision_only_plan).
      fusion -> both, independently, reconciled by the existing CV+Vision
                agreement layer (reuses run_validation's underlying
                `validate_extractions` service).
    """
    mode = (mode or ("fusion" if vision else "cv")).lower()
    if mode not in {"cv", "vision", "fusion"}:
        mode = "fusion" if vision else "cv"
    vision = mode in {"vision", "fusion"}
    # settings.vision_enabled/vision_backend/vision_max_new_tokens are
    # resolved by the caller before this function runs.

    if settings.vision_enabled:
        from backend.vision_extraction import get_vision_extractor

        extractor = get_vision_extractor()
        _log(job_id, f"Vision backend: {type(extractor).__name__} ({extractor.model_name})")

    _log(job_id, f"Extraction mode: {mode.upper()}")

    fusion_report: dict[str, Any] = {"summary": {}}

    if mode == "vision":
        # Vision mode: run ONLY the vision backend. CV/native/OpenCV is
        # never invoked for this path (Section 6) -- not even the
        # "independent CV" validation path used elsewhere.
        _log(job_id, "[1/4] Extracting architectural parameters (Vision backend only, no CV)...")
        from backend.vision_extraction import get_vision_extractor
        vision_result = get_vision_extractor().analyze_pdf(pdf_path, ground_against_native_text=True)
        for warning in vision_result.warnings:
            _log(job_id, f"WARNING: {warning}")
        _log(job_id, f"Extracted {sum(len(p.dimensions) for p in vision_result.pages)} vision dimension(s) across {len(vision_result.pages)} page(s). No CV geometry was used.")
        from backend.spatial_reasoning.final_fusion import build_vision_only_plan
        plan = build_vision_only_plan(vision_result, plan_id=f"plan-{document_id}", document_id=document_id)
        _log(job_id, "Resolved NormalizedPlan from Vision evidence only.")
        fusion_report = {"mode": "vision", "note": "CV was not run in Vision mode; no CV/Vision agreement layer applies.", "vision_pages": len(vision_result.pages)}

    elif mode == "fusion":
        # Fusion mode: run CV and Vision independently, then reconcile
        # via the SAME shared service `run_validation` (backend.validation.
        # validate_extractions) uses -- not a second/duplicate fusion
        # algorithm. We pass in the extraction we're about to build below
        # so validate_extractions does not redundantly re-run CV/Vision.
        _log(job_id, "[1/4] Extracting architectural parameters (native/OpenCV + Vision, independently)...")
        extraction = PDFHybridExtractor().extract(pdf_path, document_id)
        _log(
            job_id,
            f"Extracted {len(extraction.dimensions)} native/CV dimension(s), "
            f"{len(extraction.plot_candidates)} plot candidate(s), "
            f"{len(extraction.building_candidates)} building candidate(s), "
            f"{len(extraction.vision_pages)} vision page result(s).",
        )
        for warning in extraction.warnings:
            _log(job_id, f"WARNING: {warning}")

        plan = build_normalized_plan(extraction, plan_id=f"plan-{document_id}")
        _log(job_id, "Resolved NormalizedPlan (CV + Vision agreement layer).")
        vision_doc = VisionDocumentResult(pages=extraction.vision_pages, model_name="pipeline", enabled=bool(extraction.vision_pages))
        try:
            from backend.validation import validate_extractions
            validation_report = validate_extractions(
                pdf_path,
                document_id=document_id,
                independent_cv_result=extraction.independent_cv,
                vision_result=vision_doc,
            )
            fusion_report = {
                "mode": "fusion",
                "reused_service": "backend.validation.validate_extractions (same function backend.tools.run_validation calls)",
                "fields": [f.model_dump(mode="json") for f in validation_report.fields],
                "summary": validation_report.summary,
                "final_agreed_values": validation_report.final_agreed_values,
                "document_verified_fusion": validation_report.document_verified_fusion,
            }
            _log(job_id, f"Fusion (via run_validation service): {len(validation_report.fields)} fields compared; {validation_report.summary.get('CONFLICT', 0)} conflicts.")
        except Exception as exc:
            fusion_report = {"mode": "fusion", "error": str(exc)}
            _log(job_id, f"WARNING: document-verified fusion unavailable: {exc}")

    else:  # mode == "cv"
        _log(job_id, "[1/4] Extracting architectural parameters (native/OpenCV only, no Vision call)...")
        extraction = PDFHybridExtractor(enable_vision=False).extract(pdf_path, document_id)
        _log(
            job_id,
            f"Extracted {len(extraction.dimensions)} native/CV dimension(s), "
            f"{len(extraction.plot_candidates)} plot candidate(s), "
            f"{len(extraction.building_candidates)} building candidate(s).",
        )
        for warning in extraction.warnings:
            _log(job_id, f"WARNING: {warning}")
        plan = build_normalized_plan(extraction, plan_id=f"plan-{document_id}")
        _log(job_id, "Resolved NormalizedPlan (CV only; no Vision call was made).")
        fusion_report = {"mode": "cv", "note": "Vision was not run in CV mode; no CV/Vision agreement layer applies."}

    return plan, mode, fusion_report


def _run_pipeline(
    job_id: str,
    pdf_path: Path,
    municipality: str,
    vision: bool,
    backend_choice: Optional[str],
    max_new_tokens: Optional[int],
    top_k: Optional[int],
    all_fields: bool,
    mode: str = "cv",
    dxf_path: Optional[Path] = None,
) -> None:
    try:
        _set_status(job_id, "running")
        settings = get_settings()

        document_id = pdf_path.stem
        is_dxf = pdf_path.suffix.lower() == ".dxf"

        # Recorded so `/jobs/{id}/edit` can copy the original source file(s)
        # into a correction record later (backend/corpus/corrections.py) --
        # the job's in-memory result alone doesn't retain where the upload
        # was saved, and `upload_dir` may since have been cleared.
        source_paths: dict[str, str] = {}
        if pdf_path.suffix.lower() == ".pdf":
            source_paths["pdf"] = str(pdf_path)
        elif pdf_path.suffix.lower() == ".dxf":
            source_paths["dxf"] = str(pdf_path)
        if dxf_path is not None:
            source_paths["dxf"] = str(dxf_path)
        with _JOBS_LOCK:
            if job_id in _JOBS:
                _JOBS[job_id]["source_paths"] = source_paths

        # Resolve the Vision toggle from this REQUEST before any branch
        # below runs. This used to happen only in the PDF branch further
        # down, which meant a DXF upload with "Vision"/"Fusion" selected in
        # the dashboard silently ignored that choice -- `settings.vision_enabled`
        # stayed at whatever the server's own startup config happened to be,
        # never the per-request toggle -- so DXFHybridExtractor's Vision-
        # render fallback (`_vision_fallback_regions`) never actually ran
        # from a real upload no matter what the user picked.
        vision_requested = bool(vision) or (mode or "").lower() in {"vision", "fusion"}
        if vision_requested:
            settings.vision_enabled = True
        if backend_choice:
            settings.vision_backend = backend_choice
        if max_new_tokens:
            settings.vision_max_new_tokens = max_new_tokens

        if dxf_path is not None:
            # Dual-source mode (PDF+DXF cross-check): each extractor runs
            # completely independently, exactly as it would alone -- see
            # backend/spatial_reasoning/pdf_dxf_reconciliation.py's own
            # module docstring for why neither extractor's internals are
            # touched or made aware of the other's input. Only the POST-HOC
            # reconciliation step below ever looks at both plans together.
            _log(
                job_id,
                "Dual-source mode: PDF and DXF are extracted completely independently "
                "(see backend/spatial_reasoning/pdf_dxf_reconciliation.py); neither "
                "extractor's internals are touched or made aware of the other's input.",
            )
            pdf_plan, resolved_mode, pdf_fusion_report = _build_pdf_plan(
                job_id, pdf_path, document_id, mode, vision, settings,
            )
            dxf_plan, _dxf_fusion_report = _build_dxf_plan(job_id, dxf_path, settings)

            from backend.spatial_reasoning.pdf_dxf_reconciliation import reconcile_pdf_dxf

            reconciled_plan, reconciliation_report = reconcile_pdf_dxf(pdf_plan, dxf_plan)
            _log(job_id, f"Cross-check complete: {reconciliation_report.summary}")
            return _finish_job(
                job_id, pdf_path, municipality, resolved_mode, reconciled_plan, pdf_fusion_report, settings,
                dxf_path=dxf_path, pdf_plan=pdf_plan, dxf_plan=dxf_plan, reconciliation_report=reconciliation_report,
            )

        # --- DXF: a separate, simpler path -----------------------------
        if is_dxf:
            mode = "dxf"
            plan, fusion_report = _build_dxf_plan(job_id, pdf_path, settings)
            return _finish_job(job_id, pdf_path, municipality, mode, plan, fusion_report, settings)

        plan, mode, fusion_report = _build_pdf_plan(job_id, pdf_path, document_id, mode, vision, settings)
        _finish_job(job_id, pdf_path, municipality, mode, plan, fusion_report, settings)

    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        _log(job_id, f"FATAL ERROR: {exc}")
        with _JOBS_LOCK:
            if job_id in _JOBS:
                _JOBS[job_id]["error"] = str(exc)
                _JOBS[job_id]["traceback"] = tb
        _set_status(job_id, "error")


# --- routes --------------------------------------------------------------


@router.get("/municipalities")
def list_municipalities() -> dict[str, Any]:
    settings = get_settings()
    munis: set[str] = set()
    for base in (settings.vector_store_dir, settings.regulations_dir, settings.runtime_rules_dir):
        p = Path(base)
        if p.exists():
            munis.update(d.name for d in p.iterdir() if d.is_dir())
    return {"municipalities": sorted(munis)}


@router.get("/sample-plans")
def list_sample_plans() -> dict[str, Any]:
    settings = get_settings()
    p = Path(settings.test_plans_dir)
    plans = sorted(f.name for f in (list(p.glob("*.pdf")) + list(p.glob("*.dxf")))) if p.exists() else []
    return {"sample_plans": plans}


class AnalyzeSampleRequest(BaseModel):
    sample_plan: str
    # Optional: a second bundled sample plan (e.g. sample_plan="PLAN5.pdf",
    # second_sample_plan="PLAN5.dxf") to cross-check against the first --
    # see backend/spatial_reasoning/pdf_dxf_reconciliation.py. Must be the
    # opposite format (one PDF, one DXF); order-agnostic.
    second_sample_plan: Optional[str] = None
    municipality: str = "BBMP"
    vision: bool = False
    backend: Optional[str] = None
    top_k: Optional[int] = None
    all_fields: bool = False
    mode: Optional[str] = None  # "cv" | "vision" | "fusion" -- see Section 7


def _resolve_pdf_dxf_pair(first_path: Path, second_path: Optional[Path]) -> tuple[Path, Optional[Path]]:
    """Given a primary path and an optional second path, return
    `(pdf_path, dxf_path)` -- order-agnostic (the caller may have picked
    either one as "primary"). Raises `HTTPException(400)` if both share the
    same extension, since a cross-check needs exactly one of each."""
    if second_path is None:
        return first_path, None
    first_ext, second_ext = first_path.suffix.lower(), second_path.suffix.lower()
    if first_ext == second_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Provide one PDF and one DXF to cross-check -- got two {first_ext} files.",
        )
    if first_ext == ".pdf":
        return first_path, second_path
    return second_path, first_path


@router.post("/analyze/sample")
def analyze_sample(payload: AnalyzeSampleRequest) -> dict[str, Any]:
    settings = get_settings()
    first_path = Path(settings.test_plans_dir) / payload.sample_plan
    if not first_path.exists() or first_path.suffix.lower() not in {".pdf", ".dxf"}:
        raise HTTPException(status_code=404, detail=f"Sample plan not found: {payload.sample_plan}")

    second_path: Optional[Path] = None
    if payload.second_sample_plan:
        second_path = Path(settings.test_plans_dir) / payload.second_sample_plan
        if not second_path.exists() or second_path.suffix.lower() not in {".pdf", ".dxf"}:
            raise HTTPException(status_code=404, detail=f"Sample plan not found: {payload.second_sample_plan}")

    pdf_path, dxf_path = _resolve_pdf_dxf_pair(first_path, second_path)

    job_id = _new_job()
    thread = threading.Thread(
        target=_run_pipeline,
        args=(
            job_id,
            pdf_path,
            payload.municipality.upper(),
            payload.vision,
            payload.backend,
            None,
            payload.top_k,
            payload.all_fields,
            payload.mode,
        ),
        kwargs={"dxf_path": dxf_path},
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id}


async def _save_upload(upload: UploadFile, upload_dir: Path, max_bytes: int) -> Path:
    safe_name = f"{uuid.uuid4().hex[:8]}_{Path(upload.filename).name}"
    dest = upload_dir / safe_name
    contents = await upload.read()
    if len(contents) > max_bytes:
        raise HTTPException(status_code=413, detail="File too large.")
    dest.write_bytes(contents)
    return dest


@router.post("/analyze/upload")
async def analyze_upload(
    file: UploadFile = File(...),
    # Optional: a second file (the matching PDF/DXF of the SAME plan) to
    # cross-check against the first -- see backend/spatial_reasoning/
    # pdf_dxf_reconciliation.py. Must be the opposite format; order of the
    # two files never matters (auto-detected by extension below).
    second_file: Optional[UploadFile] = File(None),
    municipality: str = Form("BBMP"),
    vision: bool = Form(False),
    backend: Optional[str] = Form(None),
    top_k: Optional[int] = Form(None),
    all_fields: bool = Form(False),
    mode: Optional[str] = Form(None),
) -> dict[str, Any]:
    settings = get_settings()
    if not file.filename or not file.filename.lower().endswith((".pdf", ".dxf")):
        raise HTTPException(status_code=400, detail="Only PDF or DXF files are supported.")
    if second_file is not None and second_file.filename and not second_file.filename.lower().endswith((".pdf", ".dxf")):
        raise HTTPException(status_code=400, detail="The second file must also be a PDF or DXF.")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    first_dest = await _save_upload(file, upload_dir, max_bytes)
    second_dest: Optional[Path] = None
    if second_file is not None and second_file.filename:
        second_dest = await _save_upload(second_file, upload_dir, max_bytes)

    pdf_path, dxf_path = _resolve_pdf_dxf_pair(first_dest, second_dest)

    job_id = _new_job()
    thread = threading.Thread(
        target=_run_pipeline,
        args=(
            job_id,
            pdf_path,
            municipality.upper(),
            vision,
            backend,
            None,
            top_k,
            all_fields,
            mode,
        ),
        kwargs={"dxf_path": dxf_path},
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id}



class EditPlanRequest(BaseModel):
    updates: dict[str, dict[str, Any]]
    # Both optional and purely for correction-capture provenance (see
    # backend/corpus/corrections.py) -- omitting them still edits the plan
    # exactly as before, just with an anonymous/unnoted correction record.
    reviewer: Optional[str] = None
    note: Optional[str] = None


# Shared between `_set_plan_value` (mutates) and `_snapshot_plan_value`
# (reads, for correction-capture) so the two can never drift apart on which
# field name means which model attribute.
_EDITABLE_CONTEXT_FIELDS = {"building_use", "development_area"}
_FIELD_ALIASES = {
    "building.height_estimated": "building_height_estimated",
    "building.height_excluding_stilt": "building_height_excluding_stilt",
}


def _snapshot_plan_value(plan: Any, field: str) -> Optional[Any]:
    """The plan's current value for `field`, read-only, for recording what the
    pipeline predicted BEFORE a correction overwrites it. Returns None if the
    field is missing/unset/unresolvable -- never raises, since a failed
    snapshot must not block the edit itself."""
    if field in _EDITABLE_CONTEXT_FIELDS:
        return getattr(plan, field, None)
    parts = _FIELD_ALIASES.get(field, field).split(".")
    obj = plan
    for part in parts[:-1]:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return getattr(obj, parts[-1], None)


def _set_plan_value(plan: Any, field: str, payload: dict[str, Any]) -> None:
    if field not in EDITABLE_NUMERIC_FIELDS and field not in _EDITABLE_CONTEXT_FIELDS:
        raise ValueError(f"Field is not editable: {field}")
    if field in _EDITABLE_CONTEXT_FIELDS:
        from backend.schemas.normalized_plan import NormalizedPlan
        raw = payload.get("value")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"Missing value for {field}")
        if field == "development_area":
            value = raw.strip().upper()
            if value not in {"A", "B", "C"}:
                raise ValueError("development_area must be A, B, or C")
        else:
            value = NormalizedPlan._normalize_building_use(raw)
            if value is None:
                raise ValueError("Unsupported building_use classification")
        setattr(plan, field, ValueField[str](
            value=value,
            confidence=Confidence(level=ConfidenceLevel.HIGH, reason="Manually verified/edited in frontend."),
            source="USER_EDIT",
        ))
        return
    # The API exposes these two height fields at the NormalizedPlan root,
    # while the frontend displays them using the nested `building.*` names.
    # Normalize the UI path before resolving the actual model attribute.
    canonical_field = _FIELD_ALIASES.get(field, field)
    parts = canonical_field.split(".")
    obj = plan
    for part in parts[:-1]:
        obj = getattr(obj, part)
    name = parts[-1]
    current = getattr(obj, name)
    value = payload.get("value")
    if value is None:
        raise ValueError(f"Missing value for {field}")
    default_units = {
        "plot.area": "m²",
        "building.footprint_area": "m²",
        "coverage": "%",
        "far": "",
        "building.floor_count": "floors",
    }
    unit = payload.get("unit") or getattr(current, "unit", None) or default_units.get(field, "m")
    # `current` is None (not a ValueField with MISSING confidence) for
    # optional fields the extractor never touched at all -- e.g.
    # building_height_estimated when no height was detected anywhere in
    # the plan. `type(current)` would be NoneType in that case, so fall
    # back to ValueField[float] explicitly rather than assuming `current`
    # is always already a ValueField instance.
    if field == "building.floor_count":
        value_cls = type(current) if current is not None else ValueField[int]
        parsed_value = int(value)
    else:
        value_cls = type(current) if current is not None else ValueField[float]
        parsed_value = float(value)

    # Manual edits must be validated before they ever reach the rule engine
    # (Section 14): reject negative setbacks/areas/widths/depths and
    # non-positive floor counts outright rather than silently coercing them.
    canonical_for_check = _FIELD_ALIASES.get(field, field)
    if canonical_for_check == "building.floor_count":
        if parsed_value < 1:
            raise ValueError("building.floor_count must be a positive integer (at least 1 floor).")
    elif canonical_for_check == "far":
        if parsed_value < 0:
            raise ValueError("far cannot be negative.")
    else:
        # Every other editable numeric field here is a physical
        # length/area/percentage/height -- none of which can be negative.
        if parsed_value < 0:
            raise ValueError(f"{field} cannot be negative.")

    setattr(obj, name, value_cls(
        value=parsed_value,
        normalized_value=UnitValue(magnitude=float(value), unit=unit),
        confidence=Confidence(level=ConfidenceLevel.HIGH, reason="Manually verified/edited in frontend."),
        source="USER_EDIT",
    ))


def _record_edit_as_corrections(
    job_id: str, plan_after: Any, payload: "EditPlanRequest", before_snapshots: dict[str, Any],
) -> None:
    """Best-effort: append one correction event per edited field
    (backend/corpus/corrections.py), using the plan's OWN before/after values
    (not the raw request body) so a correction record always reflects the
    same parsed/normalized value the plan itself now holds. Never raises -- a
    reviewer's edit must still succeed even if correction-capture itself has
    a problem (e.g. the upload was already cleaned up), so every failure here
    is logged and swallowed rather than surfaced as a 500 on `/jobs/{id}/edit`.
    """
    try:
        from backend.corpus.corrections import record_correction

        with _JOBS_LOCK:
            job = _JOBS.get(job_id) or {}
            source_paths = dict(job.get("source_paths") or {})
        document_id = getattr(plan_after, "source_document_id", job_id)
        for field in payload.updates:
            before = before_snapshots.get(field)
            after = _snapshot_plan_value(plan_after, field)
            corrected_value = getattr(after, "value", after)
            if corrected_value is None:
                continue
            predicted_confidence = getattr(getattr(before, "confidence", None), "level", None)
            normalized = getattr(after, "normalized_value", None)
            unit = getattr(after, "unit", None) or getattr(normalized, "unit", None) or ""
            record_correction(
                job_id=job_id, document_id=document_id, field=field,
                unit=str(unit),
                predicted_value=getattr(before, "value", before) if before is not None else None,
                predicted_confidence=getattr(predicted_confidence, "value", predicted_confidence),
                predicted_source=getattr(before, "source", None),
                corrected_value=corrected_value, corrected_by=payload.reviewer or "",
                note=payload.note or "", source_files={k: Path(v) for k, v in source_paths.items()},
            )
    except Exception:  # noqa: BLE001 - correction-capture must never break an edit
        logger.exception("Failed to record correction for job %s (edit itself still succeeded)", job_id)


@router.post("/jobs/{job_id}/edit")
def edit_plan(job_id: str, payload: EditPlanRequest) -> dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job or job.get("status") != "done" or not job.get("result"):
            raise HTTPException(status_code=404, detail="Completed job not found.")
        report = job["result"]
    from backend.schemas.normalized_plan import NormalizedPlan
    plan = NormalizedPlan.model_validate(report["plan"])
    before_snapshots = {field: _snapshot_plan_value(plan, field) for field in payload.updates}
    try:
        for field, update in payload.updates.items():
            _set_plan_value(plan, field, update)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    settings = get_settings()
    engine = JsonFileRuleEngine(settings=settings)
    compliance = engine.evaluate_plan(plan, report["pipeline"]["municipality"])
    results = [json.loads(rr.model_dump_json()) for rr in compliance.rule_results]
    report["plan"] = plan.model_dump(mode="json")
    report["plan_summary"] = _plan_summary_dict(plan)
    report["compliance"] = {"overall_status": compliance.overall_status.value, "rule_results": results, "counts": compliance.count_by_status()}
    report["suggestions"] = _suggestions_for_results(results)
    with _JOBS_LOCK:
        _JOBS[job_id]["result"] = report
    _record_edit_as_corrections(job_id, plan, payload, before_snapshots)
    return report


@router.get("/jobs/{job_id}/report.pdf")
def report_pdf(job_id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job or job.get("status") != "done" or not job.get("result"):
            raise HTTPException(status_code=404, detail="Completed job not found.")
        report = job["result"]
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=32, leftMargin=32, topMargin=32, bottomMargin=32)
    styles = getSampleStyleSheet()
    story = [Paragraph("BUILDCheck India — Compliance Report", styles["Title"]), Spacer(1, 8)]
    story.append(Paragraph(f"Plan: {report['pipeline']['source_pdf']} | Municipality: {report['pipeline']['municipality']}", styles["Normal"]))
    story.append(Paragraph(f"Overall status: <b>{report['compliance']['overall_status']}</b>", styles["Heading2"]))
    rows = [["Field", "Value", "Unit", "Confidence"]]
    for group, vals in report.get("plan_summary", {}).items():
        if not isinstance(vals, dict):
            continue
        for field, v in vals.items():
            if isinstance(v, dict):
                rows.append([f"{group}.{field}", str(v.get("value", "—")), str(v.get("unit", "") or ""), str(v.get("confidence", ""))])
    table = Table(rows, repeatRows=1)
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .3, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [table, Spacer(1, 12), Paragraph("Failed rules and suggestions", styles["Heading2"])]
    for item in report.get("suggestions", []):
        story.append(Paragraph(f"<b>{item['field']}</b>: {item['suggestion']}", styles["Normal"]))
    doc.build(story)
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=buildcheck_{job_id}.pdf"})

@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return dict(job)