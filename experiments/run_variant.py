"""
Experiment driver: extract ONE plan with ONE pipeline variant and save the
NormalizedPlan + run metadata. Lives OUTSIDE backend/ on purpose, so the
corpus cache fingerprint (sha256 over backend/*.py) is unaffected and no
production code is modified.

Variants (see experiments/README.md for exact definitions):
  pdf_hybrid              production PDF path: PDFHybridExtractor(enable_vision=False)
                          -> build_normalized_plan  (identical to run_corpus 'pdf')
  pdf_native_only         as pdf_hybrid, but OCR and OpenCV raster evidence are
                          stubbed to return nothing (native text + vector geometry only)
  pdf_native_ocr          as pdf_hybrid, but OpenCV raster evidence stubbed out (OCR kept)
  pdf_legacy_resolver     as pdf_hybrid, but final_fusion.apply_final_agreement_to_plan
                          is replaced by identity, so the legacy multi-candidate
                          evidence_reconciliation path ships
  pdf_evidence_decision   as pdf_hybrid with env USE_EVIDENCE_DECISION_ENGINE=true
                          (set by the orchestrator, a real config flag)
  dxf_hybrid              production DXF path (identical to run_corpus 'dxf')
  dxf_evidence_decision   dxf_hybrid with env USE_DXF_EVIDENCE_DECISION_ENGINE=true

Usage: python experiments/run_variant.py VARIANT PLAN_ID OUT_DIR
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RECON_LOG: list[dict] = []


def _install_stubs(variant: str) -> None:
    if variant in ("pdf_native_only",):
        from backend.cv_extraction import ocr_fallback
        ocr_fallback.ocr_page_any_orientation = lambda *a, **k: []
        ocr_fallback.ocr_page = lambda *a, **k: []
    if variant in ("pdf_native_only", "pdf_native_ocr"):
        from backend.cv_extraction import opencv_geometry
        opencv_geometry.geometry_evidence_for_page = lambda *a, **k: {"lines": [], "polygons": [], "rectangles": []}
    if variant == "pdf_legacy_resolver":
        from backend.spatial_reasoning import pipeline
        pipeline.apply_final_agreement_to_plan = lambda plan, cv, vision: plan
    # Instrument the within-document reconciliation (no behaviour change).
    from backend.spatial_reasoning import evidence_reconciliation as er
    orig = er.reconcile

    def logged(candidates):
        out = orig(candidates)
        o = out[0] if isinstance(out, tuple) else out   # CONFLICTING_EVIDENCE returns (outcome, Conflict)
        RECON_LOG.append({
            "n_candidates": len(candidates),
            "values": [c.value for c in candidates],
            "sources": [c.source for c in candidates],
            "status": o.status.value,
            "value": o.value,
        })
        return out
    er.reconcile = logged


def main() -> int:
    variant, plan_id, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    from backend.corpus.store import load_manifest
    entry = next(p for p in load_manifest().plans if p.id == plan_id)
    kind = "dxf" if variant.startswith("dxf") else "pdf"
    _install_stubs(variant)
    meta = {"variant": variant, "plan": plan_id, "kind": kind}
    t0 = time.time()
    try:
        from backend.spatial_reasoning.pipeline import build_normalized_plan
        path = REPO / getattr(entry, kind)
        if kind == "pdf":
            from backend.cv_extraction.pdf_extractor import PDFHybridExtractor
            extraction = PDFHybridExtractor(enable_vision=False).extract(path, entry.id)
        else:
            from backend.cv_extraction.dxf_extractor import DXFHybridExtractor
            extraction = DXFHybridExtractor().extract(path, entry.id)
        t1 = time.time()
        plan = build_normalized_plan(extraction, plan_id=f"{entry.id}-{kind}")
        meta["extract_seconds"] = round(t1 - t0, 1)
        meta["normalize_seconds"] = round(time.time() - t1, 1)
        meta["warnings"] = list(extraction.warnings)[:50]
        meta["timed_out_internally"] = any(w.startswith("TIMEOUT") for w in extraction.warnings)
        icv = extraction.independent_cv
        meta["independent_cv"] = None if icv is None else {
            "status": getattr(icv, "status", None),
            "n_measurements": len(icv.measurements),
            "scale_points_per_metre": icv.scale_points_per_metre,
            "scale_confidence": icv.scale_confidence,
            "site_plan_page": icv.site_plan_page,
            "measurements": [
                {"field": m.field, "value": m.value, "value_m": m.value_m, "unit": m.unit,
                 "confidence": m.confidence, "source": (m.source.value if hasattr(m.source, "value") else m.source),
                 "evidence": [str(e)[:160] for e in (m.evidence or [])][:4], "note": (m.note or "")[:300]}
                for m in icv.measurements
            ],
            "warnings": list(icv.warnings)[:40],
        }
        (out_dir / f"{plan_id}.plan.json").write_text(plan.model_dump_json(indent=1), encoding="utf-8")
        meta["status"] = "ok"
    except Exception as exc:  # a crash is a result
        meta["status"] = f"crash:{type(exc).__name__}"
        meta["error"] = str(exc)[:500]
        meta["traceback"] = traceback.format_exc()[-3000:]
    meta["total_seconds"] = round(time.time() - t0, 1)
    meta["reconcile_calls"] = RECON_LOG
    (out_dir / f"{plan_id}.meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(f"{variant} {plan_id} {meta['status']} {meta['total_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
