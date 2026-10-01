"""
Offline VLM replay (no API calls). Re-runs the unmodified PDF pipeline with
vision ON, but the VLM backend is replaced by the SAVED VisionDocumentResult
from the 28 Sep 2026 Groq run (run1). Fusion variants are applied by
monkeypatching, outside backend/, to attribute errors to pipeline design vs
model output.

Usage: python experiments/replay_vlm.py VARIANT PLAN_ID OUT_DIR
Variants:
  replay          no change (must reproduce run1)
  tol5            final_fusion CV-VLM tolerance = scoring tolerance max(0.15, 5%)
  grounded_only   an ungrounded VLM value cannot create a conflict (CV kept)
  cv_priority     on CV-VLM disagreement keep the CV value at its own level (VLM only
                  confirms or fills gaps) -- 'VLM as tie-breaker'
  vlm_only_low    VLM-only values capped at LOW (in addition to cv_priority)
  no_region_floor also drop the vision floor-plan region-count heuristic (in addition to vlm_only_low)
"""
from __future__ import annotations
import json, sys, time, traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SAVED = REPO / "experiments/results/fresh/variants/pdf_hybrid_vlm_api"


def install(variant: str, plan_id: str):
    from backend.schemas.vision import VisionDocumentResult
    import backend.vision_extraction as ve
    from backend.cv_extraction import pdf_extractor
    saved = VisionDocumentResult.model_validate_json((SAVED / f"{plan_id}.vision.json").read_text())

    class Replay:
        def analyze_pdf(self, *a, **k):
            return saved.model_copy(deep=True)
    ve.get_vision_extractor = lambda: Replay()
    pdf_extractor.get_vision_extractor = lambda: Replay()

    from backend.spatial_reasoning import final_fusion as ff
    orig_tol, orig_vf = ff._tol, ff._value_field

    if variant == "tol5":
        def tol(a, b, unit="m"):
            if unit in ("%", "ratio"):
                return orig_tol(a, b, unit)
            return max(0.15, 0.05 * max(abs(a), abs(b)))
        ff._tol = tol

    if variant in ("grounded_only", "cv_priority", "vlm_only_low", "no_region_floor"):
        def vf(cv_m, vision, field, unit="m"):
            grounded = vision[-1] if vision is not None and len(vision) > 4 and isinstance(vision[-1], bool) else None
            cvv = None if cv_m is None else (cv_m.value_m if cv_m.value_m is not None else cv_m.value)
            if vision is not None and cvv is not None:
                vv = vision[0]
                disagree = abs(cvv - vv) > ff._tol(cvv, vv, unit)
                if disagree and (variant != "grounded_only" or grounded is False):
                    out = orig_vf(cv_m, None, field, unit)      # CV value at its own level
                    out.confidence.reason += f" [replay:{variant}] VLM value {vv:.4f} disagreed and was not used."
                    return out
            out = orig_vf(cv_m, vision, field, unit)
            if variant in ("vlm_only_low", "no_region_floor") and out.source == "FINAL:VISION_ONLY":
                out.confidence.level = ff.cap_confidence_level(out.confidence.level, ff.ConfidenceLevel.LOW)
            return out
        ff._value_field = vf

    if variant == "no_region_floor":
        from backend.spatial_reasoning import pipeline
        pipeline.floor_count_from_vision = lambda *a, **k: None


def main() -> int:
    variant, plan_id, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    install(variant, plan_id)
    meta = {"variant": out_dir.name, "replay_variant": variant, "plan": plan_id, "kind": "pdf"}
    t0 = time.time()
    try:
        from backend.corpus.store import load_manifest
        from backend.cv_extraction.pdf_extractor import PDFHybridExtractor
        from backend.spatial_reasoning.pipeline import build_normalized_plan
        entry = next(p for p in load_manifest().plans if p.id == plan_id)
        ex = PDFHybridExtractor(enable_vision=True).extract(REPO / entry.pdf, entry.id)
        plan = build_normalized_plan(ex, plan_id=f"{entry.id}-pdf")
        meta["warnings"] = list(ex.warnings)[:60]
        meta["timed_out_internally"] = any(w.startswith("TIMEOUT") or "did not complete" in w for w in ex.warnings)
        (out_dir / f"{plan_id}.plan.json").write_text(plan.model_dump_json(indent=1))
        meta["status"] = "ok"
    except Exception as exc:
        meta["status"] = f"crash:{type(exc).__name__}"; meta["error"] = str(exc)[:500]
        meta["traceback"] = traceback.format_exc()[-3000:]
    meta["total_seconds"] = round(time.time() - t0, 1)
    (out_dir / f"{plan_id}.meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(variant, plan_id, meta["status"], meta["total_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
