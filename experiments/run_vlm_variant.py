"""
VLM evaluation driver for BUILDCheck India (paper evaluation).
Lives OUTSIDE backend/ -- no production code is modified.

It runs ONE plan through the production PDF path with the vision layer ON:
    PDFHybridExtractor(enable_vision=True) -> build_normalized_plan
and saves <PLAN>.plan.json + <PLAN>.meta.json + <PLAN>.vision.json.

Safeguards so that a result can never be mistaken for a VLM result by accident:
  * DECONTAMINATION: the shipped prompts contain example evidence strings
    (9.14, 8.22, 222.83) that equal truth values of development sheets
    (PLAN5 depth, PLAN7 road, PLAN2 plot area). They are replaced in memory by
    values that occur on NO corpus sheet (5.27, 4.87, 317.46) before any call.
  * VALIDITY: the production extractor swallows VLM failures as a warning
    ("vision extraction unavailable ..."). This driver records that and marks
    the run  vlm_valid=false, so it is excluded from the VLM table.

Usage:  python experiments/run_vlm_variant.py PLAN_ID OUT_DIR
Backend/model come from the normal settings (.env or environment):
  VISION_ENABLED=true  VISION_BACKEND=smolvlm|qwen|api  (api: VISION_API_KEY, VISION_API_MODEL, VISION_API_BASE_URL)
"""
from __future__ import annotations

import json, os, sys, time, traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DECONTAMINATE = {"9.14": "5.27", "8.22": "4.87", "222.83": "317.46"}
PROMPT_NAMES = ["ARCHITECTURAL_PLAN_PROMPT", "SITE_PLAN_FOCUS_PROMPT", "AREA_STATEMENT_FOCUS_PROMPT",
                "HEIGHT_FOCUS_PROMPT", "DXF_VISION_FALLBACK_PROMPT"]
CAPTURED: list = []


def decontaminate_prompts() -> dict:
    import importlib
    from backend.vision_extraction import prompts
    report = {}
    new = {}
    for name in PROMPT_NAMES:
        txt = getattr(prompts, name, None)
        if txt is None:
            continue
        for old, rep in DECONTAMINATE.items():
            report[f"{name}:{old}"] = txt.count(old)
            txt = txt.replace(old, rep)
        new[name] = txt
        setattr(prompts, name, txt)
    # base.py (and any other module) imported the constants by name -> patch every loaded copy
    importlib.import_module("backend.vision_extraction.base")
    for mod in list(sys.modules.values()):
        if mod and getattr(mod, "__name__", "").startswith("backend."):
            for name, txt in new.items():
                if hasattr(mod, name):
                    setattr(mod, name, txt)
    for name, txt in new.items():
        for old in DECONTAMINATE:
            assert old not in txt, f"decontamination failed: {old} still in {name}"
    return report


def capture_vision() -> None:
    import backend.vision_extraction as ve
    from backend.cv_extraction import pdf_extractor
    orig = ve.get_vision_extractor

    def wrapped():
        ex = orig()
        orig_analyze = ex.analyze_pdf

        def analyze(path, *a, **k):
            res = orig_analyze(path, *a, **k)
            CAPTURED.append(res)
            return res
        ex.analyze_pdf = analyze
        return ex
    ve.get_vision_extractor = wrapped
    pdf_extractor.get_vision_extractor = wrapped


def grounding_stats(res) -> dict:
    d = json.loads(res.model_dump_json()) if hasattr(res, "model_dump_json") else {}
    out = {"pages": 0, "dimensions": 0, "areas": 0, "grounded_true": 0, "grounded_false": 0, "grounded_none": 0}
    for pg in d.get("pages", []):
        out["pages"] += 1
        for coll in ("dimensions", "areas"):
            for it in pg.get(coll, []) or []:
                out[coll] += 1
                g = it.get("grounded")
                out["grounded_true" if g is True else "grounded_false" if g is False else "grounded_none"] += 1
    return out


def main() -> int:
    plan_id, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    from backend.config import get_settings
    s = get_settings()
    meta = {"variant": out_dir.name, "plan": plan_id, "kind": "pdf",
            "vision_backend": s.vision_backend,
            "vision_model": s.vision_api_model if s.vision_backend == "api" else s.vision_model_name,
            "vision_render_dpi": s.vision_render_dpi}
    meta["decontamination"] = decontaminate_prompts()
    capture_vision()
    t0 = time.time()
    try:
        from backend.corpus.store import load_manifest
        from backend.cv_extraction.pdf_extractor import PDFHybridExtractor
        from backend.spatial_reasoning.pipeline import build_normalized_plan
        entry = next(p for p in load_manifest().plans if p.id == plan_id)
        extraction = PDFHybridExtractor(enable_vision=True).extract(REPO / entry.pdf, entry.id)
        plan = build_normalized_plan(extraction, plan_id=f"{entry.id}-pdf")
        warns = list(extraction.warnings)
        meta["warnings"] = warns[:80]
        meta["vision_failed"] = any("vision extraction unavailable" in w for w in warns)
        meta["dropped_ungrounded"] = sum(1 for w in warns if "Dropped ungrounded" in w)
        meta["vision_page_failures"] = sum(1 for w in warns if "vision analysis failed" in w)
        if CAPTURED:
            meta["grounding"] = grounding_stats(CAPTURED[-1])
        pages_ok = (meta.get("grounding") or {}).get("pages", 0)
        # valid only if the VLM actually answered on every page it was given
        meta["vlm_valid"] = bool(CAPTURED) and not meta["vision_failed"] and pages_ok > 0 \
            and meta["vision_page_failures"] == 0
        if CAPTURED:
            (out_dir / f"{plan_id}.vision.json").write_text(CAPTURED[-1].model_dump_json(indent=1), encoding="utf-8")
        (out_dir / f"{plan_id}.plan.json").write_text(plan.model_dump_json(indent=1), encoding="utf-8")
        meta["status"] = "ok"
    except Exception as exc:
        meta["status"] = f"crash:{type(exc).__name__}"
        meta["error"] = str(exc)[:500]
        meta["traceback"] = traceback.format_exc()[-3000:]
        meta["vlm_valid"] = False
    meta["total_seconds"] = round(time.time() - t0, 1)
    (out_dir / f"{plan_id}.meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(f"{out_dir.name} {plan_id} {meta['status']} vlm_valid={meta.get('vlm_valid')} {meta['total_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
