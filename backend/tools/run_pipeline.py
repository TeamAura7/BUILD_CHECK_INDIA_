"""
Run the FULL pipeline on an architectural PDF: native/OpenCV geometry
extraction, optional vision-language semantic extraction, and spatial
reasoning resolution -- producing the final NormalizedPlan.

This is deliberately different from `run_vision.py`, which only runs the
vision stage in isolation and returns raw VLM output. Raw vision output
alone is checked against the PDF's raw TEXT layer (does this value appear
as text anywhere on the page) -- a real but weaker check than what this
full pipeline does: `spatial_reasoning.vision_semantics` cross-references
every VLM-claimed value against the CV/native pipeline's own measured
DIMENSION LINES (`backend.cv_extraction`, real page-space geometry from
OpenCV contour/line detection and the PDF's native vector paths), not
just floating text. A VLM value that survives text-grounding but was
never actually confirmed by a measured line in the drawing is still
excluded from the NormalizedPlan by this stage (see
`vision_semantics.preferred_vision_value`/`preferred_vision_area`).

Usage:
    python -m backend.tools.run_pipeline plan.pdf --output result.json
    python -m backend.tools.run_pipeline plan.pdf --backend api --vision
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from backend.config import get_settings
from backend.cv_extraction.registry import default_registry
from backend.spatial_reasoning.pipeline import build_normalized_plan


def _summarize(plan) -> str:
    """Short human-readable summary of the resolved fields, for quick eyeballing."""

    def fmt(vf) -> str:
        if vf is None:
            return "n/a"
        if vf.value is None:
            return f"MISSING ({vf.confidence})"
        return f"{vf.value} ({vf.confidence})"

    lines = [
        f"plot:     width={fmt(plan.plot.width)}  depth={fmt(plan.plot.depth)}  area={fmt(plan.plot.area)}",
        f"building: width={fmt(plan.building.width)}  depth={fmt(plan.building.depth)}  "
        f"footprint_area={fmt(plan.building.footprint_area)}",
        f"road:     width={fmt(plan.road.width)}",
        f"setbacks: front={fmt(plan.setbacks.front)}  rear={fmt(plan.setbacks.rear)}  "
        f"left={fmt(plan.setbacks.left)}  right={fmt(plan.setbacks.right)}",
        f"coverage: {fmt(plan.coverage)}   far: {fmt(plan.far)}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full extraction + spatial-reasoning pipeline (CV geometry + "
        "vision semantics, fused) on an architectural PDF and print the resolved NormalizedPlan."
    )
    parser.add_argument("pdf", type=Path, help="Plan file to extract -- .pdf or .dxf.")
    parser.add_argument("--output", type=Path, default=None, help="Write the full NormalizedPlan JSON here.")
    parser.add_argument(
        "--extraction-output", type=Path, default=None,
        help="Also write the intermediate ExtractionResult JSON here (raw CV dimensions + raw "
        "vision_pages, before fusion/resolution) -- useful for seeing what each stage "
        "individually found before comparing to the final fused/resolved plan.",
    )
    parser.add_argument(
        "--vision", action="store_true",
        help="Force VISION_ENABLED=true for this run (equivalent to setting it in .env).",
    )
    parser.add_argument(
        "--backend", choices=["smolvlm", "qwen", "api"], default=None,
        help="Override VISION_BACKEND for this run only.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override VISION_MAX_NEW_TOKENS.")
    parser.add_argument(
        "--document-verified-fusion", action="store_true",
        help="Also run the independent CV path, independent Vision path, and the PHASEE3NEW "
        "document-evidence validation/fusion engine (backend.spatial_reasoning.final_fusion"
        ".build_document_verified_fusion), and write it alongside --output "
        "(as <output>.fusion.json, or print it if no --output was given). This is fully "
        "independent of the legacy full-page pipeline above -- it does not feed CV into "
        "Vision or vice versa.",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.vision:
        settings.vision_enabled = True
    if args.backend:
        settings.vision_backend = args.backend
    if args.max_new_tokens:
        settings.vision_max_new_tokens = args.max_new_tokens

    is_dxf = args.pdf.suffix.lower() == ".dxf"
    if is_dxf and settings.vision_enabled:
        print(
            "Note: --vision/--backend have no effect on a .dxf plan -- DXF vector geometry is "
            "exact and extracted deterministically, with no rasterization/OCR/Vision call.",
            flush=True,
        )
        settings.vision_enabled = False

    print(f"vision_enabled={settings.vision_enabled}", flush=True)
    if settings.vision_enabled:
        from backend.vision_extraction import get_vision_extractor

        extractor = get_vision_extractor()
        print(f"vision backend: {type(extractor).__name__} ({extractor.model_name})", flush=True)

    document_id = args.pdf.stem
    try:
        kind = "DXF vector geometry" if is_dxf else f"native/OpenCV geometry{' + vision' if settings.vision_enabled else ''}"
        print(f"\nExtracting ({kind})...", flush=True)
        extraction = default_registry().get_for_path(args.pdf).extract(args.pdf, document_id)
    except Exception:
        print("\nExtraction crashed with an unhandled exception:\n", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)

    print(
        f"Extracted: {len(extraction.dimensions)} native/CV dimension(s), "
        f"{len(extraction.plot_candidates)} plot candidate(s), "
        f"{len(extraction.building_candidates)} building candidate(s), "
        f"{len(extraction.vision_pages)} vision page result(s).",
        flush=True,
    )
    if extraction.warnings:
        print(f"{len(extraction.warnings)} extraction warning(s):", flush=True)
        for w in extraction.warnings:
            print(f"  - {w}", flush=True)

    if args.extraction_output:
        args.extraction_output.write_text(
            json.dumps(extraction.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nExtractionResult written to {args.extraction_output}", flush=True)

    print("\nResolving (spatial reasoning: CV geometry + vision semantics fusion)...", flush=True)
    try:
        plan = build_normalized_plan(extraction, plan_id=f"plan-{document_id}")
    except Exception:
        print("\nResolution crashed with an unhandled exception:\n", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)

    print("\n" + _summarize(plan) + "\n", flush=True)

    payload = plan.model_dump(mode="json")
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"NormalizedPlan written to {args.output}", flush=True)
    else:
        print(text)

    if args.document_verified_fusion and is_dxf:
        print(
            "\n--document-verified-fusion is a PDF-only path (it re-parses native PDF text/vector "
            "layers independently of the main extractor above); skipping it for this .dxf plan. "
            "The NormalizedPlan above is already the DXF result, straight from the same "
            "deterministic geometry.",
            flush=True,
        )
    elif args.document_verified_fusion:
        from backend.cv_extraction.site_plan import extract_independent_cv
        from backend.spatial_reasoning.final_fusion import build_document_verified_fusion
        from backend.vision_extraction import get_vision_extractor

        print("\nRunning independent CV + independent Vision + document-evidence fusion "
              "(PHASEE3NEW)...", flush=True)
        cv_result = extract_independent_cv(args.pdf, document_id)
        vision_result = None
        if settings.vision_enabled:
            vision_result = get_vision_extractor().analyze_pdf(args.pdf, ground_against_native_text=False)
        fusion = build_document_verified_fusion(cv_result, vision_result, pdf_path=str(args.pdf))
        fusion_text = json.dumps(fusion, indent=2, ensure_ascii=False)
        if args.output:
            fusion_path = args.output.with_suffix(args.output.suffix + ".fusion.json")
            fusion_path.write_text(fusion_text, encoding="utf-8")
            print(f"Document-verified fusion written to {fusion_path}", flush=True)
        else:
            print(fusion_text)


if __name__ == "__main__":
    main()
