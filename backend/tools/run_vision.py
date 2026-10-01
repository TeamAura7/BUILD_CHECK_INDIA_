from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from backend.config import get_settings
from backend.vision_extraction import get_vision_extractor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the configured local vision-language backend on an architectural PDF.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--backend",
        choices=["smolvlm", "qwen", "api"],
        default=None,
        help="Override VISION_BACKEND for this run only (default: use configured/env setting, "
        "falls back to 'smolvlm'). Use 'api' on machines without enough RAM/VRAM for a local "
        "VLM -- it calls a hosted OpenAI-compatible vision endpoint (Groq by default) instead "
        "of loading any model weights.",
    )
    parser.add_argument(
        "--no-grounding",
        action="store_true",
        help="Return raw Vision evidence without native-PDF text grounding. "
             "Use for independent CV-vs-Vision validation; production mode should keep grounding enabled.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override VISION_MAX_NEW_TOKENS for this run only. Lower this (e.g. 300) for a "
        "fast CPU smoke test that just checks the pipeline runs end-to-end.",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.backend:
        settings.vision_backend = args.backend
    if args.max_new_tokens:
        settings.vision_max_new_tokens = args.max_new_tokens

    extractor = get_vision_extractor()
    print(f"Using vision backend: {type(extractor).__name__} ({extractor.model_name})", flush=True)
    print(f"max_new_tokens={settings.vision_max_new_tokens}, render_dpi={settings.vision_render_dpi}", flush=True)

    try:
        result = extractor.analyze_pdf(
            args.pdf,
            ground_against_native_text=not args.no_grounding,
        )
    except Exception:
        # Never fail silently: on CPU, model load / generate is the step
        # most likely to blow up (OOM, missing optional deps, etc.) --
        # always show the real traceback instead of leaving the user
        # staring at a returned shell prompt with no explanation.
        print("\nVision extraction crashed with an unhandled exception:\n", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)

    payload = result.model_dump(mode="json")
    # Phase 3.3+ contract: Vision CLI also runs the independent CV/native/OCR
    # path and emits a final agreement object. A disagreement remains a CONFLICT;
    # no source is silently selected as the winner.
    try:
        from backend.cv_extraction.site_plan import extract_independent_cv
        from backend.spatial_reasoning.final_fusion import build_final_agreement
        cv_result = extract_independent_cv(args.pdf, args.pdf.stem)
        payload["independent_cv"] = cv_result.model_dump(mode="json")
        payload["final_agreed_values"] = build_final_agreement(cv_result, result)
    except Exception as exc:
        payload["final_agreed_values"] = {
            "document_id": args.pdf.stem,
            "values": {},
            "summary": {},
            "has_conflicts": False,
            "error": f"CV/Vision agreement layer failed: {exc}",
        }
    text = json.dumps(payload, indent=2, ensure_ascii=False)

    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"\nVision extraction written to {args.output}", flush=True)
    else:
        print(text)

    if result.warnings:
        print(f"\n{len(result.warnings)} warning(s):", flush=True)
        for w in result.warnings:
            print(f"  - {w}", flush=True)


if __name__ == "__main__":
    main()
