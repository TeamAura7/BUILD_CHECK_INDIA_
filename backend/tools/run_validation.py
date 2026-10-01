from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.config import get_settings
from backend.validation import validate_extractions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run independent CV and Vision extraction, then compare them field-by-field. Legacy CV is excluded."
    )
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--vision-backend",
        choices=["smolvlm", "qwen", "api"],
        default=None,
        help="Override VISION_BACKEND for this validation run.",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.vision_backend:
        settings.vision_backend = args.vision_backend

    report = validate_extractions(args.pdf)
    payload = report.model_dump(mode="json")

    print(f"\nDocument: {report.document_id}")
    print("\nIndependent CV vs Independent Vision")
    print("-" * 78)
    for field in report.fields:
        if field.cv_value_m is not None or field.vision_value_m is not None:
            cv = "MISSING" if field.cv_value_m is None else f"{field.cv_value_m:.3f} m"
            vision = "MISSING" if field.vision_value_m is None else f"{field.vision_value_m:.3f} m"
            diff = "-" if field.absolute_difference_m is None else f"{field.absolute_difference_m:.3f} m"
        else:
            unit = field.unit or "value"
            cv = "MISSING" if field.cv_value is None else f"{field.cv_value:.3f} {unit}"
            vision = "MISSING" if field.vision_value is None else f"{field.vision_value:.3f} {unit}"
            diff = "-" if field.absolute_difference is None else f"{field.absolute_difference:.3f} {unit}"
        print(f"{field.field:28s} CV={cv:>14s}  Vision={vision:>14s}  diff={diff:>12s}  {field.status}")

    print("\nSummary:", json.dumps(report.summary, sort_keys=True))
    print("\nFINAL AGREED VALUES:")
    print(json.dumps(report.final_agreed_values, indent=2, ensure_ascii=False))

    if report.document_verified_fusion:
        print("\nDOCUMENT-EVIDENCE-VERIFIED FUSION (PHASEE3NEW):")
        print("Summary:", json.dumps(report.document_verified_fusion.get("summary", {}), sort_keys=True))
        if report.document_verified_fusion.get("has_unresolved_conflicts"):
            print("WARNING: one or more fields are UNRESOLVED_CONFLICT (final_value=null).")
        print(json.dumps(report.document_verified_fusion.get("final_agreed_values", {}), indent=2, ensure_ascii=False))

    if args.output:
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nValidation report written to {args.output}")


if __name__ == "__main__":
    main()
