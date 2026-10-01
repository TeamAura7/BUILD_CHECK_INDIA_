from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.cv_extraction.site_plan import extract_independent_cv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the independent CV/native-text site-plan extractor with Vision disabled."
    )
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = extract_independent_cv(args.pdf, args.pdf.stem)
    payload = result.model_dump(mode="json")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nCV extraction written to {args.output}")


if __name__ == "__main__":
    main()
