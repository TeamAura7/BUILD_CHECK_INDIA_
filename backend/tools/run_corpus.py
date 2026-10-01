"""
Corpus command line.

  python -m backend.tools.run_corpus validate
  python -m backend.tools.run_corpus list
  python -m backend.tools.run_corpus add --id HO-001 --split heldout --pdf a.pdf [--dxf a.dxf] --source "..." --licence "..."
  python -m backend.tools.run_corpus freeze
  python -m backend.tools.run_corpus eval [--split dev|heldout] [--modalities pdf dxf reconciled] [--no-cache]
                                          [--include-unverified] [--json out.json]
  python -m backend.tools.run_corpus corrections list [--all]
  python -m backend.tools.run_corpus corrections promote CORRECTION_ID [--annotator NAME]

`add` never runs an extractor. `eval --split heldout` refuses unless the
held-out set is frozen and unchanged. `corrections promote` always lands in
"dev" -- a correction can never become held-out truth (see
backend/corpus/corrections.py's module docstring for why).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.corpus import store
from backend.corpus.runner import MODALITIES, format_report, run_eval
from backend.corpus.schema import SCORABLE_DEFAULT, Provenance, VERIFICATION_ORDER


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    sub.add_parser("list")
    add = sub.add_parser("add")
    add.add_argument("--id", required=True)
    add.add_argument("--split", choices=("dev", "heldout"), required=True)
    add.add_argument("--pdf", type=Path)
    add.add_argument("--dxf", type=Path)
    add.add_argument("--source", default="")
    add.add_argument("--licence", default="")
    add.add_argument("--redacted", action="store_true")
    add.add_argument("--tags", nargs="*", default=[])
    sub.add_parser("freeze")
    ev = sub.add_parser("eval")
    ev.add_argument("--split", choices=("dev", "heldout"), default="dev")
    ev.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    ev.add_argument("--no-cache", action="store_true")
    ev.add_argument("--include-unverified", action="store_true")
    ev.add_argument("--json", type=Path)
    corr = sub.add_parser("corrections")
    corr_sub = corr.add_subparsers(dest="corrections_cmd", required=True)
    corr_list = corr_sub.add_parser("list")
    corr_list.add_argument("--all", action="store_true", help="include already-promoted corrections")
    corr_promote = corr_sub.add_parser("promote")
    corr_promote.add_argument("correction_id")
    corr_promote.add_argument("--annotator", default="")
    args = parser.parse_args(argv)

    try:
        if args.cmd == "validate":
            problems = store.validate_corpus()
            print("corpus valid" if not problems else "\n".join(problems))
            return 0 if not problems else 1
        if args.cmd == "list":
            for p in store.load_manifest().plans:
                print(f"{p.split:8s} {p.id:8s} pdf={'y' if p.pdf else '-'} dxf={'y' if p.dxf else '-'} tags={','.join(p.tags)}")
            return 0
        if args.cmd == "add":
            entry = store.add_plan(
                args.id, args.split, pdf=args.pdf, dxf=args.dxf, tags=args.tags,
                provenance=Provenance(source=args.source, licence=args.licence, redacted=args.redacted),
            )
            print(f"registered {entry.id} ({entry.split}); fill in {entry.truth} by hand, then set annotation.human_verified")
            return 0
        if args.cmd == "freeze":
            lock = store.freeze_heldout()
            print(f"froze {len(lock['plans'])} held-out plan(s) at {lock['frozen_at']}")
            return 0
        if args.cmd == "corrections":
            from backend.corpus import corrections as corr_mod

            if args.corrections_cmd == "list":
                events = corr_mod.load_corrections() if args.all else corr_mod.pending_corrections()
                promoted = corr_mod.load_promotions() if args.all else {}
                if not events:
                    print("no corrections recorded" if args.all else "no pending (un-promoted) corrections")
                    return 0
                for e in events:
                    tag = f" -> {promoted[e.correction_id]}" if e.correction_id in promoted else ""
                    print(f"{e.correction_id[:8]}  {e.document_id:24s} {e.field:24s} "
                          f"{e.predicted_value!r} -> {e.corrected_value!r} {e.unit}  "
                          f"by {e.corrected_by or '(unknown)'} at {e.recorded_at}{tag}")
                return 0
            if args.corrections_cmd == "promote":
                matches = [e for e in corr_mod.load_corrections() if e.correction_id.startswith(args.correction_id)]
                if not matches:
                    print(f"refused: no correction id starts with {args.correction_id!r}", file=sys.stderr)
                    return 2
                if len(matches) > 1:
                    print(f"refused: {args.correction_id!r} matches {len(matches)} corrections; give more characters", file=sys.stderr)
                    return 2
                entry = corr_mod.promote_correction(matches[0].correction_id, annotator=args.annotator)
                print(f"promoted {matches[0].correction_id[:8]} into corpus plan {entry.id!r} (dev); "
                      f"other fields still need independent annotation before this plan is fully scorable")
                return 0
        verifications = VERIFICATION_ORDER if args.include_unverified else SCORABLE_DEFAULT
        result = run_eval(args.split, modalities=tuple(args.modalities), verifications=verifications,
                          use_cache=not args.no_cache)
        print(format_report(result))
        if args.json:
            args.json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        return 0
    except store.CorpusError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
