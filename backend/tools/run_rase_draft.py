"""
Draft a RuntimeRuleDefinition from a municipality's ingested regulation
corpus (RAG-retrieved clauses -> Groq -> structured draft), or promote a
previously drafted rule into the live ruleset the compliance engine reads.

Usage:
    # Draft (requires GROQ_API_KEY and an ingested index for the municipality)
    python -m backend.tools.run_rase_draft draft BBMP "minimum front setback for residential plots"

    # Inspect drafts
    python -m backend.tools.run_rase_draft list BBMP

    # Promote a reviewed draft into data/runtime_rules/BBMP/rules.json
    python -m backend.tools.run_rase_draft promote BBMP bbmp-front-setback-residential
"""

from __future__ import annotations

import argparse
import json
import sys

from backend.rase.extractor import draft_rule, load_drafts, promote_draft, save_draft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_draft = sub.add_parser("draft", help="Draft a new rule from a natural-language query")
    p_draft.add_argument("municipality")
    p_draft.add_argument("query")

    p_draft_all = sub.add_parser(
        "draft-all",
        help="One-time bootstrap: draft a rule for every default compliance field "
        "(~13 Groq calls, run ONCE per municipality after ingestion — not per plan)",
    )
    p_draft_all.add_argument("municipality")

    p_list = sub.add_parser("list", help="List drafted (unpromoted) rules")
    p_list.add_argument("municipality")

    p_promote = sub.add_parser("promote", help="Promote a draft into the live ruleset")
    p_promote.add_argument("municipality")
    p_promote.add_argument("rule_id")
    p_promote.add_argument("--notes", default=None)

    args = parser.parse_args()

    if args.command == "draft":
        draft = draft_rule(args.query, args.municipality)
        if draft is None:
            print(
                "Drafting was refused: either too few regulation chunks were retrieved, "
                "or no concrete numeric threshold was found in the retrieved text.",
                file=sys.stderr,
            )
            sys.exit(1)
        save_draft(draft)
        print(json.dumps(json.loads(draft.model_dump_json()), indent=2))
        print(
            f"\nDraft saved. Review the citation(s) above, then promote with:\n"
            f"  python -m backend.tools.run_rase_draft promote {args.municipality.upper()} "
            f"{draft.rule.rule_id}"
        )

    elif args.command == "draft-all":
        from backend.tools.extractor import DEFAULT_FIELD_QUERIES, draft_all_fields

        results = draft_all_fields(args.municipality)
        drafted = 0
        for field, draft in results.items():
            if draft is None:
                print(f"- {field}: no rule drafted (refused, or too few chunks retrieved)")
            else:
                drafted += 1
                print(f"- {field}: drafted {draft.rule.rule_id}  [citation={draft.rule.citation}]")
        print(
            f"\n{drafted}/{len(DEFAULT_FIELD_QUERIES)} fields drafted. Review each with "
            f"'list', then promote the ones you accept:\n"
            f"  python -m backend.tools.run_rase_draft promote {args.municipality.upper()} <rule_id>\n"
            f"This is a ONE-TIME setup step — per-plan compliance checks after promotion "
            f"(backend.tools.run_compliance) make zero further API calls."
        )

    elif args.command == "list":
        drafts = load_drafts(args.municipality)
        if not drafts:
            print(f"No drafts for {args.municipality.upper()}.")
            return
        for d in drafts:
            print(f"- {d.rule.rule_id}: {d.rule.description}  [citation={d.rule.citation}]")

    elif args.command == "promote":
        promoted = promote_draft(args.rule_id, args.municipality, reviewer_notes=args.notes)
        print(f"Promoted: {promoted.rule_id} -> {promoted.description} (v{promoted.version})")


if __name__ == "__main__":
    main()
