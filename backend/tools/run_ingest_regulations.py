"""
Ingest a municipality's regulation documents (data/regulations/<MUNICIPALITY>/
*.pdf and *.txt) into that municipality's FAISS + BM25-backing RAG index.

Usage:
    python -m backend.tools.run_ingest_regulations BBMP
    python -m backend.tools.run_ingest_regulations BBMP --force
"""

from __future__ import annotations

import argparse
import sys

from backend.rag.ingestion.pipeline import ingest_municipality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("municipality", help="e.g. BBMP")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-parse and rebuild the whole index even for already-indexed files.",
    )
    args = parser.parse_args()

    stats = ingest_municipality(args.municipality, force=args.force)
    print(
        f"Ingested {args.municipality.upper()}: "
        f"{stats['pages']} pages -> {stats['chunks']} new chunks -> "
        f"{stats['vectors']} total vectors in index."
    )
    if stats["vectors"] == 0:
        print(
            f"No vectors indexed. Add PDFs/.txt files under "
            f"data/regulations/{args.municipality.upper()}/ and re-run.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
