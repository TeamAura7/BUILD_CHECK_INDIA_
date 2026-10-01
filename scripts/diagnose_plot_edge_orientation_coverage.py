"""
Diagnostic: what fraction of `plot_edge` nodes get a real orientation label
(front/rear/left/right, not None) under the CURRENT graph_builder.py, across
a real DXF corpus?

This answers a specific question, deliberately narrower than "did the model
train well": it says nothing about the GNN itself, only about how much real
silver-label signal `build_entity_graph` -> `resolve_front_side` is actually
able to produce today. Two very different situations both look like "the
front/rear/left/right classes trained badly", and this number is what tells
you which one you're in:

  * Still a small labeled fraction  -> the evidence-generation problem (no
    road layer, no FRONT/STREET/ACCESS/GATE text on most sheets) is still
    there. No amount of retraining fixes that; the fix has to happen
    upstream of the model, in what evidence resolve_front_side has to work
    with.

  * A healthy labeled fraction -> any past below-chance accuracy you
    measured was on a checkpoint trained before whatever fixed this, i.e.
    on stale/arbitrary labels. The honest next step is a clean retrain, not
    another diagnosis pass.

Per-file failures never abort the run (same convention as
backend/gnn_extraction/dataset.py::build_dataset) -- they're collected and
reported, not raised.

Usage:
    python scripts/diagnose_plot_edge_orientation_coverage.py /path/to/dxf_corpus
    python scripts/diagnose_plot_edge_orientation_coverage.py /path/to/dxf_corpus --pattern "*.dxf" --per-file
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.gnn_extraction.graph_builder import build_entity_graph  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus_dir", type=Path, help="Directory of DXF files (searched recursively).")
    parser.add_argument("--pattern", default="*.dxf", help="Glob pattern for files within corpus_dir (default: *.dxf).")
    parser.add_argument("--k-nearest", type=int, default=4, help="k_nearest passed through to build_entity_graph.")
    parser.add_argument("--per-file", action="store_true", help="Also print each file's own labeled fraction.")
    args = parser.parse_args()

    paths = sorted(args.corpus_dir.rglob(args.pattern))
    if not paths:
        print(f"No files matching {args.pattern!r} under {args.corpus_dir}", file=sys.stderr)
        sys.exit(1)

    total_edges = 0
    labeled_edges = 0
    role_counts: Counter[str] = Counter()  # front/rear/left/right counts among labeled edges
    evidence_counts: Counter[str] = Counter()  # which resolve_front_side tier resolved each labeled file
    files_with_any_label = 0
    files_with_zero_labels = 0
    skipped: list[tuple[str, str]] = []
    empty: list[str] = []
    per_file_rows: list[tuple[str, int, int, str]] = []  # (name, labeled, total, evidence_level)

    for path in paths:
        try:
            graph = build_entity_graph(path, path.stem, k_nearest=args.k_nearest)
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
            skipped.append((path.name, str(exc)))
            continue

        plot_edges = [n for n in graph.nodes if n.node_type == "plot_edge"]
        if not plot_edges:
            empty.append(path.name)
            continue

        file_labeled = sum(1 for n in plot_edges if n.label is not None)
        total_edges += len(plot_edges)
        labeled_edges += file_labeled
        for n in plot_edges:
            if n.label is not None:
                role_counts[n.label] += 1

        # Every plot_edge node in a file carries the SAME evidence_level
        # (resolve_front_side is one call per file, not per edge) -- read
        # it off the first edge that has one, defaulting to "none" for a
        # zero-label file (evidence_level is None there by construction).
        evidence_level = next(
            (n.raw_measurements.get("evidence_level") for n in plot_edges if n.raw_measurements.get("evidence_level")),
            "none",
        )
        if file_labeled > 0:
            evidence_counts[evidence_level] += 1
            files_with_any_label += 1
        else:
            files_with_zero_labels += 1
        per_file_rows.append((path.name, file_labeled, len(plot_edges), evidence_level))

    print(f"Corpus:            {args.corpus_dir} ({args.pattern})")
    print(f"Files found:       {len(paths)}")
    print(f"Parsed OK:         {len(per_file_rows)}")
    print(f"Skipped (error):   {len(skipped)}")
    print(f"Empty (no polys):  {len(empty)}")
    print()

    if total_edges == 0:
        print("No plot_edge nodes were produced by any file -- can't compute a labeled fraction.")
    else:
        frac = labeled_edges / total_edges
        print(f"plot_edge nodes:          {total_edges}")
        print(f"  with a real label:      {labeled_edges}  ({frac:.1%})")
        print(f"  left as None:           {total_edges - labeled_edges}  ({1 - frac:.1%})")
        print()
        print(f"Files with >=1 labeled plot_edge:  {files_with_any_label}")
        print(f"Files with ZERO labeled plot_edge: {files_with_zero_labels}")
        print()
        print("Label breakdown (front/rear/left/right should each be roughly 25% of labeled edges")
        print("if labeling isn't systematically biased toward one side):")
        for role, count in role_counts.most_common():
            print(f"  {role:<8} {count:>6}  ({count / labeled_edges:.1%} of labeled)")
        print()
        print("Which resolve_front_side evidence tier resolved each labeled file (see front_side.py's")
        print("priority order) -- lets you check whether a skew in the role breakdown above traces to")
        print("one particular tier (e.g. a left/right convention bug) rather than being spread evenly:")
        for level, count in evidence_counts.most_common():
            print(f"  {level:<20} {count:>5} file(s)  ({count / files_with_any_label:.1%} of labeled files)")

    if args.per_file:
        print()
        print("Per-file (labeled / total plot_edge nodes, resolve_front_side evidence tier):")
        for name, file_labeled, file_total, evidence_level in per_file_rows:
            marker = "" if file_labeled else "  <-- zero labels"
            print(f"  {name:<40} {file_labeled}/{file_total}  [{evidence_level}]{marker}")

    if skipped:
        print()
        print(f"Skipped {len(skipped)} file(s):")
        for name, reason in skipped[:20]:
            print(f"  {name}: {reason}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")

    if empty:
        print()
        print(f"{len(empty)} file(s) parsed but had no plot_edge nodes at all (no closed plot polygon found):")
        for name in empty[:20]:
            print(f"  {name}")
        if len(empty) > 20:
            print(f"  ... and {len(empty) - 20} more")


if __name__ == "__main__":
    main()
