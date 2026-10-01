"""
Follow-up to diagnose_plot_edge_orientation_coverage.py: for files that come
back with ZERO labeled plot_edge nodes, which evidence tier is actually
responsible?

    - "road_layer_present"   at least one polygon sits on a ROAD/ROW/
                              RIGHT-OF-WAY/STREET-named layer, but
                              resolve_front_side still found no usable
                              road_bbox for this file (its road_polygons
                              list ended up empty at the point graph_builder
                              calls resolve_front_side -- e.g. the layer
                              exists but no CLOSED polygon was on it, only
                              open lines).
    - "access_text_present"  at least one text entity matches FRONT/STREET/
                              ACCESS/MAIN-ENTRY/GATE wording, but it didn't
                              end up resolving front-side either (e.g. it's
                              far from every plot-boundary bbox side, or on
                              a different part of a still-noisy region).
    - "no_evidence_at_all"   neither: the sheet genuinely has nothing for
                              resolve_front_side to work with. No amount of
                              regex/layer-matching tuning fixes this one --
                              it needs a different evidence source
                              (e.g. a north-arrow block, sheet position
                              convention) or human labeling.

This tells you which of three very different fixes to prioritize:
tightening the road-layer regex, tightening the access-text regex, or
accepting that a chunk of the corpus has no recoverable orientation signal
at all and needs another approach.

Usage:
    python scripts/diagnose_orientation_evidence_gaps.py /path/to/dxf_corpus
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.cv_extraction.dxf_extractor import _ROAD_LAYER_RE  # noqa: E402
from backend.gnn_extraction.graph_builder import build_entity_graph  # noqa: E402
from backend.spatial_reasoning.road_access import _ACCESS_PATTERNS  # noqa: E402


def _classify(graph) -> str:
    has_road_layer = any(
        n.node_type in ("road_polygon", "plot_polygon", "building_candidate", "other_polygon")
        and _ROAD_LAYER_RE.search(n.layer or "")
        for n in graph.nodes
    )
    has_access_text = any(
        n.node_type == "text"
        and n.raw_text
        and any(p.search(n.raw_text) for p in _ACCESS_PATTERNS.values())
        for n in graph.nodes
    )
    if has_road_layer and has_access_text:
        return "road_layer_and_access_text_present"
    if has_road_layer:
        return "road_layer_present"
    if has_access_text:
        return "access_text_present"
    return "no_evidence_at_all"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus_dir", type=Path)
    parser.add_argument("--pattern", default="*.dxf")
    parser.add_argument("--k-nearest", type=int, default=4)
    parser.add_argument("--list-examples", type=int, default=5, help="How many example filenames to print per bucket.")
    args = parser.parse_args()

    paths = sorted(args.corpus_dir.rglob(args.pattern))
    if not paths:
        print(f"No files matching {args.pattern!r} under {args.corpus_dir}", file=sys.stderr)
        sys.exit(1)

    buckets: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    skipped = 0
    zero_label_total = 0

    for path in paths:
        try:
            graph = build_entity_graph(path, path.stem, k_nearest=args.k_nearest)
        except Exception:
            skipped += 1
            continue

        plot_edges = [n for n in graph.nodes if n.node_type == "plot_edge"]
        if not plot_edges:
            continue
        if any(n.label is not None for n in plot_edges):
            continue  # only diagnosing the ZERO-label files

        zero_label_total += 1
        bucket = _classify(graph)
        buckets[bucket] += 1
        examples.setdefault(bucket, [])
        if len(examples[bucket]) < args.list_examples:
            examples[bucket].append(path.name)

    print(f"Corpus:                     {args.corpus_dir} ({args.pattern})")
    print(f"Files with ZERO labels:     {zero_label_total}")
    print(f"Skipped (parse error):      {skipped}")
    print()
    for bucket, count in buckets.most_common():
        pct = count / zero_label_total if zero_label_total else 0.0
        print(f"  {bucket:<38} {count:>5}  ({pct:.1%})")
        for name in examples.get(bucket, []):
            print(f"      e.g. {name}")
    print()
    print("Reading this:")
    print("  - road_layer_present / access_text_present -> evidence IS on the sheet but the")
    print("    matcher isn't using it. Pull a few example files above into a DXF viewer and check")
    print("    the actual layer name / text wording against _ROAD_LAYER_RE / _ACCESS_PATTERNS.")
    print("  - no_evidence_at_all -> nothing to extract here; needs a different evidence source")
    print("    or human labeling, not a regex fix.")


if __name__ == "__main__":
    main()
