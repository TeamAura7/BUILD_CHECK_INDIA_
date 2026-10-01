"""
Split a directory of real (ODA-converted) DXF files into "site plan" vs
"floor/room plan" buckets, so the GNN in backend/gnn_extraction only trains
on files where plot/setback/road labels are meaningful.

Why this exists: graph_builder.py's silver-labeling heuristic always picks
the largest closed polygon in a file and calls it "plot" -- correct on a
real site plan, but on a plain room/floor-plan DXF (no plot boundary at
all) it just picks the biggest room and fabricates a fake "plot" plus
fake front/rear/left/right setback edges around it. Training on a corpus
that mixes both silently injects wrong labels into exactly the classes
(plot, front, rear, left, right) that were already scoring weakest.

This is a transparent, inspectable SCORING heuristic, not another trained
model -- the whole point is to stop compounding "heuristic labels a
heuristic" on top of itself. Every verdict comes with the reasons that
produced it, and everything lands in a CSV you can spot-check by eye
before trusting the split.

NOTE: on a real corpus, most files turn out to be COMBINED sheets (site
plan + floor plan + elevation all in one DXF) -- see
scripts/split_dxf_into_site_regions.py, which reuses `score_entities`
below but applies it per spatially-clustered region of a file instead of
to the whole file at once. Use that script instead of this one once you've
confirmed (as we did) that your corpus is dominated by combined sheets;
this whole-file version is still useful for a first-pass sanity check and
for corpora that really are one-drawing-per-file.

Usage:
    python -m scripts.classify_site_vs_floor_plans /path/to/dxf_dir \
        --out-manifest site_plan_classification.csv \
        --split-into /path/to/split_output

    # after eyeballing the CSV and fixing any obvious mistakes there:
    python -m backend.gnn_extraction.train /path/to/split_output/site_plans --epochs 100
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from backend.cv_extraction.dxf_extractor import (
    DXFHybridExtractor,
    _BUILDING_LAYER_RE,
    _PLOT_LAYER_RE,
    _ROAD_LAYER_RE,
    _RawPoly,
    _RawText,
)

# --- text signals ------------------------------------------------------
# Indian building-plan title blocks are fairly consistent about naming the
# drawing type explicitly. These are deliberately generous (case-insensitive,
# word-boundary-loose) since a false SITE match just means "worth a second
# look", not an auto-accept -- the geometric signals below carry the real
# weight of the decision.
SITE_TEXT_RE = re.compile(
    r"SITE\s*PLAN|LOCATION\s*PLAN|KEY\s*PLAN|SETBACK|PLOT\s*(AREA|BOUNDARY|DIMENSION)|"
    r"NORTH\s*ARROW|ADJOINING|ROAD\s*WIDTH|SITE\s*AREA|PROPERTY\s*LINE",
    re.I,
)
FLOOR_TEXT_RE = re.compile(
    r"FLOOR\s*PLAN|GROUND\s*FLOOR|FIRST\s*FLOOR|SECOND\s*FLOOR|GA\s*PLAN|FURNITURE\s*LAYOUT|"
    r"BED\s*ROOM|BEDROOM|LIVING\s*ROOM|KITCHEN|TOILET|W\.?C\.?|DINING|"
    r"ELEVATION|SECTION\s*[A-Z]-[A-Z]|ROOF\s*PLAN",
    re.I,
)


@dataclass
class Verdict:
    filename: str
    label: str  # "site_plan" | "floor_plan" | "uncertain"
    score: float  # positive = site-plan-like, negative = floor-plan-like
    reasons: list[str] = field(default_factory=list)
    error: str = ""


def label_for_score(score: float) -> str:
    if score >= 2.0:
        return "site_plan"
    elif score <= -2.0:
        return "floor_plan"
    return "uncertain"


def score_entities(raw_polygons: list[_RawPoly], raw_texts: list[_RawText]) -> tuple[float, list[str]]:
    """
    Core scoring heuristic, factored out so it can be applied to a WHOLE
    file (classify_one, below) or to a single spatially-clustered REGION of
    a file (scripts/split_dxf_into_site_regions.py) with no duplicated
    logic. Positive = site-plan-like, negative = floor-plan-like.
    """
    score = 0.0
    reasons: list[str] = []

    # --- signal 1: title-block / annotation text keywords ---
    # On real corpora where files are combined "plan + elevation" sheets
    # (plot + full room-labeled floor plan + elevation all in one DXF --
    # the standard convention for Indian sanctioned-plan submissions),
    # floor/room/elevation keywords are near-universal and do NOT
    # distinguish "has a real plot boundary" from "doesn't" -- they just
    # mean "this is a normal combined sheet", true almost everywhere.
    # Weighted down hard. The site-plan keyword side stays rarer/more
    # specific, so it keeps real weight.
    all_text = " ".join(t.text for t in raw_texts).upper()
    site_hits = len(SITE_TEXT_RE.findall(all_text))
    floor_hits = len(FLOOR_TEXT_RE.findall(all_text))
    if site_hits:
        score += 1.5 * min(site_hits, 3)
        reasons.append(f"+{1.5 * min(site_hits, 3):.1f}: {site_hits} site-plan keyword hit(s) in text")
    if floor_hits:
        score -= 0.4 * min(floor_hits, 3)
        reasons.append(f"-{0.4 * min(floor_hits, 3):.1f}: {floor_hits} floor-plan/room keyword hit(s) in text (weak on combined plan+elevation sheets -- see note)")

    # --- signal 2: an explicit road/ROW layer present ---
    has_road_layer = any(_ROAD_LAYER_RE.search(p.layer or "") for p in raw_polygons)
    if has_road_layer:
        score += 2.5
        reasons.append("+2.5: a polygon on a road/ROW-named layer is present")

    # --- signal 3: an explicit plot/boundary-named layer present ---
    has_plot_layer = any(_PLOT_LAYER_RE.search(p.layer or "") for p in raw_polygons)
    if has_plot_layer:
        score += 3.5
        reasons.append("+3.5: a polygon on a plot/boundary-named layer is present")

    # --- signal 4: dominant-outer-polygon geometry ---
    # A real site plan usually has ONE polygon (the plot) that's clearly
    # larger than everything else. Doesn't depend on layer-naming
    # convention at all, so it's the most portable signal across offices.
    areas = sorted((p.polygon.area for p in raw_polygons), reverse=True)
    if len(areas) >= 2 and areas[1] > 0:
        dominance_ratio = areas[0] / areas[1]
        if dominance_ratio >= 8.0:
            score += 4.5
            reasons.append(f"+4.5: largest polygon is {dominance_ratio:.1f}x the 2nd-largest (strongly dominant outer boundary)")
        elif dominance_ratio >= 3.0:
            score += 3.0
            reasons.append(f"+3.0: largest polygon is {dominance_ratio:.1f}x the 2nd-largest (dominant outer boundary)")
        elif dominance_ratio <= 1.5:
            score -= 0.75
            reasons.append(f"-0.75: largest polygon is only {dominance_ratio:.1f}x the 2nd-largest (looks like same-scale rooms, but see combined-sheet note)")
    elif len(areas) == 1:
        reasons.append("0.0: only one closed polygon found (no internal subdivision to compare against)")

    # --- signal 5: polygon count ---
    n_polys = len(raw_polygons)
    if n_polys >= 15:
        score -= 0.4
        reasons.append(f"-0.4: {n_polys} closed polygons (many rooms/partitions is floor-plan-like)")
    elif 0 < n_polys <= 4:
        score += 0.5
        reasons.append(f"+0.5: only {n_polys} closed polygon(s) (plot+building only is site-plan-like)")

    # --- signal 6: a BUILDING-layer polygon meaningfully smaller than the
    # largest polygon (footprint sitting inside a boundary) ---
    building_matches = [p for p in raw_polygons if _BUILDING_LAYER_RE.search(p.layer or "")]
    if building_matches and areas:
        largest_building_area = max(p.polygon.area for p in building_matches)
        if areas[0] > 0 and largest_building_area / areas[0] < 0.6:
            score += 1.0
            reasons.append("+1.0: a building-layer polygon covers well under the full drawing extent (sits inside a boundary)")

    return score, reasons


def _classify_one(path: Path) -> Verdict:
    try:
        import ezdxf
        from ezdxf import recover as ezdxf_recover

        try:
            doc = ezdxf.readfile(str(path))
        except ezdxf.DXFStructureError:
            doc, _ = ezdxf_recover.readfile(str(path))
        msp = doc.modelspace()
    except Exception as exc:
        return Verdict(filename=path.name, label="uncertain", score=0.0, error=f"could not parse: {exc}")

    extractor = DXFHybridExtractor()
    warnings: list[str] = []
    raw_polygons, raw_texts, _raw_dims, *_rest = extractor._collect_raw(msp, warnings)

    score, reasons = score_entities(raw_polygons, raw_texts)
    return Verdict(filename=path.name, label=label_for_score(score), score=round(score, 2), reasons=reasons)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dxf_dir", type=Path, help="Directory of real (ODA-converted) DXF files")
    parser.add_argument("--out-manifest", type=Path, default=Path("site_plan_classification.csv"))
    parser.add_argument(
        "--split-into",
        type=Path,
        default=None,
        help="If given, COPY files into <this>/site_plans, <this>/floor_plans, <this>/uncertain "
        "subfolders based on the verdict, instead of only writing the manifest.",
    )
    args = parser.parse_args()

    files = sorted(args.dxf_dir.glob("*.dxf"))
    if not files:
        print(f"No .dxf files found directly under {args.dxf_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Classifying {len(files)} DXF file(s)...")
    verdicts: list[Verdict] = []
    for i, f in enumerate(files, 1):
        v = _classify_one(f)
        verdicts.append(v)
        if i % 100 == 0 or i == len(files):
            print(f"  {i}/{len(files)}...")

    counts = {"site_plan": 0, "floor_plan": 0, "uncertain": 0}
    for v in verdicts:
        counts[v.label] += 1

    with args.out_manifest.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "label", "score", "reasons", "error"])
        for v in verdicts:
            writer.writerow([v.filename, v.label, v.score, " | ".join(v.reasons), v.error])
    print(f"\nWrote manifest: {args.out_manifest}")

    print("\n=== Summary ===")
    print(f"  site_plan  : {counts['site_plan']:5d}  ({counts['site_plan'] / len(files):.1%})")
    print(f"  floor_plan : {counts['floor_plan']:5d}  ({counts['floor_plan'] / len(files):.1%})")
    print(f"  uncertain  : {counts['uncertain']:5d}  ({counts['uncertain'] / len(files):.1%})")
    n_errors = sum(1 for v in verdicts if v.error)
    if n_errors:
        print(f"  ({n_errors} file(s) failed to parse -- see the manifest's 'error' column)")

    print(
        "\nSpot-check the manifest before trusting it -- sort by score and eyeball a handful near "
        "the +/-2.0 thresholds, since those are the ones the heuristic itself is least sure about. "
        "'uncertain' files are deliberately NOT auto-assigned either way; decide those by hand or "
        "just exclude them from training."
    )

    if args.split_into:
        for sub in ("site_plans", "floor_plans", "uncertain"):
            (args.split_into / sub).mkdir(parents=True, exist_ok=True)
        dest_map = {"site_plan": "site_plans", "floor_plan": "floor_plans", "uncertain": "uncertain"}
        copied = 0
        for f, v in zip(files, verdicts):
            if v.error:
                continue
            dest = args.split_into / dest_map[v.label] / f.name
            shutil.copy2(f, dest)
            copied += 1
        print(f"\nCopied {copied} file(s) into {args.split_into}/{{site_plans,floor_plans,uncertain}}/")


if __name__ == "__main__":
    main()
