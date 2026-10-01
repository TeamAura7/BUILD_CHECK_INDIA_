"""
Split each DXF into spatially-separated "regions" (a site/key plan, a floor
plan, an elevation, etc -- commonly all present on one combined sanctioned-
plan sheet), score EACH region independently as site-plan-like or not
(reusing the exact same heuristic as classify_site_vs_floor_plans.py), and
extract the winning region into its own small, clean DXF -- ready to feed
straight into backend.gnn_extraction.train without graph_builder.py ever
seeing the hundreds of room/door/window polygons from the floor-plan and
elevation portions of the same sheet.

Why file-level classification (classify_site_vs_floor_plans.py) isn't
enough on its own: even a file that scores strongly "site_plan" there can
still contain hundreds of unrelated closed polygons, because combined
submission sheets put a site/key plan, a full floor plan, and an elevation
all in the same DXF's modelspace, at different (but usually
non-overlapping) positions. This script isolates just the spatial region
that actually looks like the site plan, using the same scoring function
applied to each region instead of the whole file.

Usage:
    python -m scripts.split_dxf_into_site_regions /path/to/dxf_dir \
        --out-dir /path/to/site_plan_regions \
        --gap-fraction 0.015

    python -m backend.gnn_extraction.train /path/to/site_plan_regions --epochs 100
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from backend.cv_extraction.dxf_extractor import _RawPoly, _RawText, DXFHybridExtractor
from scripts.classify_site_vs_floor_plans import score_entities


@dataclass
class _Item:
    kind: str  # "polygon" | "text"
    index: int  # index into raw_polygons or raw_texts
    bbox: tuple[float, float, float, float]  # min_x, min_y, max_x, max_y (already unit-scaled)


def _bbox_gap(a: tuple, b: tuple) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def _bbox_center(bbox: tuple) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _robust_extent(items: "list[_Item]", outlier_factor: float = 20.0) -> tuple[float, float, float, "list[_Item]"]:
    """
    Bounding-box extent of `items`, robust to a single stray/misplaced
    entity (leftover construction geometry, an accidental far-off click --
    extremely common in real AutoCAD files). A plain min/max over every
    item's bbox lets ONE such point balloon the extent arbitrarily, which
    then corrupts the gap threshold derived from it (gap_fraction * extent)
    for the entire file -- exactly the failure mode found by hand-tracing a
    real corpus file with a stray point (23 polygons got merged into one
    cluster instead of the correct 3).

    Robust by NEAREST-NEIGHBOUR distance, not distance-from-centroid/median:
    a genuinely isolated stray point has nothing near it at all (its
    nearest neighbour is enormously far away), whereas every point in a
    real -- even small -- separate region (e.g. a 5-item site plan sitting
    apart from a 21-item floor plan) has other points from ITS OWN region
    close by. That distinction is what makes this safe against mistaking a
    legitimately separate but small region for an outlier, which a simpler
    "far from the global median" check does not reliably avoid (a large
    region can drag the median toward itself and make a smaller, but
    perfectly normal, region look anomalous).

    Returns (extent_w, extent_h, extent_diag, kept_items) where kept_items
    excludes only the detected outliers -- clustering itself still runs on
    every item, including outliers (each ends up correctly isolated in its
    own cluster); only the DERIVED EXTENT used to size the gap threshold
    ignores them.
    """
    if len(items) <= 2:
        xs = [v for it in items for v in (it.bbox[0], it.bbox[2])]
        ys = [v for it in items for v in (it.bbox[1], it.bbox[3])]
        return (max(xs) - min(xs) if xs else 0.0, max(ys) - min(ys) if ys else 0.0,
                math.hypot(max(xs) - min(xs), max(ys) - min(ys)) if xs else 0.0, items)

    centers = [_bbox_center(it.bbox) for it in items]
    nn_dists = []
    for i, c in enumerate(centers):
        best = min(math.hypot(c[0] - centers[j][0], c[1] - centers[j][1]) for j in range(len(centers)) if j != i)
        nn_dists.append(best)

    sorted_nn = sorted(nn_dists)
    mid = len(sorted_nn) // 2
    median_nn = sorted_nn[mid] if len(sorted_nn) % 2 else (sorted_nn[mid - 1] + sorted_nn[mid]) / 2.0

    if median_nn <= 0:
        kept = items
    else:
        threshold = median_nn * outlier_factor
        kept = [it for it, d in zip(items, nn_dists) if d <= threshold]
        # Never trim so aggressively that almost nothing is left -- if that
        # happens the "outliers" are more likely a real, unusual drawing
        # than a stray point, so fall back to using everything.
        if len(kept) < max(2, len(items) // 4):
            kept = items

    xs = [v for it in kept for v in (it.bbox[0], it.bbox[2])]
    ys = [v for it in kept for v in (it.bbox[1], it.bbox[3])]
    extent_w = max(xs) - min(xs)
    extent_h = max(ys) - min(ys)
    return extent_w, extent_h, math.hypot(extent_w, extent_h), kept


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _cluster_items(items: list[_Item], gap_threshold: float) -> list[list[int]]:
    """
    Grid-bucketed union-find: entities whose bounding boxes are within
    `gap_threshold` of each other end up in the same cluster. Runs well
    below the O(n^2) of a naive all-pairs comparison by only comparing
    entities that share or neighbor a grid cell sized to the threshold --
    matters at the scale of a real corpus, where some combined sheets have
    close to a thousand polygons.
    """
    n = len(items)
    if n == 0:
        return []
    cell = max(gap_threshold, 1e-6)
    grid: dict[tuple[int, int], list[int]] = {}

    def cell_range(bbox):
        return (
            int(bbox[0] // cell), int(bbox[1] // cell),
            int(bbox[2] // cell), int(bbox[3] // cell),
        )

    ranges = [cell_range(item.bbox) for item in items]
    for i, (min_cx, min_cy, max_cx, max_cy) in enumerate(ranges):
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                grid.setdefault((cx, cy), []).append(i)

    uf = _UnionFind(n)
    checked: set[tuple[int, int]] = set()
    for i, (min_cx, min_cy, max_cx, max_cy) in enumerate(ranges):
        for cx in range(min_cx - 1, max_cx + 2):
            for cy in range(min_cy - 1, max_cy + 2):
                for j in grid.get((cx, cy), ()):
                    if j <= i:
                        continue
                    pair = (i, j)
                    if pair in checked:
                        continue
                    checked.add(pair)
                    if _bbox_gap(items[i].bbox, items[j].bbox) <= gap_threshold:
                        uf.union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)
    return list(clusters.values())


def _poly_bbox(p: _RawPoly) -> tuple[float, float, float, float]:
    xs = [pt.x for pt in p.polygon.points]
    ys = [pt.y for pt in p.polygon.points]
    return (min(xs), min(ys), max(xs), max(ys))


def _text_bbox(t: _RawText, eps: float = 0.05) -> tuple[float, float, float, float]:
    x, y = t.position
    return (x - eps, y - eps, x + eps, y + eps)


def _write_region_dxf(out_path: Path, insunits: Optional[int], polys: list[_RawPoly], texts: list[_RawText]) -> None:
    import ezdxf

    doc = ezdxf.new("R2018")
    if insunits:
        doc.header["$INSUNITS"] = insunits
    msp = doc.modelspace()
    for p in polys:
        if p.layer not in doc.layers:
            doc.layers.add(p.layer)
        pts = [(pt.x, pt.y) for pt in p.polygon.points]
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": p.layer})
    for t in texts:
        if t.layer not in doc.layers:
            doc.layers.add(t.layer)
        entity = msp.add_text(t.text, dxfattribs={"layer": t.layer})
        entity.dxf.insert = t.position
    doc.saveas(str(out_path))


def process_file(path: Path, gap_fraction: float, min_score: float, min_texts: int = 1, max_polygons: int = 60) -> dict:
    import ezdxf
    from ezdxf import recover as ezdxf_recover

    try:
        try:
            doc = ezdxf.readfile(str(path))
        except ezdxf.DXFStructureError:
            doc, _ = ezdxf_recover.readfile(str(path))
        msp = doc.modelspace()
    except Exception as exc:
        return {"filename": path.name, "n_regions": 0, "chosen_score": None, "chosen_polygons": 0, "chosen_texts": 0, "status": f"parse_error: {exc}"}

    extractor = DXFHybridExtractor()
    warnings: list[str] = []
    raw_polygons, raw_texts, _raw_dims, *_rest = extractor._collect_raw(msp, warnings)
    if not raw_polygons:
        return {"filename": path.name, "n_regions": 0, "chosen_score": None, "chosen_polygons": 0, "chosen_texts": 0, "status": "no_polygons"}

    # NOTE: unit resolution (_resolve_unit_factor) is DELIBERATELY not used
    # to size the clustering threshold. It works by sanity-checking the
    # single largest closed polygon in the WHOLE FILE against a plausible
    # plot-area range -- but on a combined plan+elevation+floor-plan sheet,
    # that largest polygon is very often a sheet border/title-block frame or
    # an elevation outline, not the plot. Get that one guess wrong and the
    # gap threshold below (previously an absolute "gap_metres" run through
    # that same guess) becomes wrong for the ENTIRE file, either fragmenting
    # a single plot boundary into hundreds of clusters or merging distinct
    # views together. Clustering directly in RAW (unscaled) drawing-unit
    # space, with a threshold sized as a FRACTION of the file's own overall
    # extent, sidesteps unit-guessing for this step entirely -- it's scale-
    # invariant by construction, so it can't be thrown off by a wrong m/unit
    # factor. score_entities() itself needs no absolute scale either
    # (dominance ratio, keyword counts, and layer-name matches are all
    # scale-invariant), so nothing downstream of clustering loses anything.
    insunits = None
    try:
        insunits = doc.header.get("$INSUNITS")
    except Exception:
        pass

    items: list[_Item] = []
    for i, p in enumerate(raw_polygons):
        items.append(_Item(kind="polygon", index=i, bbox=_poly_bbox(p)))
    for i, t in enumerate(raw_texts):
        items.append(_Item(kind="text", index=i, bbox=_text_bbox(t)))

    extent_w, extent_h, extent_diag, _extent_items = _robust_extent(items)

    # A title-block/sheet-border rectangle spans nearly the FULL extent of
    # the drawing -- if left in, its bbox overlaps every other region's
    # bbox directly (gap = 0), silently bridging genuinely separate views
    # (site plan / floor plan / elevation) into one cluster no matter how
    # small the gap threshold is. It's also never a legitimate plot/building
    # candidate, so it's excluded from clustering (and therefore from
    # scoring) entirely, not just down-weighted. Only fires when there's
    # other geometry it would otherwise bridge together -- on a sparse
    # sheet where a real plot polygon happens to be most of the drawing,
    # there's nothing for it to bridge, so excluding it costs nothing.
    poly_items = [it for it in items if it.kind == "polygon"]
    non_border_items = []
    for it in items:
        w = it.bbox[2] - it.bbox[0]
        h = it.bbox[3] - it.bbox[1]
        spans_full_extent = extent_w > 0 and extent_h > 0 and w >= 0.9 * extent_w and h >= 0.9 * extent_h
        if it.kind == "polygon" and spans_full_extent and len(poly_items) > 1:
            continue
        non_border_items.append(it)
    items = non_border_items

    gap_threshold = max(extent_diag * gap_fraction, 1e-9)

    clusters = _cluster_items(items, gap_threshold)

    # Score every cluster, but prefer the best-scoring one that also has at
    # least `min_texts` corroborating text entities and at most
    # `max_polygons` polygons. min_texts guards against a bare 2-3 polygon
    # cluster clearing min_score on geometry alone (dominance ratio + few-
    # polygons bonus, no text). max_polygons guards the opposite failure:
    # verified against a real 948-file extraction where the clustering
    # still occasionally (a handful of files, not most -- median was a
    # healthy 14 polygons) merges far more than a real site plan into one
    # cluster, up to 7,352 polygons in the worst case. A handful of such
    # files can single-handedly dominate a training run: in one real
    # validation split, "other" (the catch-all for everything that isn't
    # plot/building/road) was ~10,400 of ~11,700 total nodes -- traced back
    # to just 2-3 oversized files, not a systemic problem across the corpus.
    # A real site plan region (plot + building + a handful of annotations)
    # essentially never legitimately needs dozens of polygons, let alone
    # thousands, so a hard cap catches this cheaply. If NO cluster in the
    # file has any text at all, or NONE stay under the cap, fall through to
    # the plain best-by-score cluster so the file still gets a real (likely
    # sub-threshold) verdict instead of silently picking nothing.
    best = None  # (score, poly_idxs, text_idxs) -- best regardless of text/size
    best_grounded = None  # (score, poly_idxs, text_idxs) -- best with text_idxs >= min_texts AND poly count <= max_polygons
    for cluster_idxs in clusters:
        poly_idxs = [items[i].index for i in cluster_idxs if items[i].kind == "polygon"]
        text_idxs = [items[i].index for i in cluster_idxs if items[i].kind == "text"]
        if not poly_idxs:
            continue
        cluster_polys = [raw_polygons[i] for i in poly_idxs]
        cluster_texts = [raw_texts[i] for i in text_idxs]
        score, _reasons = score_entities(cluster_polys, cluster_texts)
        candidate = (score, poly_idxs, text_idxs)
        if best is None or score > best[0]:
            best = candidate
        if len(text_idxs) >= min_texts and len(poly_idxs) <= max_polygons and (best_grounded is None or score > best_grounded[0]):
            best_grounded = candidate

    chosen = best_grounded if best_grounded is not None else best
    if chosen is None:
        return {"filename": path.name, "n_regions": len(clusters), "chosen_score": None, "chosen_polygons": 0, "chosen_texts": 0, "status": "no_polygon_cluster"}

    score, poly_idxs, text_idxs = chosen
    if score < min_score:
        status = "best_region_below_threshold"
    elif len(text_idxs) < min_texts:
        status = "no_text_grounding"
    elif len(poly_idxs) > max_polygons:
        status = "region_too_large"
    else:
        status = "extracted"
    return {
        "filename": path.name,
        "n_regions": len(clusters),
        "chosen_score": round(score, 2),
        "chosen_polygons": len(poly_idxs),
        "chosen_texts": len(text_idxs),
        "status": status,
        "_poly_idxs": poly_idxs,
        "_text_idxs": text_idxs,
        "_raw_polygons": raw_polygons,
        "_raw_texts": raw_texts,
        "_insunits": insunits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dxf_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--gap-fraction", type=float, default=0.015,
        help="Entities within this fraction of the file's own overall drawing extent (diagonal) "
        "are considered the same region -- scale-invariant, so it doesn't depend on guessing the "
        "file's real-world units. Tune by inspecting n_regions in the manifest: too small "
        "over-splits one drawing into many tiny pieces, too large merges genuinely separate "
        "views (site plan / floor plan / elevation) together.",
    )
    parser.add_argument(
        "--min-score", type=float, default=2.0,
        help="A region's score (same scale as classify_site_vs_floor_plans.py) must reach this to be extracted.",
    )
    parser.add_argument(
        "--min-texts", type=int, default=1,
        help="A region also needs at least this many text entities to be extracted, even if its score "
        "alone clears --min-score. A bare 2-3 polygon cluster can pass on geometry alone (dominance "
        "ratio + few-polygons bonus) with nothing -- no SETBACK/SITE PLAN label, not even a room name -- "
        "corroborating it as a real site plan; such regions land in 'no_text_grounding' instead.",
    )
    parser.add_argument(
        "--max-polygons", type=int, default=60,
        help="A region with more than this many polygons is rejected as 'region_too_large' instead of "
        "extracted. A real site plan (plot + building + a handful of annotations) essentially never "
        "legitimately needs dozens of polygons, let alone the thousands a badly-merged cluster can reach; "
        "a handful of such oversized regions can single-handedly dominate a training run's node counts.",
    )
    parser.add_argument("--out-manifest", type=Path, default=Path("site_region_extraction.csv"))
    args = parser.parse_args()

    files = sorted(args.dxf_dir.glob("*.dxf"))
    if not files:
        print(f"No .dxf files found under {args.dxf_dir}", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    n_extracted = 0
    print(f"Processing {len(files)} file(s)...")
    for i, f in enumerate(files, 1):
        result = process_file(f, args.gap_fraction, args.min_score, args.min_texts, args.max_polygons)
        rows.append({k: v for k, v in result.items() if not k.startswith("_")})
        if result["status"] == "extracted":
            polys = [result["_raw_polygons"][j] for j in result["_poly_idxs"]]
            texts = [result["_raw_texts"][j] for j in result["_text_idxs"]]
            _write_region_dxf(args.out_dir / f.name, result["_insunits"], polys, texts)
            n_extracted += 1
        if i % 100 == 0 or i == len(files):
            print(f"  {i}/{len(files)}...")

    with args.out_manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "n_regions", "chosen_score", "chosen_polygons", "chosen_texts", "status"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote manifest: {args.out_manifest}")
    print(f"Extracted {n_extracted}/{len(files)} file(s) into {args.out_dir}")
    print("\nStatus breakdown:", dict(Counter(r["status"] for r in rows)))


if __name__ == "__main__":
    main()