"""Phase 5: is the DXF error a consistent scale factor (the documented 'quarter-scale'
failure) or something else? For each plan, the ratio predicted/truth of every answered
length field (m) and area field (m2); a pure scale error s gives length ratios == s and
area ratios == s^2."""
import csv, json, statistics
from pathlib import Path
R = Path(__file__).resolve().parent / "results/fresh"
fl = [r for r in csv.DictReader(open(R / "field_level.csv")) if r["variant"] == "dxf_hybrid"]
L = ("plot.width", "plot.depth", "building.width", "building.depth")
A = ("plot.area", "building.footprint_area")
rows = []
for plan in sorted({r["plan"] for r in fl}):
    pr = {r["field"]: r for r in fl if r["plan"] == plan}
    lr = []
    for f in L:
        r = pr[f]
        if r["predicted"] and r["truth"] not in ("", "None") and r["truth_tier"] in ("printed", "derived", "inspected", "legacy"):
            lr.append(float(r["predicted"]) / float(r["truth"]))
    ar = []
    for f in A:
        r = pr[f]
        if r["predicted"] and r["truth"] not in ("", "None") and r["truth_tier"] in ("printed", "derived", "inspected", "legacy"):
            ar.append(float(r["predicted"]) / float(r["truth"]))
    meta = json.loads((R / f"variants/dxf_hybrid/{plan}.meta.json").read_text())
    unit_w = next((w for w in meta.get("warnings", []) if w.startswith("Unit resolution")), "")
    rows.append({"plan": plan, "length_ratios": json.dumps([round(x, 3) for x in lr]),
                 "area_ratios": json.dumps([round(x, 3) for x in ar]),
                 "length_ratio_median": round(statistics.median(lr), 3) if lr else None,
                 "length_ratio_spread(max/min)": round(max(lr) / min(lr), 2) if len(lr) > 1 and min(lr) > 0 else None,
                 "sqrt_area_ratio_median": round(statistics.median(ar) ** 0.5, 3) if ar else None,
                 "consistent_single_scale": (len(lr) > 1 and min(lr) > 0 and max(lr) / min(lr) < 1.15),
                 "unit_resolution": unit_w[:160],
                 "n_answered": sum(1 for r in pr.values() if r["predicted"]),
                 "n_low": sum(1 for r in pr.values() if r["predicted"] and r["confidence"] == "LOW")})
with open(R / "dxf_scale_analysis.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows: print(r["plan"], r["length_ratios"], r["area_ratios"], r["length_ratio_spread(max/min)"], r["consistent_single_scale"], r["n_low"], "/", r["n_answered"])
