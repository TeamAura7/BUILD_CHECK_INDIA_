"""Tolerance-vs-threshold analysis: for every GT-decidable rule, how far is the
ground-truth value from the rule threshold, compared with the extraction
evaluation tolerance max(0.15, 5%*truth)? A rule whose margin is smaller than
the tolerance can flip even when the extraction is scored CORRECT."""
import csv
import json
import re
from pathlib import Path

R = Path(__file__).resolve().parent / "results/fresh"
rows = [r for r in csv.DictReader(open(R / "compliance_rule_level.csv"))
        if r["development_area"] == "A" and r["gt_status"] in ("PASS", "FAIL")]
pat = re.compile(r"^([\w.]+)\s*(>=|<=|>|<|==)\s*([0-9.]+)")
field_rows = {(r["plan"], r["field"], r["variant"]): r for r in csv.DictReader(open(R / "field_level.csv"))}
out = []
seen = set()
for r in rows:
    m = pat.match(r["requirement"] or "")
    obs = json.loads(r["gt_observed"])
    if not m or len(obs) != 1 or obs[0] is None:
        continue
    fld, op, thr = m.group(1), m.group(2), float(m.group(3))
    gt = float(obs[0])
    tol = max(0.15, 0.05 * abs(gt))
    key = (r["plan"], r["rule_id"])
    rec = {"plan": r["plan"], "rule_id": r["rule_id"], "field": fld, "op": op, "threshold": thr, "gt_value": gt,
           "gt_status": r["gt_status"], "margin": round(abs(gt - thr), 4), "eval_tolerance": round(tol, 4),
           "fragile": abs(gt - thr) < tol}
    for v in ("pdf_hybrid", "dxf_hybrid", "pdf_legacy_resolver"):
        vr = next((x for x in rows if x["variant"] == v and x["plan"] == r["plan"] and x["rule_id"] == r["rule_id"]), None)
        fr = field_rows.get((r["plan"], fld, v))
        rec[f"{v}_pred_status"] = vr["pred_status"] if vr else None
        rec[f"{v}_field_status"] = fr["status"] if fr else None
        rec[f"{v}_pred_value"] = fr["predicted"] if fr else None
    if key not in seen:
        seen.add(key)
        out.append(rec)
with open(R / "threshold_margin.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
fr = [o for o in out if o["fragile"]]
print(f"GT-decidable single-field rules: {len(out)}; margin < eval tolerance: {len(fr)}")
for o in fr:
    print(" ", o["plan"], o["rule_id"], o["field"], o["op"], o["threshold"], "gt", o["gt_value"], "tol", o["eval_tolerance"],
          "| pdf:", o["pdf_hybrid_field_status"], o["pdf_hybrid_pred_value"], o["pdf_hybrid_pred_status"])
flip_ok = [o for o in out if o["pdf_hybrid_field_status"] == "CORRECT" and o["pdf_hybrid_pred_status"] not in (o["gt_status"],)]
print("pdf_hybrid: extraction CORRECT but decision differs:", [(o["plan"], o["rule_id"]) for o in flip_ok])
