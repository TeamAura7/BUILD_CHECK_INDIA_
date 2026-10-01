"""Phase 6 on real data: what each reconciliation layer actually did on the 8 plans,
cross-tabulated with correctness against ground truth (default scorable tiers)."""
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

R = Path(__file__).resolve().parent / "results/fresh"
fl = list(csv.DictReader(open(R / "field_level.csv")))
ex = json.loads((R / "extraction_eval.json").read_text())
SCORED = ("CORRECT", "WRONG", "MISSING", "ABSTAINED_CONFLICT", "FALSE_POSITIVE", "ABSTAINED_CORRECTLY")
out = {}


def status_from_source(src):
    if not src:
        return "NO_VALUE"
    m = re.search(r":\s*(agreed|uncorroborated|minor_variance|outlier_detected|conflicting_evidence)\s+across", src)
    if m:
        return "LEGACY_" + m.group(1).upper()
    if src.startswith("FINAL:"):
        return src
    if src.startswith("evidence_decision:"):
        return "EVDEC_" + src.split(":", 1)[1].split("_DOCUMENT")[0]
    if src.startswith("legacy_fallback:"):
        return "EVDEC_LEGACY_FALLBACK"
    return "OTHER(" + src.split("=")[0].strip()[:40] + ")"


for v in ("pdf_hybrid", "pdf_legacy_resolver", "pdf_evidence_decision", "dxf_hybrid", "dxf_evidence_decision"):
    tab = defaultdict(Counter)
    for r in fl:
        if r["variant"] != v or r["status"] not in SCORED:
            continue
        tab[status_from_source(r["source"]) if r["predicted"] else ("CONFLICT_WITHHELD" if r["confidence"] == "CONFLICTING" else "NO_VALUE")][r["status"]] += 1
    out[v] = {k: dict(c) for k, c in sorted(tab.items())}

# within-document multi-candidate reconcile() calls (legacy path; computed in every PDF run)
calls = Counter()
ncand = Counter()
for mp in sorted((R / "variants/pdf_hybrid").glob("*.meta.json")):
    m = json.loads(mp.read_text())
    for c in m.get("reconcile_calls", []):
        calls[c["status"]] += 1
        ncand[min(c["n_candidates"], 4)] += 1
out["legacy_reconcile_calls_pdf_hybrid"] = {"status": dict(calls), "n_candidates(4=4+)": dict(ncand)}

# PDF <-> DXF reconciliation, scored
truth = {(r["plan"], r["field"]): (r["truth"], r["truth_tier"]) for r in fl if r["variant"] == "pdf_hybrid"}
recon = Counter()
recon_ok = defaultdict(Counter)
for pid, rep in ex["pdf_dxf_reconciliation_reports"].items():
    for c in rep["fields"]:
        recon[c["status"]] += 1
        t, tier = truth.get((pid, c["field"]), (None, None))
        if tier not in ("printed", "derived", "inspected", "legacy") or t in ("", "None", None):
            recon_ok[c["status"]]["no_truth"] += 1
            continue
        t = float(t)
        ok = lambda x: x is not None and abs(x - t) <= max(0.15, 0.05 * abs(t))
        if c["status"] == "CONFLICT":
            recon_ok[c["status"]]["resolved_correctly(PDF right)" if ok(c["pdf_value"]) else
                                  ("resolved_wrongly(DXF right)" if ok(c["dxf_value"]) else "both_wrong")] += 1
        elif c["status"] == "AGREED":
            recon_ok[c["status"]]["agreed_correct" if ok(c["pdf_value"]) else "agreed_wrong"] += 1
        elif c["status"] == "PDF_ONLY":
            recon_ok[c["status"]]["pdf_correct" if ok(c["pdf_value"]) else "pdf_wrong"] += 1
        elif c["status"] == "DXF_ONLY":
            recon_ok[c["status"]]["dxf_correct_not_shipped" if ok(c["dxf_value"]) else "dxf_wrong_not_shipped"] += 1
        else:
            recon_ok[c["status"]]["both_missing"] += 1
out["pdf_dxf_reconciliation"] = {"status_counts": dict(recon), "scored": {k: dict(v) for k, v in recon_ok.items()},
                                 "note": "COMPARED_FIELDS = 14 per plan x 8 plans = 112 comparisons"}
(R / "reconciliation_stats.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
