"""Phase 2: dataset composition table from the corpus manifest + truth files (no extraction)."""
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from backend.corpus.schema import CANONICAL_FIELDS, SCORABLE_DEFAULT
from backend.corpus.store import load_manifest, load_truth

EXTRA = ("plot.net_area", "building.gross_built_up_area", "far.area")
rows, cells = [], []
for e in load_manifest().plans:
    t = load_truth(e)
    exp_path = REPO / f"data/test_plans/{e.id}.expected.json"
    exp = json.loads(exp_path.read_text()) if exp_path.exists() else {}
    tiers = {}
    for f in CANONICAL_FIELDS:
        tf = t.fields.get(f)
        tier = tf.verification if tf else "absent"
        tiers[f] = tier
        cells.append({"plan": e.id, "field": f, "truth": tf.value if tf else None, "tier": tier,
                      "scored": tier in SCORABLE_DEFAULT and tf is not None and tf.value is not None,
                      "must_abstain": tier == "must_abstain"})
    for f in EXTRA:
        cells.append({"plan": e.id, "field": f, "truth": exp.get(f), "tier": "legacy_expected_json" if f in exp else "absent",
                      "scored": False, "must_abstain": False})
    pdf = REPO / e.pdf if e.pdf else None
    dxf = REPO / e.dxf if e.dxf else None
    rows.append({
        "plan": e.id, "split": e.split, "tags": ";".join(e.tags),
        "pdf": bool(pdf and pdf.exists()), "pdf_kb": round(pdf.stat().st_size / 1024) if pdf else None,
        "dxf": bool(dxf and dxf.exists()), "dxf_kb": round(dxf.stat().st_size / 1024) if dxf else None,
        "gt_file": True, "human_verified": t.annotation.human_verified,
        "n_scored_fields": sum(1 for f in CANONICAL_FIELDS if tiers[f] in SCORABLE_DEFAULT and t.fields[f].value is not None),
        "n_must_abstain": sum(1 for f in CANONICAL_FIELDS if tiers[f] == "must_abstain"),
        "n_unverified_excluded": sum(1 for f in CANONICAL_FIELDS if tiers[f] == "unverified"),
        "n_absent": sum(1 for f in CANONICAL_FIELDS if tiers[f] == "absent"),
        "tiers": json.dumps({k: sum(1 for f in CANONICAL_FIELDS if tiers[f] == k) for k in sorted(set(tiers.values()))}),
        "supplementary_legacy_fields": ";".join(f for f in EXTRA if f in exp),
        "n_discrepancies_recorded": len(t.discrepancies),
    })
out = REPO / "experiments/results/fresh"
out.mkdir(parents=True, exist_ok=True)
for name, data in (("dataset.csv", rows), ("dataset_field_matrix.csv", cells)):
    with open(out / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)
for r in rows:
    print(r)
