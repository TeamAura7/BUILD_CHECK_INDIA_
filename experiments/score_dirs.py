"""Score plan.json directories with the repository scorer (same metric definitions
as analyze_extraction.py). Usage: python experiments/score_dirs.py DIR [DIR ...]"""
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from backend.corpus.runner import plan_to_predictions
from backend.corpus.scoring import score_plan
from backend.corpus.store import load_manifest, load_truth
from backend.corpus.schema import CANONICAL_FIELDS
from backend.schemas.normalized_plan import NormalizedPlan

def score_dir(d):
    d = Path(d); man = load_manifest(); rows = {}
    for e in man.plans:
        pj = d / f"{e.id}.plan.json"
        preds = plan_to_predictions(NormalizedPlan.model_validate_json(pj.read_text())) if pj.exists() else {}
        for s in score_plan(load_truth(e), preds):
            p = preds.get(s.field) or {}
            st = s.status
            if st == "MISSING" and p.get("confidence") == "CONFLICTING": st = "WITHHELD"
            rows[(e.id, s.field)] = dict(status=st, value=p.get("value"), conf=p.get("confidence"),
                                         flagged=bool(p.get("flagged")), truth=s.truth, unit=CANONICAL_FIELDS[s.field])
    return rows

def metrics(rows):
    C = sum(r["status"] == "CORRECT" for r in rows.values()); W = sum(r["status"] == "WRONG" for r in rows.values())
    M = sum(r["status"] == "MISSING" for r in rows.values()); Wh = sum(r["status"] == "WITHHELD" for r in rows.values())
    Sp = sum(r["status"] == "SPURIOUS" for r in rows.values())
    conf = [r for r in rows.values() if r["status"] in ("CORRECT", "WRONG", "SPURIOUS") and r["conf"] in ("HIGH", "MEDIUM") and not r["flagged"]]
    cw = sum(r["status"] != "CORRECT" for r in conf)
    return dict(C=C, W=W, M=M, Wh=Wh, Sp=Sp, CW=f"{cw}/{len(conf)}")

if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(Path(d).name, metrics(score_dir(d)))
