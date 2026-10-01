"""
Phases 3-6 scoring. Uses ONLY the repository's own scorer
(backend/corpus/scoring.py: ABS_TOL=0.15 m, REL_TOL=5 %, unordered axis
pairs, tiers printed/derived/inspected/legacy) and adds the extra metrics
the paper needs (MAE, MAPE, completeness, abstention breakdown).

Inputs : experiments/results/fresh/variants/<variant>/<PLAN>.plan.json|meta.json
Outputs: experiments/results/fresh/
           field_level.csv          one row per (variant, plan, field)
           plan_level.csv           one row per (variant, plan)
           pipeline_comparison.csv  one row per variant
           field_summary.csv        one row per (variant, field)
           supplementary_fields.csv plot.net_area / far.area / gross BUA (independent-CV layer only)
           extraction_eval.json     everything above + run metadata
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.corpus.runner import plan_to_predictions
from backend.corpus.schema import AXIS_PAIRS, CANONICAL_FIELDS, SCORABLE_DEFAULT
from backend.corpus.scoring import ABS_TOL, REL_TOL, aggregate, score_plan, within_tolerance
from backend.corpus.store import load_manifest, load_truth
from backend.schemas.normalized_plan import NormalizedPlan

FRESH = REPO / "experiments/results/fresh"
VDIR = FRESH / "variants"
PARTNER = {a: b for a, b in AXIS_PAIRS} | {b: a for a, b in AXIS_PAIRS}
VARIANT_ORDER = ["pdf_native_only", "pdf_native_ocr", "pdf_hybrid", "pdf_legacy_resolver",
                 "pdf_evidence_decision", "dxf_hybrid", "dxf_evidence_decision", "pdf_dxf_reconciled"]


def load_runs():
    from backend.spatial_reasoning.pdf_dxf_reconciliation import reconcile_pdf_dxf
    runs = defaultdict(dict)  # variant -> plan -> (plan|None, meta)
    for vdir in sorted(p for p in VDIR.iterdir() if p.is_dir()):
        for mp in sorted(vdir.glob("*.meta.json")):
            pid = mp.name.split(".")[0]
            meta = json.loads(mp.read_text(encoding="utf-8"))
            pj = vdir / f"{pid}.plan.json"
            plan = NormalizedPlan.model_validate_json(pj.read_text(encoding="utf-8")) if pj.exists() else None
            runs[vdir.name][pid] = (plan, meta)
    recon_reports = {}
    for pid, (pp, pm) in runs.get("pdf_hybrid", {}).items():
        dx = runs.get("dxf_hybrid", {}).get(pid)
        if dx is None:
            continue
        if pp is not None and dx[0] is not None:
            rec, rep = reconcile_pdf_dxf(pp, dx[0])
            runs["pdf_dxf_reconciled"][pid] = (rec, {"status": "ok", "total_seconds": None})
            recon_reports[pid] = json.loads(rep.model_dump_json())
        else:
            runs["pdf_dxf_reconciled"][pid] = (None, {"status": "needs both pdf and dxf plans"})
    return runs, recon_reports


def matched_truth(s, truth_vals):
    if s.axis_swapped and s.field in PARTNER:
        return truth_vals.get(PARTNER[s.field])
    if (s.field in PARTNER and s.predicted is not None and s.truth is not None
            and s.status == "CORRECT" and not within_tolerance(float(s.predicted), float(s.truth))):
        return truth_vals.get(PARTNER[s.field])
    return s.truth


def main():
    manifest = load_manifest()
    truths = {e.id: load_truth(e) for e in manifest.plans}
    runs, recon_reports = load_runs()
    field_rows = []
    for variant in [v for v in VARIANT_ORDER if v in runs] + [v for v in runs if v not in VARIANT_ORDER]:
        for pid, truth in truths.items():
            plan, meta = runs[variant].get(pid, (None, {"status": "not_run"}))
            preds = plan_to_predictions(plan) if plan is not None else {}
            scores = {s.field: s for s in score_plan(truth, preds)}
            tvals = {k: tf.value for k, tf in truth.fields.items()}
            for name in CANONICAL_FIELDS:
                tf = truth.fields.get(name)
                p = preds.get(name) or {}
                row = {"variant": variant, "plan": pid, "field": name, "run_status": meta.get("status"),
                       "truth": tf.value if tf else None, "truth_tier": tf.verification if tf else "absent",
                       "predicted": p.get("value"), "confidence": p.get("confidence"),
                       "flagged": bool(p.get("flagged")), "source": None}
                if plan is not None:
                    obj = plan
                    for part in name.split("."):
                        obj = getattr(obj, part, None) if obj is not None else None
                    row["source"] = getattr(obj, "source", None) if obj is not None else None
                s = scores.get(name)
                if s is None:
                    if tf is None or tf.verification == "unannotated":
                        row["status"] = "N/A_NO_TRUTH"
                    elif tf.verification == "unverified":
                        row["status"] = "N/A_UNVERIFIED_TRUTH"
                    else:
                        row["status"] = "N/A"
                    row["abs_error"] = row["pct_error"] = None
                    row["axis_swapped"] = False
                else:
                    st = s.status
                    if st == "MISSING" and p.get("confidence") == "CONFLICTING":
                        st = "ABSTAINED_CONFLICT"   # system explicitly refused: evidence conflicted
                    elif st == "SPURIOUS":
                        st = "FALSE_POSITIVE"
                    elif st == "ABSTAINED_OK":
                        st = "ABSTAINED_CORRECTLY"
                    row["status"] = st
                    row["axis_swapped"] = s.axis_swapped
                    mt = matched_truth(s, tvals)
                    unit = CANONICAL_FIELDS[name]
                    if s.predicted is not None and mt is not None and unit not in ("category",):
                        err = abs(float(s.predicted) - float(mt))
                        row["abs_error"] = round(err, 4)
                        row["pct_error"] = round(100 * err / abs(float(mt)), 3) if float(mt) != 0 else None
                    else:
                        row["abs_error"] = row["pct_error"] = None
                    row["confident"] = s.confident
                field_rows.append(row)

    def metrics(rows):
        valued = [r for r in rows if r["status"] in ("CORRECT", "WRONG", "MISSING", "ABSTAINED_CONFLICT")]
        answered = [r for r in valued if r["status"] in ("CORRECT", "WRONG")]
        correct = sum(1 for r in answered if r["status"] == "CORRECT")
        must = [r for r in rows if r["status"] in ("FALSE_POSITIVE", "ABSTAINED_CORRECTLY")]
        conf = [r for r in answered + [r for r in must if r["status"] == "FALSE_POSITIVE"]
                if r["confidence"] in ("HIGH", "MEDIUM") and not r["flagged"]]
        conf_wrong = [r for r in conf if r["status"] != "CORRECT"]
        wrong = [r for r in answered if r["status"] == "WRONG"]
        # MAE / MAPE: length fields (m) only, answered only -- mixing m, m2, %, ratio in one mean is meaningless.
        len_err = [r["abs_error"] for r in answered if CANONICAL_FIELDS[r["field"]] == "m" and r["abs_error"] is not None]
        pct = [r["pct_error"] for r in answered if r["pct_error"] is not None and CANONICAL_FIELDS[r["field"]] != "category"]
        return {
            "truth_valued": len(valued), "answered": len(answered), "correct": correct,
            "wrong": len(wrong), "missing": sum(1 for r in valued if r["status"] == "MISSING"),
            "abstained_conflict": sum(1 for r in valued if r["status"] == "ABSTAINED_CONFLICT"),
            "must_abstain_fields": len(must),
            "false_positive": sum(1 for r in must if r["status"] == "FALSE_POSITIVE"),
            "abstained_correctly": sum(1 for r in must if r["status"] == "ABSTAINED_CORRECTLY"),
            "accuracy": (correct / len(answered)) if answered else None,
            "completeness": (len(answered) / len(valued)) if valued else None,
            "correct_over_truth_valued": (correct / len(valued)) if valued else None,
            "mae_m_length_fields": (sum(len_err) / len(len_err)) if len_err else None,
            "n_mae": len(len_err),
            "mape_pct_all_numeric": (sum(pct) / len(pct)) if pct else None,
            "median_ape_pct": (sorted(pct)[len(pct) // 2] if pct else None),
            "n_mape": len(pct),
            "confident_answers": len(conf), "confident_wrong": len(conf_wrong),
            "confident_wrong_rate": (len(conf_wrong) / len(conf)) if conf else None,
            "wrong_flagged_or_low": sum(1 for r in wrong if r["flagged"] or r["confidence"] == "LOW"),
            "wrong_caught_rate": (sum(1 for r in wrong if r["flagged"] or r["confidence"] == "LOW") / len(wrong)) if wrong else None,
            "false_positive_rate_on_must_abstain": (sum(1 for r in must if r["status"] == "FALSE_POSITIVE") / len(must)) if must else None,
            "axis_swapped": sum(1 for r in answered if r.get("axis_swapped")),
        }

    by_v = defaultdict(list)
    by_vp = defaultdict(list)
    by_vf = defaultdict(list)
    for r in field_rows:
        by_v[r["variant"]].append(r)
        by_vp[(r["variant"], r["plan"])].append(r)
        by_vf[(r["variant"], r["field"])].append(r)

    pipeline = []
    for v, rows in by_v.items():
        metas = [runs[v][p][1] for p in runs[v]]
        secs = [m.get("total_seconds") for m in metas if isinstance(m.get("total_seconds"), (int, float))]
        pipeline.append({"variant": v, **metrics(rows),
                         "runs_ok": sum(1 for m in metas if m.get("status") == "ok"),
                         "runs_failed_or_timeout": sum(1 for m in metas if m.get("status") != "ok"),
                         "internal_timeouts": sum(1 for m in metas if m.get("timed_out_internally")),
                         "mean_seconds": (sum(secs) / len(secs)) if secs else None,
                         "max_seconds": max(secs) if secs else None})
    plan_level = []
    for (v, p), rows in by_vp.items():
        m = runs[v].get(p, (None, {}))[1]
        plan_level.append({"variant": v, "plan": p, "run_status": m.get("status"),
                           "seconds": m.get("total_seconds"),
                           "timed_out_internally": m.get("timed_out_internally"), **metrics(rows)})
    field_summary = [{"variant": v, "field": f, **metrics(rows)} for (v, f), rows in by_vf.items()]

    # Supplementary fields: not representable in NormalizedPlan; compare the
    # independent-CV layer's own measurements (recorded in meta) against legacy expected.json.
    supp = []
    for p in sorted(truths):
        exp_path = REPO / f"data/test_plans/{p}.expected.json"
        exp = json.loads(exp_path.read_text()) if exp_path.exists() else {}
        for fld in ("plot.net_area", "far.area", "building.gross_built_up_area"):
            if exp.get(fld) is None:
                continue
            for v in ("pdf_hybrid", "dxf_hybrid"):
                meta = runs.get(v, {}).get(p, (None, {}))[1]
                ms = [m for m in ((meta.get("independent_cv") or {}).get("measurements") or []) if m["field"] == fld]
                best = max(ms, key=lambda m: m["confidence"]) if ms else None
                val = None if best is None else (best["value_m"] if best["value_m"] is not None else best["value"])
                supp.append({"plan": p, "field": fld, "variant": v, "truth_legacy_expected_json": exp[fld],
                             "independent_cv_value": val,
                             "status": "MISSING" if val is None else ("CORRECT" if within_tolerance(val, exp[fld]) else "WRONG"),
                             "reaches_normalized_plan": False})

    def write(name, rows):
        if not rows:
            return
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(FRESH / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    write("field_level.csv", field_rows)
    write("plan_level.csv", plan_level)
    write("pipeline_comparison.csv", pipeline)
    write("field_summary.csv", field_summary)
    write("supplementary_fields.csv", supp)
    (FRESH / "extraction_eval.json").write_text(json.dumps({
        "tolerance": {"abs_m_or_unit": ABS_TOL, "rel": REL_TOL, "rule": "|pred-truth| <= max(ABS_TOL, REL_TOL*|truth|)"},
        "truth_tiers_scored": list(SCORABLE_DEFAULT),
        "pipeline_comparison": pipeline, "plan_level": plan_level, "field_summary": field_summary,
        "pdf_dxf_reconciliation_reports": recon_reports,
        "run_meta": {v: {p: {k: m.get(k) for k in ("status", "total_seconds", "extract_seconds", "timed_out_internally")}
                         for p, (_, m) in d.items()} for v, d in runs.items()},
    }, indent=1, default=str), encoding="utf-8")
    for r in pipeline:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
