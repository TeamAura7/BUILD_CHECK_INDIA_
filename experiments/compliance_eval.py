"""
Phase 7: downstream compliance error propagation.

For every plan:
  GT arm        a NormalizedPlan built ONLY from corpus truth values of the
                default scorable tiers (printed/derived/inspected/legacy),
                every value at HIGH confidence; everything else MISSING.
  predicted arm the NormalizedPlan a pipeline variant actually produced.

Context fields that the app treats as reviewer-supplied (never extracted;
see backend/app/routes/analyze.py _EDITABLE_CONTEXT_FIELDS) are held
IDENTICAL in both arms, so only measurement differences propagate:
  building_use     = "residential"   (control assumption, both arms)
  development_area = A, B and C      (each run separately: sensitivity)

Both arms are evaluated by the unmodified JsonFileRuleEngine against
data/runtime_rules/BBMP/rules.json (ACTIVE rules only).

Decision units:
  rule level        (plan, rule_id)
  requirement level (plan, rule.target): conservative rollup over that
                    target's rules using ComplianceResult.overall_status
                    semantics (any FAIL dominates; NOT_APPLICABLE ignored).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.compliance.engine import JsonFileRuleEngine
from backend.corpus.schema import SCORABLE_DEFAULT
from backend.corpus.store import load_manifest, load_truth
from backend.schemas.compliance import ComplianceResult
from backend.schemas.enums import ConfidenceLevel
from backend.schemas.evidence import Confidence, ValueField
from backend.schemas.normalized_plan import (BuildingSection, NormalizedPlan, PlotSection, RoadSection,
                                             SetbackSection)
from backend.schemas.units import UnitValue

FRESH = REPO / "experiments/results/fresh"
VARIANT_DIR = FRESH / "variants"
UNITS = {"plot.width": "m", "plot.depth": "m", "plot.area": "m2", "building.width": "m", "building.depth": "m",
         "building.footprint_area": "m2", "road.width": "m", "setbacks.front": "m", "setbacks.rear": "m",
         "setbacks.left": "m", "setbacks.right": "m", "coverage": "%", "far": "ratio"}
DECIDED = {"PASS", "FAIL"}
ABSTAIN = {"REQUIRES_REVIEW", "INSUFFICIENT_DATA", "CONFLICTING_EVIDENCE"}


def gt_plan(plan_id: str, truth) -> NormalizedPlan:
    vals = {k: tf.value for k, tf in truth.fields.items()
            if tf.verification in SCORABLE_DEFAULT and tf.value is not None}

    def f(name):
        v = vals.get(name)
        if v is None:
            return ValueField[float].missing("no scorable ground truth")
        return ValueField[float](value=float(v), normalized_value=UnitValue(magnitude=float(v), unit=UNITS[name]),
                                 confidence=Confidence(level=ConfidenceLevel.HIGH, reason="ground truth"),
                                 source="ground_truth")
    fc = vals.get("building.floor_count")
    floor = None if fc is None else ValueField[int](
        value=int(fc), confidence=Confidence(level=ConfidenceLevel.HIGH, reason="ground truth"), source="ground_truth")
    return NormalizedPlan(
        plan_id=f"{plan_id}-GT", source_document_id=plan_id,
        plot=PlotSection(width=f("plot.width"), depth=f("plot.depth"), area=f("plot.area")),
        building=BuildingSection(width=f("building.width"), depth=f("building.depth"),
                                 footprint_area=f("building.footprint_area"), floor_count=floor),
        road=RoadSection(width=f("road.width")),
        setbacks=SetbackSection(front=f("setbacks.front"), rear=f("setbacks.rear"),
                                left=f("setbacks.left"), right=f("setbacks.right")),
        coverage=f("coverage"), far=f("far"),
    )


def with_context(plan: NormalizedPlan, dev_area: str) -> NormalizedPlan:
    p = plan.model_copy(deep=True)
    p.building_use = ValueField[str](value="residential", confidence=Confidence(
        level=ConfidenceLevel.HIGH, reason="experimental control (both arms)"), source="experiment_context")
    p.development_area = ValueField[str](value=dev_area, confidence=Confidence(
        level=ConfidenceLevel.HIGH, reason="experimental control (both arms)"), source="experiment_context")
    return p


def rollup(statuses):
    s = set(statuses)
    if "FAIL" in s:
        return "FAIL"
    for u in ("CONFLICTING_EVIDENCE", "INSUFFICIENT_DATA", "REQUIRES_REVIEW"):
        if u in s:
            return u
    if "PASS" in s:
        return "PASS"
    return "NOT_APPLICABLE"


def load_variant_plans():
    from backend.spatial_reasoning.pdf_dxf_reconciliation import reconcile_pdf_dxf
    out = defaultdict(dict)
    for vdir in sorted(VARIANT_DIR.iterdir()):
        for meta_path in sorted(vdir.glob("*.meta.json")):
            pid = meta_path.name.split(".")[0]
            pj = vdir / f"{pid}.plan.json"
            out[vdir.name][pid] = (NormalizedPlan.model_validate_json(pj.read_text(encoding="utf-8"))
                                   if pj.exists() else None)
    for pid in out.get("pdf_hybrid", {}):
        a, b = out["pdf_hybrid"].get(pid), out.get("dxf_hybrid", {}).get(pid)
        if a is not None and b is not None:
            out["pdf_dxf_reconciled"][pid] = reconcile_pdf_dxf(a, b)[0]
        elif pid in out.get("dxf_hybrid", {}):
            out["pdf_dxf_reconciled"][pid] = None
    # Counterfactual plans (analysis-only manipulations of the SAVED plans; no code change):
    #   <v>__ungated         every LOW confidence raised to MEDIUM -> what the engine would decide
    #                        if the LOW -> REQUIRES_REVIEW gate did not exist
    #   pdf_dxf_reconciled__flag_as_conflict  a field carrying a PDF<->DXF `.conflict` flag gets
    #                        level CONFLICTING -> what happens if the cross-check flag propagated
    def ungate(plan):
        p = plan.model_copy(deep=True)
        for sec in ("plot", "building", "road", "setbacks"):
            obj = getattr(p, sec)
            for name, vf in list(obj.__dict__.items()):
                if isinstance(vf, ValueField) and vf.confidence and vf.confidence.level == ConfidenceLevel.LOW:
                    setattr(obj, name, vf.model_copy(update={"confidence": vf.confidence.model_copy(update={"level": ConfidenceLevel.MEDIUM})}))
        for name in ("coverage", "far"):
            vf = getattr(p, name)
            if vf.confidence and vf.confidence.level == ConfidenceLevel.LOW:
                setattr(p, name, vf.model_copy(update={"confidence": vf.confidence.model_copy(update={"level": ConfidenceLevel.MEDIUM})}))
        return p

    def flag_as_conflict(plan):
        p = plan.model_copy(deep=True)
        for sec in ("plot", "building", "road", "setbacks"):
            obj = getattr(p, sec)
            for name, vf in list(obj.__dict__.items()):
                if isinstance(vf, ValueField) and vf.conflict is not None and vf.value is not None:
                    setattr(obj, name, vf.model_copy(update={"confidence": vf.confidence.model_copy(update={"level": ConfidenceLevel.CONFLICTING})}))
        for name in ("coverage", "far"):
            vf = getattr(p, name)
            if vf.conflict is not None and vf.value is not None:
                setattr(p, name, vf.model_copy(update={"confidence": vf.confidence.model_copy(update={"level": ConfidenceLevel.CONFLICTING})}))
        return p

    for v in ("pdf_hybrid", "dxf_hybrid", "dxf_evidence_decision", "pdf_evidence_decision"):
        if v in out:
            out[f"{v}__ungated"] = {pid: (None if pl is None else ungate(pl)) for pid, pl in out[v].items()}
    if "pdf_dxf_reconciled" in out:
        out["pdf_dxf_reconciled__flag_as_conflict"] = {pid: (None if pl is None else flag_as_conflict(pl))
                                                      for pid, pl in out["pdf_dxf_reconciled"].items()}
    return out


def main():
    engine = JsonFileRuleEngine()
    rules = {r.rule_id: r for r in engine.load_ruleset("BBMP")}
    manifest = load_manifest()
    truths = {e.id: load_truth(e) for e in manifest.plans}
    variants = load_variant_plans()
    rule_rows, req_rows = [], []
    for dev_area in ("A", "B", "C"):
        gt_results = {pid: engine.evaluate_plan(with_context(gt_plan(pid, t), dev_area), "BBMP")
                      for pid, t in truths.items()}
        for vname, plans in variants.items():
            for pid, plan in plans.items():
                gt_res = gt_results[pid]
                if plan is None:  # extraction failed/timed out: no plan -> every rule INSUFFICIENT_DATA
                    pred_status = {r.rule_id: "INSUFFICIENT_DATA" for r in gt_res.rule_results}
                    pred_obs = {}
                else:
                    pr = engine.evaluate_plan(with_context(plan, dev_area), "BBMP")
                    pred_status = {r.rule_id: r.status.value for r in pr.rule_results}
                    pred_obs = {r.rule_id: [(o.value, o.confidence.level.value, o.conflict is not None)
                                            for o in (r.observed_values or [])] for r in pr.rule_results}
                by_target = defaultdict(lambda: ([], []))
                for r in gt_res.rule_results:
                    g, p = r.status.value, pred_status[r.rule_id]
                    if g == "NOT_APPLICABLE" and p == "NOT_APPLICABLE":
                        continue
                    tgt = rules[r.rule_id].target
                    by_target[tgt][0].append(g)
                    by_target[tgt][1].append(p)
                    rule_rows.append({
                        "development_area": dev_area, "variant": vname, "plan": pid, "rule_id": r.rule_id,
                        "target": tgt, "gt_status": g, "pred_status": p,
                        "gt_observed": [o.value for o in (r.observed_values or [])],
                        "pred_observed": pred_obs.get(r.rule_id),
                        "requirement": r.required_value_description,
                    })
                for tgt, (gs, ps) in by_target.items():
                    req_rows.append({"development_area": dev_area, "variant": vname, "plan": pid, "target": tgt,
                                     "gt_status": rollup(gs), "pred_status": rollup(ps)})

    def outcome(g, p):
        if g in DECIDED:
            if p == g:
                return "CORRECT_DECISION"
            if p in DECIDED:
                return "FALSE_PASS" if p == "PASS" else "FALSE_FAIL"
            if p == "NOT_APPLICABLE":
                return "APPLICABILITY_FLIP"
            return f"ABSTAIN_{p}"
        if g == "NOT_APPLICABLE":
            return "SPURIOUS_APPLICABILITY_" + p if p != "NOT_APPLICABLE" else "BOTH_NA"
        return "GT_UNDECIDABLE"

    for r in rule_rows + req_rows:
        r["outcome"] = outcome(r["gt_status"], r["pred_status"])

    def summarize(rows, unit):
        agg = []
        keyf = lambda r: (r["development_area"], r["variant"])
        groups = defaultdict(list)
        for r in rows:
            groups[keyf(r)].append(r)
        for (da, v), rs in sorted(groups.items()):
            dec = [r for r in rs if r["gt_status"] in DECIDED]
            c = Counter(r["outcome"] for r in dec)
            answered = [r for r in dec if r["pred_status"] in DECIDED]
            agg.append({
                "unit": unit, "development_area": da, "variant": v,
                "gt_decidable": len(dec),
                "correct": c["CORRECT_DECISION"], "false_pass": c["FALSE_PASS"], "false_fail": c["FALSE_FAIL"],
                "review": c["ABSTAIN_REQUIRES_REVIEW"], "insufficient_data": c["ABSTAIN_INSUFFICIENT_DATA"],
                "conflict": c["ABSTAIN_CONFLICTING_EVIDENCE"], "applicability_flip": c["APPLICABILITY_FLIP"],
                "decision_accuracy_over_gt_decidable": (c["CORRECT_DECISION"] / len(dec)) if dec else None,
                "decision_accuracy_when_decided": (c["CORRECT_DECISION"] / len(answered)) if answered else None,
                "abstention_rate": (sum(1 for r in dec if r["pred_status"] in ABSTAIN) / len(dec)) if dec else None,
                "gt_undecidable": sum(1 for r in rs if r["gt_status"] in ABSTAIN),
                "spurious_applicability": sum(1 for r in rs if r["outcome"].startswith("SPURIOUS_APPLICABILITY")),
            })
        return agg

    summary = summarize(rule_rows, "rule") + summarize(req_rows, "requirement")
    FRESH.mkdir(parents=True, exist_ok=True)
    for name, rows in (("compliance_rule_level.csv", rule_rows), ("compliance_requirement_level.csv", req_rows),
                       ("compliance_summary.csv", summary)):
        with open(FRESH / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    (FRESH / "compliance_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for s in summary:
        if s["development_area"] == "A":
            print(s)


if __name__ == "__main__":
    main()
