"""Per-plan extraction report across evidence sources (markdown).

Sources (all from saved runs; no backend change):
  PDF      : PDF pipeline, vision off (experiments/results/fresh_rerun/pdf_hybrid)
  CV       : independent CV layer of that PDF run, before fusion (native text + vector
             geometry + OCR + OpenCV raster evidence), from meta.json
  VLM      : vision-only values -- the saved VLM output (30 Sep re-run) passed through
             final_fusion with an EMPTY CV result, i.e. exactly what fusion would ship
             from vision alone (grounding caps applied)
  FUSION   : PDF pipeline with the VLM on (CV + VLM through final_fusion), 30 Sep re-run
  DXF      : DXF pipeline (experiments/results/fresh_rerun/dxf_hybrid)
  PDF+DXF  : pdf_dxf_reconciliation of the PDF and DXF plans
"""
import json, sys
from pathlib import Path
from collections import Counter, defaultdict
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from backend.corpus.runner import plan_to_predictions
from backend.corpus.store import load_manifest, load_truth
from backend.corpus.scoring import within_tolerance, score_plan
from backend.schemas.normalized_plan import NormalizedPlan
from backend.schemas.vision import VisionDocumentResult
from backend.schemas.independent_measurements import IndependentCVResult
from backend.schemas.enums import confidence_level_from_score
from backend.spatial_reasoning.final_fusion import build_final_agreement
from backend.spatial_reasoning.pdf_dxf_reconciliation import reconcile_pdf_dxf

R = REPO / "experiments/results"
PDF = R / "fresh_rerun/pdf_hybrid"; DXF = R / "fresh_rerun/dxf_hybrid"
VLMD = R / "vlm_rerun_2026-09-30/pdf_hybrid_vlm_api"
FIELDS = ["plot.width", "plot.depth", "plot.area", "building.width", "building.depth", "building.footprint_area",
          "building.floor_count", "road.width", "setbacks.front", "setbacks.rear", "setbacks.left", "setbacks.right",
          "coverage", "far", "building_use"]
UNIT = {"plot.area": "m²", "building.footprint_area": "m²", "coverage": "%", "far": "", "building.floor_count": "", "building_use": ""}
SRC = ["CV", "PDF", "VLM", "FUSION", "DXF", "PDF+DXF"]
LV = {"HIGH": "H", "MEDIUM": "M", "LOW": "L", "CONFLICTING": "C", "MISSING": "–"}


def load_plan(p):
    return NormalizedPlan.model_validate_json(p.read_text()) if p.exists() else None


def preds_of(plan):
    return plan_to_predictions(plan) if plan is not None else {}


def cv_preds(meta):
    out = {}
    icv = (meta or {}).get("independent_cv") or {}
    for m in icv.get("measurements") or []:
        v = m.get("value_m") if m.get("value_m") is not None else m.get("value")
        if v is None or m["field"] in out: continue
        out[m["field"]] = {"value": v, "confidence": confidence_level_from_score(m.get("confidence") or 0).value}
    return out


def vlm_preds(pid):
    f = VLMD / f"{pid}.vision.json"
    if not f.exists(): return {}
    vis = VisionDocumentResult.model_validate_json(f.read_text())
    fa = build_final_agreement(IndependentCVResult(document_id=pid, status="NOT_RUN"), vis)
    return {k: {"value": v["value"], "confidence": v["confidence"]} for k, v in fa["values"].items() if v["value"] is not None}


def status(field, pred, tf):
    """✓ correct, ✗ wrong, ⊘ answered a must-abstain field, · no truth / not scored"""
    if tf is None: return "·"
    if tf.verification == "must_abstain": return "⊘" if pred is not None else "✓abst"
    if tf.verification not in ("printed", "derived", "inspected"): return "·"
    if pred is None: return "miss"
    tv = tf.value
    if isinstance(tv, (int, float)) and isinstance(pred, (int, float)):
        return "✓" if within_tolerance(float(pred), float(tv)) else "✗"
    return "✓" if str(pred).lower() == str(tv).lower() else "✗"


def status2(sc, tf, v):
    if tf is not None and tf.verification == "must_abstain":
        return "⊘" if v is not None else "✓abst"
    if sc is None: return "·"
    return {"CORRECT": "✓", "WRONG": "✗", "MISSING": "miss"}.get(sc.status, sc.status) + ("↔" if getattr(sc, "axis_swapped", False) else "")


def fmt(v, field):
    if v is None: return "—"
    if field == "building.floor_count" and isinstance(v, (int, float)):
        return str(int(round(v)))
    if isinstance(v, float):
        return f"{v:.2f}" if field not in ("far",) else f"{v:.3f}"
    return str(v)


def main():
    man = load_manifest(); lines = []; tally = {s: Counter() for s in SRC}
    lines += ["# BUILDCheck India — extraction results by evidence source", "",
              "All 8 corpus plans (the corpus has no PLAN3), PDF and DXF, code snapshot 699fb74.",
              "",
              "| Column | What it is | Run |",
              "|---|---|---|",
              "| **CV** | Independent CV layer of the PDF path before fusion: native text, vector geometry, OCR and OpenCV raster evidence through the site-plan resolver | fresh re-run |",
              "| **PDF** | Final PDF pipeline output, vision off (CV + pipeline reasoning such as floor count, building use, derived coverage/FAR) | fresh re-run |",
              "| **VLM** | Vision only: saved Qwen3.8-27B output passed through `final_fusion` with no CV, i.e. what vision alone would ship (grounding caps applied) | 30 Sep 2026 VLM run |",
              "| **FUSION** | PDF pipeline with vision on: CV and VLM reconciled by `final_fusion` | 30 Sep 2026 VLM run |",
              "| **DXF** | DXF pipeline alone | fresh re-run |",
              "| **PDF+DXF** | `pdf_dxf_reconciliation` of the PDF and DXF plans (PDF value ships; DXF disagreement only adds a flag, shown as ⚑) | fresh re-run |",
              "",
              "Cell format: `value (level) mark`. Levels: H high, M medium, L low; `withheld (C)` = conflicting evidence, no value shipped; — = no value.",
              "Marks against the re-verified truth, tolerance max(0.15, 5 %): ✓ correct, ✗ wrong, miss = no answer, "
              "⊘ answered a must-abstain field, ✓abst = correctly abstained, ↔ correct as an unordered width/depth pair (axes swapped), · truth unverified/absent (not scored).",
              "Units: lengths m, areas m², coverage %.", ""]
    hdr = len(lines)
    for e in man.plans:
        pid = e.id; truth = load_truth(e)
        pplan = load_plan(PDF / f"{pid}.plan.json"); dplan = load_plan(DXF / f"{pid}.plan.json")
        pmeta = json.loads((PDF / f"{pid}.meta.json").read_text()) if (PDF / f"{pid}.meta.json").exists() else {}
        fplan = load_plan(VLMD / f"{pid}.plan.json")
        rplan = None
        if pplan is not None and dplan is not None:
            rplan, rep = reconcile_pdf_dxf(pplan, dplan)
        P = {"CV": cv_preds(pmeta), "PDF": preds_of(pplan), "VLM": vlm_preds(pid), "FUSION": preds_of(fplan),
             "DXF": preds_of(dplan), "PDF+DXF": preds_of(rplan)}
        SC = {src: {sc.field: sc for sc in score_plan(truth, {k: v for k, v in P[src].items()})} for src in SRC}
        sheet = "S" + str(["PLAN1", "PLAN2", "PLAN4", "PLAN5", "PLAN6", "PLAN7", "PLAN8", "PLAN9"].index(pid) + 1)
        lines += [f"## {pid} (paper sheet {sheet})", "",
                  "| Field | Truth (tier) | " + " | ".join(SRC) + " |",
                  "|---|---|" + "---|" * len(SRC)]
        for f in FIELDS:
            tf = truth.fields.get(f)
            tcell = "—" if tf is None or tf.value is None else f"{fmt(tf.value, f)} ({tf.verification})"
            if tf is not None and tf.verification == "must_abstain": tcell = "must abstain"
            row = [f"`{f}`", tcell]
            for s in SRC:
                p = P[s].get(f) or {}
                v = p.get("value"); lvl = LV.get(p.get("confidence"), p.get("confidence") or "")
                if v is None and p.get("confidence") == "CONFLICTING": lvl = "C"
                st = status2(SC[s].get(f), tf, v)
                flag = " ⚑" if s == "PDF+DXF" and p.get("flagged") else ""
                if v is None and lvl == "C":
                    cell = "withheld (C)"; st = "withheld" if st == "miss" else st
                    tally[s][st] += 1
                    row.append(cell if st == "withheld" else f"{cell} {st}")
                    continue
                elif v is None:
                    cell = "—"
                else:
                    cell = f"{fmt(v, f)} ({lvl}){flag}"
                tally[s][st] += 1
                row.append(f"{cell} {st}" if st != "·" else cell)
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    # summary
    summ = ["## Summary over all plans (84 scored truth values, 10 must-abstain fields)", "",
            "| Source | Correct | Wrong | Missing | Withheld (conflict) | Answered must-abstain | Correctly abstained |",
            "|---|---|---|---|---|---|---|"]
    for s in SRC:
        t = Counter({k.rstrip('↔'): 0 for k in tally[s]})
        for k, n in tally[s].items(): t[k.rstrip('↔')] += n
        summ.append(f"| {s} | {t['✓']} | {t['✗']} | {t['miss']} | {t['withheld']} | {t['⊘']} | {t['✓abst']} |")
    summ += ["", "Notes:", "- CV and VLM are intermediate layers; they are not what the system ships. PDF, FUSION, DXF and PDF+DXF are shipped outputs.",
             "- VLM covers only the fields `final_fusion` takes from vision (lengths, areas, coverage, FAR); floor count and building use come from the pipeline, so they are blank in that column.",
             "- CV does not produce floor count, building use or derived coverage/FAR on its own; the PDF pipeline adds these.",
             "- Level shown for PDF+DXF is the PDF value's level after reconciliation (agreement can promote it to H; disagreement leaves it unchanged and adds ⚑).",
             "",
             "Run notes (30 Sep 2026):",
             "- The PDF and DXF pipelines were re-run today for all 8 plans (VLM off, default settings, fresh processes).",
             "- **PDF:** identical to the paper's run on all 84 values (status, confidence and value).",
             "- **DXF:** 6 correct / 46 wrong today versus 5 / 47 in the paper. On PLAN6 (S5) the DXF plot outline changed between runs (plot width 16.96 m → 12.78 m, area 78.05 → 58.80 m²). The width now scores correct only as an unordered width/depth pair (12.78 m is within tolerance of the 13.10 m depth). The DXF reconstruction is therefore not fully deterministic. All DXF values, including the new one, remain LOW, so rule-level containment is unchanged.",
             "- **VLM and FUSION:** taken from the 30 Sep VLM re-run, which reproduced the paper's VLM run on all 84 values. No new API calls were made for this report.",
             ""]
    out = lines[:hdr] + summ + lines[hdr:]
    (REPO / "EXTRACTION_RESULTS.md").write_text("\n".join(out))
    print("\n".join(summ))


if __name__ == "__main__":
    main()
