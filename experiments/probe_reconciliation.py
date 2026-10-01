"""
Phase 6 behavioural probes: feed the ACTUAL reconciliation functions the five
canonical evidence situations (agreement, small disagreement, major conflict,
missing evidence, uncertain evidence) and record exactly what each returns.
No production code is modified. Output: experiments/results/fresh/reconciliation_probes.json/.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.schemas.enums import ConfidenceLevel
from backend.schemas.evidence import Confidence, ValueField
from backend.schemas.independent_measurements import IndependentMeasurement
from backend.schemas.normalized_plan import (BuildingSection, NormalizedPlan, PlotSection, RoadSection,
                                             SetbackSection)
from backend.schemas.units import UnitValue
from backend.spatial_reasoning.evidence_reconciliation import EvidenceCandidate, reconcile, to_value_field
from backend.spatial_reasoning.final_fusion import _value_field
from backend.spatial_reasoning.pdf_dxf_reconciliation import reconcile_pdf_dxf

SCENARIOS = [
    # name, source A, source B, conf A (0-1), conf B (0-1)
    ("agreement", 10.59, 10.59, 0.9, 0.9),
    ("small_disagreement", 10.59, 10.62, 0.9, 0.9),
    ("moderate_disagreement_0.40m", 10.59, 10.99, 0.9, 0.9),
    ("major_conflict", 10.59, 7.25, 0.9, 0.9),
    ("missing_evidence_single_source", 10.59, None, 0.9, None),
    ("uncertain_single_source_low_conf", 10.59, None, 0.3, None),
    ("uncertain_second_source_low_conf_conflicting", 10.59, 7.25, 0.9, 0.3),
]


def vf(value, level=ConfidenceLevel.MEDIUM):
    if value is None:
        return ValueField[float].missing("probe: no value")
    return ValueField[float](value=value, normalized_value=UnitValue(magnitude=value, unit="m"),
                             confidence=Confidence(level=level, reason="probe"), source="probe")


def plan_with_building_width(pid, value, level=ConfidenceLevel.MEDIUM):
    m = lambda: ValueField[float].missing("probe")
    return NormalizedPlan(
        plan_id=pid, source_document_id=pid,
        plot=PlotSection(width=m(), depth=m(), area=m()),
        building=BuildingSection(width=vf(value, level), depth=m(), footprint_area=m()),
        road=RoadSection(width=m()),
        setbacks=SetbackSection(front=m(), rear=m(), left=m(), right=m()),
        coverage=m(), far=m(),
    )


def level_for(conf):
    if conf is None:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW if conf < 0.5 else ConfidenceLevel.HIGH


rows = []
for name, a, b, ca, cb in SCENARIOS:
    # 1) Within-document multi-candidate reconciliation (legacy resolver path)
    cands = [EvidenceCandidate(value=a, source="pdf_text", weight=level_for(ca))]
    if b is not None:
        cands.append(EvidenceCandidate(value=b, source="cv_geometry", weight=level_for(cb)))
    out = reconcile(cands)
    if isinstance(out, tuple):  # CONFLICTING_EVIDENCE returns (outcome, Conflict)
        out = out[0]
    field = to_value_field(cands, field_label="building.width")
    rows.append({"layer": "evidence_reconciliation (legacy, within-PDF)", "scenario": name, "a": a, "b": b,
                 "conf_a": ca, "conf_b": cb, "status": out.status.value, "value": out.value,
                 "confidence": field.confidence.level.value, "flagged": field.conflict is not None,
                 "note": out.note})

    # 2) Final CV-vs-Vision agreement layer (the shipping layer for PDF)
    cv_m = IndependentMeasurement(field="building.width", value_m=a, value=a, unit="m", confidence=ca)
    vision = None if b is None else (b, cb, "probe", 1, None)
    f = _value_field(cv_m, vision, "building.width", "m")
    status = ("AGREED" if f.source == "FINAL:CV+VISION_AGREED" else "CV_ONLY" if f.source == "FINAL:CV_ONLY"
              else "VISION_ONLY" if f.source == "FINAL:VISION_ONLY" else ("CONFLICT" if f.conflict else "MISSING"))
    rows.append({"layer": "final_fusion (CV vs Vision, shipping)", "scenario": name, "a": a, "b": b,
                 "conf_a": ca, "conf_b": cb, "status": status, "value": f.value,
                 "confidence": f.confidence.level.value, "flagged": f.conflict is not None,
                 "note": f.confidence.reason if f.confidence else ""})

    # 3) PDF <-> DXF post-hoc reconciliation (dual upload)
    pdf_plan = plan_with_building_width("pdf", a, level_for(ca))
    dxf_plan = plan_with_building_width("dxf", b, level_for(cb) if b is not None else ConfidenceLevel.MEDIUM)
    rec, report = reconcile_pdf_dxf(pdf_plan, dxf_plan)
    cmp = next(c for c in report.fields if c.field == "building.width")
    shipped = rec.building.width
    rows.append({"layer": "pdf_dxf_reconciliation (dual upload)", "scenario": name, "a": a, "b": b,
                 "conf_a": ca, "conf_b": cb, "status": cmp.status, "value": shipped.value,
                 "confidence": shipped.confidence.level.value, "flagged": shipped.conflict is not None,
                 "note": cmp.note})

outdir = REPO / "experiments/results/fresh"
outdir.mkdir(parents=True, exist_ok=True)
(outdir / "reconciliation_probes.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
with open(outdir / "reconciliation_probes.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(f"{r['layer'][:26]:26s} {r['scenario']:45s} -> {r['status']:22s} value={r['value']} conf={r['confidence']} flagged={r['flagged']}")
