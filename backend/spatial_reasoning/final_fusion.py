"""Final CV + Vision agreement layer.

This module is intentionally independent from the legacy global candidate resolver.
The final value is produced only from the independent CV/native/OCR site-plan path
and the independent Vision semantic path. A conflict is never silently resolved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from backend.schemas.evidence import Confidence, Conflict, ValueField
from backend.schemas.enums import ConfidenceLevel, cap_confidence_level, confidence_level_from_score
from backend.schemas.independent_measurements import IndependentCVResult, IndependentMeasurement
from backend.schemas.vision import VisionDocumentResult
from backend.schemas.units import CanonicalUnit, UnitValue
from backend.spatial_reasoning import geometry_utils as geo
from backend.spatial_reasoning.areas import coverage_field, far_field
from backend.spatial_reasoning.setbacks import flag_if_setback_implausible

# A directly-extracted area value (from a printed "AREA OF PLOT" style label,
# or a Vision-reported area) is trusted as-is only while it roughly agrees
# with width x depth of the SAME final, independently-agreed dimensions. Once
# width and depth are individually confirmed by the CV+Vision agreement layer,
# a lingering area value computed earlier from different (pre-agreement,
# potentially wrong) width/depth candidates is no longer trustworthy -- this
# is precisely the "17.59 x 9.21 => 15.87 m2" inconsistency this module exists
# to catch and correct (see FINAL_CV_VISION_FUSION_CHANGES.md / the plot-area
# task notes). Below this tolerance the direct value is kept (it may carry
# useful provenance, e.g. a printed "SITE AREA" label); above it we trust the
# geometry over a stale/mismatched area reading.
_AREA_CONFLICT_TOLERANCE = 0.08

# Confidence levels emitted by `vf()` below that indicate the underlying
# width/depth value is an independently-resolved final value (as opposed to
# missing or in outright conflict) and can therefore be used to derive area.
_RELIABLE_STATUSES = {"AGREED", "CV_ONLY", "VISION_ONLY"}
from backend.spatial_reasoning import document_evidence as doc_ev

LENGTH_FIELDS = {
    "plot.width": "PLOT_WIDTH",
    "plot.depth": "PLOT_DEPTH",
    "building.width": "BUILDING_WIDTH",
    "building.depth": "BUILDING_DEPTH",
    "road.width": "ROAD_WIDTH",
    "setbacks.front": "FRONT_SETBACK",
    "setbacks.rear": "REAR_SETBACK",
    "setbacks.left": "LEFT_SETBACK",
    "setbacks.right": "RIGHT_SETBACK",
    "building.height_estimated": "BUILDING_HEIGHT",
    "building.height": "BUILDING_HEIGHT",
    "floor.height": "FLOOR_HEIGHT",
    "plinth.height": "PLINTH_HEIGHT",
    "parapet.height": "PARAPET_HEIGHT",
}

NUMERIC_FIELDS = {
    "plot.area": ("PLOT_AREA", "NET_PLOT_AREA"),
    "plot.net_area": ("NET_PLOT_AREA",),
    "building.plinth_area": ("BUILDING_FOOTPRINT_AREA", "PROPOSED_COVERAGE_AREA", "PLINTH_AREA"),
    "building.footprint_area": ("BUILDING_FOOTPRINT_AREA", "PROPOSED_COVERAGE_AREA", "PLINTH_AREA"),
    "coverage": ("COVERAGE_PERCENT",),
    "far.area": ("FAR_AREA",),
    "far": ("FAR_RATIO",),
    "building.gross_built_up_area": ("TOTAL_BUILT_UP_AREA", "BUILT_UP_AREA"),
}


def _cv_map(cv: IndependentCVResult) -> dict[str, IndependentMeasurement]:
    out: dict[str, IndependentMeasurement] = {}
    for m in cv.measurements:
        value = m.value_m if m.value_m is not None else m.value
        if value is None:
            continue
        cur = out.get(m.field)
        if cur is None or m.confidence > cur.confidence:
            out[m.field] = m
    return out


def _vision_length(result: VisionDocumentResult, semantic: str) -> tuple[float, float, str, int, Optional[bool]] | None:
    candidates = []
    for page in result.pages:
        for d in page.dimensions:
            if d.type.upper() != semantic.upper() or d.value is None:
                continue
            unit = (d.unit or "m").lower()
            factor = {"m": 1.0, "meter": 1.0, "metre": 1.0, "mm": .001, "cm": .01, "ft": .3048, "feet": .3048}.get(unit)
            if factor is None:
                continue
            candidates.append((float(d.value) * factor, float(d.confidence), d.evidence or "", page.page_number, d.grounded))
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: (-x[1], x[3]))[0]


def _vision_numeric(result: VisionDocumentResult | None, semantics: tuple[str, ...]) -> tuple[float, float, str, int, str, Optional[bool]] | None:
    if result is None:
        return None
    wanted = {s.upper() for s in semantics}
    candidates = []
    for page in result.pages:
        for a in page.areas:
            if a.type.upper() not in wanted or a.value is None:
                continue
            unit = (a.unit or "m2").lower().replace("²", "2")
            if unit in {"sqm", "sq.m", "square_metre", "square_meter"}:
                unit = "m2"
            if unit in {"%", "percent", "percentage"}:
                unit = "%"
            if unit == "ratio":
                pass
            elif unit == "ft2":
                unit = "m2"
                value = float(a.value) * 0.09290304
                candidates.append((value, float(a.confidence), a.evidence or "", page.page_number, unit, a.grounded))
                continue
            elif unit != "m2" and unit != "%":
                continue
            candidates.append((float(a.value), float(a.confidence), a.evidence or "", page.page_number, unit, a.grounded))
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: (-x[1], x[3]))[0]


def _tol(a: float, b: float, unit: str = "m") -> float:
    if unit == "%":
        return max(0.20, 0.01 * max(abs(a), abs(b)))
    if unit == "ratio":
        return max(0.02, 0.01 * max(abs(a), abs(b)))
    return max(0.05, 0.01 * max(abs(a), abs(b)))


def _value_field(cv_m: IndependentMeasurement | None, vision, field: str, unit: str = "m") -> ValueField[float]:
    cvv = None if cv_m is None else (cv_m.value_m if cv_m.value_m is not None else cv_m.value)
    vv = None if vision is None else vision[0]
    if cvv is None and vv is None:
        return ValueField[float].missing(f"No independent CV or Vision evidence for {field}.")
    if cvv is None:
        # `vision` is (value, confidence, evidence, page[, unit], grounded)
        # -- see `_vision_length`/`_vision_numeric`. A single-source (no
        # cross-validation) value must reflect how much ITS OWN source
        # trusts it, not a blanket MEDIUM regardless -- see the CV-only
        # branch below for why this was a real, confirmed bug, not a style
        # choice.
        vision_confidence = vision[1] if len(vision) > 1 and vision[1] is not None else None
        level = confidence_level_from_score(vision_confidence) if vision_confidence is not None else ConfidenceLevel.MEDIUM
        reason = (
            f"Vision-only; CV did not resolve this field. (Vision confidence: {vision_confidence:.2f})"
            if vision_confidence is not None else "Vision-only; CV did not resolve this field."
        )
        # `grounded` (the last tuple element) is False when grounding ran
        # but this value's presence could not be confirmed anywhere in the
        # PDF's native text layer (e.g. a scanned page with no text layer
        # at all). CV did not resolve this field either, so there is
        # nothing else to cross-check against -- the model's own
        # self-reported score alone must not be laundered into HIGH/MEDIUM
        # here (audit finding: `_value_field`'s Vision-only branch is the
        # actual shipping location `apply_final_agreement_to_plan` uses for
        # every LENGTH_FIELDS entry, including setbacks -- confirmed live
        # to override `pipeline.py`'s own, already-grounding-aware
        # `vision_setbacks`/`build_vision_only_plan` computation for this
        # exact reason). `grounded is True` or `None` (grounding did not
        # run, e.g. an older cached vision_result) leave this unchanged.
        grounded = vision[-1] if len(vision) > 4 and isinstance(vision[-1], bool) else None
        if grounded is False:
            level = cap_confidence_level(level, ConfidenceLevel.LOW)
            reason = f"{reason} Not independently grounded: no native text layer to verify this value against."
        return ValueField[float](value=round(vv, 4), normalized_value=UnitValue(magnitude=round(vv,4), unit=unit),
            confidence=Confidence(level=level, reason=reason),
            source="FINAL:VISION_ONLY")
    if vv is None:
        canonical = "m" if unit == "m" else unit
        # This is the actual bug this session's DXF confidence-cap work
        # (`_CAPTION_OVERRIDE_CONFIDENCE_CAP`, item 8/9 in DXF_FAILURE_
        # TAXONOMY.md) was found to be silently defeated by: a CV-only
        # value was ALWAYS tagged MEDIUM here, completely independent of
        # `cv_m.confidence` -- a DXF measurement the extractor itself
        # capped at 0.4 (well below LOW) reached `NormalizedPlan` (and
        # from there, `backend/compliance/engine.py`'s own REQUIRES_REVIEW
        # gate, which only fires on ConfidenceLevel.LOW) tagged MEDIUM,
        # identical to an ordinary, fully-trusted reading. Confirmed
        # directly on real PLAN5/PLAN6 DXF extractions before this fix.
        level = confidence_level_from_score(cv_m.confidence) if cv_m is not None else ConfidenceLevel.MEDIUM
        return ValueField[float](value=round(cvv,4), normalized_value=UnitValue(magnitude=round(cvv,4), unit=canonical),
            confidence=Confidence(
                level=level,
                reason=(
                    f"CV-only; Vision did not emit this field. (CV confidence: {cv_m.confidence:.2f})"
                    if cv_m is not None else "CV-only; Vision did not emit this field."
                ),
            ),
            source="FINAL:CV_ONLY")
    diff = abs(cvv - vv)
    tol = _tol(cvv, vv, unit)
    if diff > tol:
        conflict = Conflict(
            description=f"CV={cvv:.4f} and Vision={vv:.4f} disagree for {field}; difference {diff:.4f} exceeds tolerance {tol:.4f}.",
            conflicting_raw_values=[UnitValue(magnitude=float(cvv), unit=unit), UnitValue(magnitude=float(vv), unit=unit)],
            conflicting_sources=["independent_cv", "vision"],
        )
        return ValueField[float].conflicting(conflict)
    agreed = (cvv + vv) / 2.0
    return ValueField[float](value=round(agreed,4), normalized_value=UnitValue(magnitude=round(agreed,4), unit=unit),
        confidence=Confidence(level=ConfidenceLevel.HIGH, reason=f"Independent CV and Vision agree within {tol:.3f} {unit}."),
        source="FINAL:CV+VISION_AGREED")


def build_final_agreement(cv: IndependentCVResult, vision: VisionDocumentResult) -> dict[str, Any]:
    cm = _cv_map(cv)
    values: dict[str, Any] = {}
    counts = {"AGREED": 0, "CV_ONLY": 0, "VISION_ONLY": 0, "CONFLICT": 0, "MISSING": 0}
    for field, semantic in LENGTH_FIELDS.items():
        cvm = cm.get(field)
        cv_value = None if cvm is None else (cvm.value_m if cvm.value_m is not None else cvm.value)
        vision_res = _vision_length(vision, semantic)
        vf = _value_field(cvm, vision_res, field, "m")
        status = "AGREED" if vf.source == "FINAL:CV+VISION_AGREED" else ("CV_ONLY" if vf.source == "FINAL:CV_ONLY" else ("VISION_ONLY" if vf.source == "FINAL:VISION_ONLY" else ("CONFLICT" if vf.conflict else "MISSING")))
        counts[status] += 1
        values[field] = {
            "value": vf.value, "unit": "m", "status": status, "source": vf.source,
            "confidence": vf.confidence.level.value, "reason": vf.confidence.reason,
            # Raw per-source values, preserved even on CONFLICT/MISSING so a
            # caller can build a fully structured Conflict instead of only
            # a prose reason string -- see apply_final_agreement_to_plan.
            "cv_value": cv_value, "vision_value": (vision_res[0] if vision_res else None),
        }

    for field, semantics in NUMERIC_FIELDS.items():
        # Skip duplicate alias; the final output keeps both explicit aliases if CV has either.
        vision_ev = _vision_numeric(vision, semantics)
        cvm = cm.get(field)
        cv_value = None if cvm is None else (cvm.value_m if cvm.value_m is not None else cvm.value)
        unit = (vision_ev[4] if vision_ev else (cvm.unit if cvm and cvm.unit else "m2"))
        vf = _value_field(cvm, vision_ev, field, unit)
        status = "AGREED" if vf.source == "FINAL:CV+VISION_AGREED" else ("CV_ONLY" if vf.source == "FINAL:CV_ONLY" else ("VISION_ONLY" if vf.source == "FINAL:VISION_ONLY" else ("CONFLICT" if vf.conflict else "MISSING")))
        counts[status] += 1
        values[field] = {
            "value": vf.value, "unit": unit, "status": status, "source": vf.source,
            "confidence": vf.confidence.level.value, "reason": vf.confidence.reason,
            "cv_value": cv_value, "vision_value": (vision_ev[0] if vision_ev else None),
        }

    return {"document_id": cv.document_id, "values": values, "summary": counts, "has_conflicts": counts["CONFLICT"] > 0}


def build_document_verified_fusion(
    cv: Optional[IndependentCVResult],
    vision: Optional[VisionDocumentResult],
    pdf_path: Optional[str] = None,
) -> dict[str, Any]:
    """PHASEE3NEW.md Validation / Fusion Engine.

    This is the deterministic third layer sitting on top of the two
    independent extraction pipelines. It NEVER re-runs CV or Vision
    extraction and never lets one pipeline's output influence the
    other's -- it only reconciles the evidence they already produced
    against the document itself (native text / OCR / geometry), per the
    conflict-resolution hierarchy in sections 14-18.

    Returns the full schema described in section 30: cv_values,
    vision_values, document_evidence, conflicts, validation,
    final_agreed_values, plus provenance for every field.
    """
    text_index = doc_ev.build_document_text_index(pdf_path)

    cm = _cv_map(cv) if cv is not None else {}
    all_measurements = list(cv.measurements) if cv is not None else []

    cv_values: dict[str, Any] = {}
    vision_values: dict[str, Any] = {}
    document_evidence: dict[str, Any] = {}
    conflicts: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    final_agreed_values: dict[str, Any] = {}
    reported_calculated: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    counts: dict[str, int] = {}

    def _record(field: str, cv_value, vision_value, unit: str):
        field_measurements = [m for m in all_measurements if m.field == field]
        verdict = doc_ev.resolve_field(field, cv_value, vision_value, unit, field_measurements, text_index)

        cv_values[field] = {
            "value": cv_value,
            "unit": unit,
            "status": doc_ev.FOUND if cv_value is not None else doc_ev.NOT_FOUND_BY_CV,
        }
        vision_values[field] = {
            "value": vision_value,
            "unit": unit,
            "status": doc_ev.FOUND if vision_value is not None else doc_ev.NOT_FOUND_BY_VISION,
        }
        document_evidence[field] = {
            "cv_candidate": verdict.cv_evidence.to_dict() if verdict.cv_evidence else None,
            "vision_candidate": verdict.vision_evidence.to_dict() if verdict.vision_evidence else None,
            "text_index_available": text_index.available,
        }
        if verdict.status in (doc_ev.CONFLICT_RESOLVED_BY_DOCUMENT_EVIDENCE, doc_ev.UNRESOLVED_CONFLICT):
            conflicts[field] = {
                "cv": cv_value, "vision": vision_value,
                "winner": verdict.winner, "reason": verdict.reason,
            }
        validation[field] = {"status": verdict.status, "reason": verdict.reason, "winner": verdict.winner}
        final_agreed_values[field] = {"final_value": verdict.final_value, "unit": unit, "status": verdict.status}

        rvc = doc_ev.reported_vs_calculated(field_measurements)
        if rvc is not None:
            reported_calculated[field] = {
                "reported_value": rvc.reported_value,
                "calculated_value": rvc.calculated_value,
                "final_value": rvc.final_value,
                "status": rvc.status,
            }
            # A reported document value always wins over a purely derived one,
            # per section 19 -- never silently overwritten.
            if rvc.reported_value is not None and verdict.status in (
                doc_ev.CV_ONLY_DOCUMENT_VERIFIED, doc_ev.CV_ONLY_UNVERIFIED,
                doc_ev.AGREED_DOCUMENT_VERIFIED, doc_ev.AGREED_UNVERIFIED,
            ):
                final_agreed_values[field]["final_value"] = rvc.reported_value

        provenance[field] = {
            "cv_source": [m.source for m in field_measurements if m.source],
            "vision_present": vision_value is not None,
            "document_text_index_available": text_index.available,
        }
        counts[verdict.status] = counts.get(verdict.status, 0) + 1

    for field in LENGTH_FIELDS:
        cvv = cm.get(field)
        cv_value = None if cvv is None else (cvv.value_m if cvv.value_m is not None else cvv.value)
        vres = _vision_length(vision, LENGTH_FIELDS[field]) if vision is not None else None
        vision_value = None if vres is None else vres[0]
        _record(field, cv_value, vision_value, "m")

    for field, semantics in NUMERIC_FIELDS.items():
        cvv = cm.get(field)
        cv_value = None if cvv is None else (cvv.value_m if cvv.value_m is not None else cvv.value)
        vres = _vision_numeric(vision, semantics) if vision is not None else None
        vision_value = None if vres is None else vres[0]
        unit = (vres[4] if vres else (cvv.unit if cvv and cvv.unit else "m2"))
        _record(field, cv_value, vision_value, unit)

    # Derived fallback for coverage/far: neither CV nor Vision has to have
    # printed an explicit coverage%/FAR figure for these to be computable --
    # the reconciled footprint_area/plot_area above are already enough (same
    # formula `apply_final_agreement_to_plan` uses for the production
    # pipeline; see `areas.coverage_field`/`areas.far_field`). A DIRECT
    # reading recorded above always wins; this only fills in an otherwise
    # honest MISSING when the ingredients are actually available. Floor
    # count is not resolved at this independent-CV/Vision layer, so `far`
    # is computed as the single-storey lower bound, matching `far_field`'s
    # own documented fallback semantics.
    footprint_final = final_agreed_values.get("building.footprint_area", {}).get("final_value")
    plot_area_final = final_agreed_values.get("plot.area", {}).get("final_value")
    if footprint_final is not None and plot_area_final:
        footprint_field = ValueField[float](value=footprint_final, confidence=Confidence(level=ConfidenceLevel.MEDIUM))
        plot_field = ValueField[float](value=plot_area_final, confidence=Confidence(level=ConfidenceLevel.MEDIUM))
        if final_agreed_values.get("coverage", {}).get("final_value") is None:
            computed = coverage_field(footprint_field, plot_field)
            if computed.value is not None:
                final_agreed_values["coverage"] = {
                    "final_value": round(computed.value, 4), "unit": "%",
                    "status": "DERIVED_FROM_FOOTPRINT_AND_PLOT_AREA",
                }
                counts["DERIVED_FROM_FOOTPRINT_AND_PLOT_AREA"] = counts.get("DERIVED_FROM_FOOTPRINT_AND_PLOT_AREA", 0) + 1
        if final_agreed_values.get("far", {}).get("final_value") is None:
            computed = far_field(footprint_field, plot_field, None)
            if computed.value is not None:
                final_agreed_values["far"] = {
                    "final_value": round(computed.value, 4), "unit": "ratio",
                    "status": "DERIVED_FROM_FOOTPRINT_AND_PLOT_AREA",
                }
                counts["DERIVED_FROM_FOOTPRINT_AND_PLOT_AREA"] = counts.get("DERIVED_FROM_FOOTPRINT_AND_PLOT_AREA", 0) + 1

    return {
        "document_id": (cv.document_id if cv is not None else None),
        "cv_values": cv_values,
        "vision_values": vision_values,
        "document_evidence": document_evidence,
        "conflicts": conflicts,
        "validation": validation,
        "final_agreed_values": final_agreed_values,
        "reported_vs_calculated": reported_calculated,
        "provenance": provenance,
        "summary": counts,
        "has_unresolved_conflicts": any(v["status"] == doc_ev.UNRESOLVED_CONFLICT for v in validation.values()),
    }


def apply_final_agreement_to_plan(plan, cv: IndependentCVResult | None, vision: VisionDocumentResult | None):
    """Return a plan whose scalar fields are sourced from independent CV/Vision agreement.

    The legacy resolver remains available for tests/older callers. When independent CV is
    attached to ExtractionResult, this function becomes the authoritative final-value layer.

    Finalization ALWAYS executes, even when independent CV evidence is
    unavailable. `cv is None` (no caller ever attempted independent CV
    validation) and `cv.status == "FAILED"` (it was attempted and raised)
    are both treated as "zero independent CV measurements" and fed through
    the same reconciliation `build_final_agreement` already uses for a
    real-but-empty CV result -- they must NOT short-circuit into returning
    the pre-fusion plan untouched, or a stale/misattributed first-pass
    Vision value could ship as final with no reconciliation, no
    footprint-vs-plot sanity check, and no visible indication that CV
    validation never actually confirmed it. A pre-fusion value Vision does
    not compete with is still preserved via the `fallback` parameter inside
    `vf()` below -- CV failure must not blank out otherwise-good values.
    """
    cv_status_note: Optional[str] = None
    if cv is None:
        cv_status_note = (
            "Independent CV validation was not attempted for this document; "
            "fields below are Vision-only or carried over from the pre-fusion pass, "
            "not cross-source verified."
        )
        cv = IndependentCVResult(document_id=getattr(plan, "source_document_id", "unknown"), status="NOT_RUN")
    elif cv.status == "FAILED":
        cv_status_note = (
            f"Independent CV validation failed ({cv.error or 'unknown error'}); "
            "fields below are Vision-only or carried over from the pre-fusion pass, "
            "not cross-source verified."
        )
    if vision is None:
        # Keep CV values, but mark them explicitly as CV-only.
        empty = VisionDocumentResult(model_name="disabled", pages=[], enabled=False)
        fusion = build_final_agreement(cv, empty)
    else:
        fusion = build_final_agreement(cv, vision)
    v = fusion["values"]

    def vf(path: str, fallback):
        item = v.get(path)
        if item and item["status"] == "CONFLICT":
            # Independent CV and Vision genuinely disagree beyond tolerance
            # for this field. `fallback` (the pre-agreement first-pass
            # value from pipeline.py) is NOT a safe substitute here: on a
            # real sheet that first pass itself often already picked
            # whichever value Vision reported, grounded against the page's
            # own printed text but with NO check that the number actually
            # belongs to the RIGHT semantic region (e.g. a first-floor
            # room's width being read as the overall building width,
            # confirmed live -- see PHASE4_SPATIAL_AWARENESS_NOTES.md).
            # Silently keeping that fallback here would mean this
            # "authoritative final agreement" layer detects the exact
            # disagreement it exists to catch and then does nothing about
            # it. Surface the conflict explicitly instead.
            cv_val, vision_val = item.get("cv_value"), item.get("vision_value")
            return ValueField[float].conflicting(Conflict(
                description=item["reason"],
                conflicting_raw_values=(
                    [UnitValue(magnitude=cv_val, unit=item["unit"]), UnitValue(magnitude=vision_val, unit=item["unit"])]
                    if cv_val is not None and vision_val is not None else []
                ),
                conflicting_sources=["independent_cv", "vision"],
            ))
        if item and item["value"] is not None and item["status"] in {"AGREED", "CV_ONLY", "VISION_ONLY"}:
            # `item["confidence"]` already carries whatever
            # `build_final_agreement`/`_value_field` correctly derived from
            # the SOURCE measurement's own confidence (see `_value_field`'s
            # own docstring for the real bug this replaces: a CV-only or
            # Vision-only value used to be hardcoded MEDIUM here regardless
            # of status, independent of how much the extractor itself
            # trusted the number -- silently defeating every downstream
            # LOW-confidence gate, including `backend/compliance/engine.py`'s
            # own REQUIRES_REVIEW check. This is the exact same bug already
            # fixed in `pipeline.build_normalized_plan`'s own `fv()` closure
            # -- this was the second, still-live copy of it, confirmed live
            # on every real PDF plan's own NormalizedPlan via
            # `backend/tools/run_evidence_decision_shadow.py`'s shadow-mode
            # comparison against `evidence_decision.decide_field` before
            # this fix: every scored field on PLAN2/4/5/6 shipped
            # ConfidenceLevel.MEDIUM regardless of whether the underlying
            # evidence was strong or weak, because this branch is the one
            # every PDF plan with a resolved plot/building candidate
            # actually takes -- DXF-only plans go through the already-fixed
            # `pipeline.py` `fv()` above instead, since their candidate
            # lists are always empty).
            return ValueField[float](value=item["value"], normalized_value=UnitValue(magnitude=item["value"], unit=item["unit"]),
                confidence=Confidence(level=ConfidenceLevel(item["confidence"]), reason=item["reason"]),
                source=item["source"])
        return fallback

    plan.plot.width = vf("plot.width", plan.plot.width)
    plan.plot.depth = vf("plot.depth", plan.plot.depth)
    building_width_final = vf("building.width", plan.building.width)
    building_depth_final = vf("building.depth", plan.building.depth)
    plan.building.width = building_width_final
    plan.building.depth = building_depth_final

    # --- plot.area: reconcile against the FINAL (post-agreement) width/depth --
    # `vf("plot.area", plan.plot.area)` alone would silently keep whatever
    # `plan.plot.area` was computed to be *before* this function possibly just
    # replaced plot.width/plot.depth with different, better values above --
    # producing an area inconsistent with the very width/depth shown next to
    # it on the dashboard. Recompute from the final geometry instead whenever
    # that's reliable, and only fall back to a direct area reading/legacy
    # value when the final width/depth pair itself isn't trustworthy.
    direct_plot_area = vf("plot.area", None)
    if direct_plot_area is None and plan.plot.area is not None and "Area Statement" in (plan.plot.area.source or ""):
        # The independent CV/Vision layer has no notion of the sheet's own
        # printed "Site Area" label (it only measures width/depth), so it
        # never appears in `v`/`vf` above. Without this, a printed Site
        # Area reading computed upstream (backend.spatial_reasoning.areas
        # .find_area_statement) would be silently discarded by
        # `_reconcile_area_field` below in favour of width x depth the
        # moment both final dimensions are present -- exactly wrong for a
        # non-rectangular/irregular plot, which is what that printed label
        # exists to correct for. Feed it in as `direct` evidence instead,
        # so it goes through the same agreement-vs-conflict tolerance check
        # as any other direct area reading.
        direct_plot_area = plan.plot.area
    plan.plot.area = _reconcile_area_field(
        direct=direct_plot_area,
        width_field=plan.plot.width,
        depth_field=plan.plot.depth,
        polygon=(plan.plot.geometry.polygon if plan.plot.geometry else None),
        fallback=plan.plot.area,
        label="plot.area",
        trust_direct_over_geometry=(direct_plot_area is not None and "Area Statement" in (direct_plot_area.source or "")),
    )

    # --- building.footprint_area: same reconciliation, plus the physical
    # plausibility re-check (footprint cannot exceed plot area) must use the
    # just-corrected plot.area above, not a stale pre-agreement value, or a
    # genuinely fine footprint gets wrongly flagged CONFLICTING (and shows up
    # as MISSING on the dashboard) purely because it was compared against the
    # wrong plot-area number.
    direct_footprint_area = vf("building.footprint_area", None)
    if (
        direct_footprint_area is None
        and plan.building.footprint_area is not None
        and "Area Statement" in (plan.building.footprint_area.source or "")
    ):
        # Same reasoning as the plot.area case above, for the printed
        # per-floor footprint reading.
        direct_footprint_area = plan.building.footprint_area
    footprint_area = _reconcile_area_field(
        direct=direct_footprint_area,
        width_field=plan.building.width,
        depth_field=plan.building.depth,
        polygon=(plan.building.geometry.polygon if plan.building.geometry else None),
        fallback=plan.building.footprint_area,
        label="building.footprint_area",
        trust_direct_over_geometry=(
            direct_footprint_area is not None and "Area Statement" in (direct_footprint_area.source or "")
        ),
    )
    if (
        footprint_area.value is not None
        and plan.plot.area.value is not None
        and plan.plot.area.value > 0
        and footprint_area.value / plan.plot.area.value > 1.02
    ):
        footprint_area = ValueField[float].conflicting(
            Conflict(
                description=(
                    f"Computed building.footprint_area ({footprint_area.value:.3f} m2) exceeds "
                    f"the final resolved plot.area ({plan.plot.area.value:.3f} m2), which is "
                    "physically impossible. This most likely means the selected building "
                    "candidate does not actually belong to the same drawing region as the "
                    "selected plot candidate."
                ),
                conflicting_raw_values=[
                    UnitValue(magnitude=round(footprint_area.value, 4), unit=CanonicalUnit.SQUARE_METRE.value),
                    UnitValue(magnitude=plan.plot.area.value, unit=CanonicalUnit.SQUARE_METRE.value),
                ],
                conflicting_sources=["building.footprint_area (computed)", "plot.area"],
            )
        )
    plan.building.footprint_area = footprint_area

    plan.road.width = vf("road.width", plan.road.width)
    plan.setbacks.front = vf("setbacks.front", plan.setbacks.front)
    plan.setbacks.rear = vf("setbacks.rear", plan.setbacks.rear)
    plan.setbacks.left = vf("setbacks.left", plan.setbacks.left)
    plan.setbacks.right = vf("setbacks.right", plan.setbacks.right)

    # A CV/Vision-fused setback reading is still just one more candidate
    # value, not ground truth -- it can come from a misread caption or a
    # label anchored to the wrong sub-drawing. Flag (never null out or
    # downgrade) any value that cannot physically fit the FINAL, just-
    # reconciled plot/building width & depth above -- the numbers this
    # setback is actually shown next to on the dashboard, not the
    # pre-fusion ones the legacy path checked against. See
    # `flag_if_setback_implausible`'s own docstring for why this only ever
    # attaches `.conflict`: an earlier version of this check instead shipped
    # `ValueField.conflicting()` (value=None, confidence=CONFLICTING) and
    # caused a real regression -- PLAN5's correct, ground-truthed
    # setbacks.front=3.0m got nulled out because the budget was computed
    # from an unrelated, independently low-confidence building.depth
    # reading. `flag_if_setback_implausible` only evaluates the budget when
    # both companion dimensions are themselves HIGH confidence, which is
    # what avoids re-triggering that regression while still catching the
    # originally-reported bug (a plan whose building.width resolved
    # fractionally larger than plot.width, both at HIGH confidence, still
    # shipping a right-setback reading several times larger than the near-
    # zero width budget that left).
    for _side in ("front", "rear", "left", "right"):
        setattr(plan.setbacks, _side, flag_if_setback_implausible(
            _side, getattr(plan.setbacks, _side), f"setbacks.{_side}",
            plan.plot.width, plan.plot.depth, plan.building.width, plan.building.depth,
            # Fused values come from the sheet itself (independent CV /
            # Vision), so "width" is the drawing's horizontal extent, not
            # necessarily the frontage-parallel one.
            frontage_relative_axes=False,
        ))

    # --- coverage / far: recompute from the now-reconciled areas rather than
    # trusting a separately-extracted "coverage"/"far" reading in isolation --
    # those two numbers are *defined* as footprint_area/plot_area, so deriving
    # them from the final areas keeps the dashboard internally consistent.
    # A direct, agreed reading is still preferred when both independent
    # measurements agree with each other AND with the recomputed value
    # (kept via `vf` fallback semantics below), so an explicitly printed
    # coverage/FAR statement is not silently discarded.
    computed_coverage = coverage_field(plan.building.footprint_area, plan.plot.area)
    computed_far = far_field(plan.building.footprint_area, plan.plot.area, plan.building.floor_count)
    plan.coverage = vf("coverage", computed_coverage if computed_coverage.value is not None else plan.coverage)
    plan.far = vf("far", computed_far if computed_far.value is not None else plan.far)

    for field in (
        plan.plot.width, plan.plot.depth, plan.plot.area,
        plan.building.width, plan.building.depth, plan.building.footprint_area,
        plan.road.width,
        plan.setbacks.front, plan.setbacks.rear, plan.setbacks.left, plan.setbacks.right,
        plan.coverage, plan.far,
    ):
        if field is not None and field.conflict is not None and field.conflict not in plan.conflicts:
            plan.conflicts.append(field.conflict)
    plan.overall_confidence_note = (plan.overall_confidence_note or "") + " | Final scalar values passed through independent CV + Vision agreement layer."
    if cv_status_note:
        plan.overall_confidence_note = plan.overall_confidence_note + " | " + cv_status_note
        plan.metadata["cv_validation_status"] = cv.status
        if cv.error:
            plan.metadata["cv_validation_error"] = cv.error
    return plan


def build_vision_only_plan(vision: VisionDocumentResult, plan_id: str, document_id: str):
    """Build a NormalizedPlan using ONLY independent Vision evidence.

    This is the dashboard's "Vision" extraction mode: unlike Fusion, CV must
    have zero influence on the result (see BUILDCheck task Section 6). It
    deliberately does not touch `backend.cv_extraction` at all -- no plot
    candidate scoring, no page-space geometry, no independent CV
    measurements -- and reuses the same `LENGTH_FIELDS`/`NUMERIC_FIELDS`
    Vision-reading helpers this module already uses for Fusion, just without
    ever consulting the CV side of them.
    """
    from backend.schemas.normalized_plan import (
        NormalizedPlan, PlotSection, BuildingSection, RoadSection, SetbackSection,
    )

    def _vision_only_level(conf: float, grounded: Optional[bool], reason: str) -> tuple[ConfidenceLevel, str]:
        """Vision's own self-reported score alone may not establish HIGH/
        MEDIUM confidence in Vision-only mode (audit finding: self-reported
        confidence was being laundered straight into ConfidenceLevel with
        no independent check). `grounded is False` means grounding actually
        ran and found no native text layer to confirm this value against at
        all (e.g. a scanned page) -- not a real independent confirmation,
        so the level is capped at LOW regardless of the model's own score,
        the same way `dimension_classification.vision_only_dimensions`
        already caps an uncorroborated Vision-only dimension. `grounded is
        True` (value actually found in the PDF's native text) or `None`
        (grounding did not run, e.g. an older cached result) leave the
        self-reported score in charge, unchanged from before this fix.
        """
        level = confidence_level_from_score(conf)
        if grounded is False:
            level = cap_confidence_level(level, ConfidenceLevel.LOW)
            reason = f"{reason} (not independently grounded: no native text layer to verify this value against)."
        return level, reason

    def length_field(semantic: str) -> ValueField[float]:
        r = _vision_length(vision, semantic)
        if r is None:
            return ValueField[float].missing(f"No Vision evidence for {semantic} (Vision-only mode; CV not consulted).")
        val, conf, evidence, page, grounded = r
        level, reason = _vision_only_level(conf, grounded, evidence or f"Vision semantic {semantic} evidence.")
        return ValueField[float](
            value=round(val, 4),
            normalized_value=UnitValue(magnitude=round(val, 4), unit="m"),
            confidence=Confidence(level=level, reason=reason),
            source="VISION_ONLY",
        )

    def numeric_field(semantics: tuple[str, ...], unit_hint: str = "m2") -> ValueField[float]:
        r = _vision_numeric(vision, semantics)
        if r is None:
            return ValueField[float].missing(f"No Vision evidence for {semantics} (Vision-only mode; CV not consulted).")
        val, conf, evidence, page, unit, grounded = r
        level, reason = _vision_only_level(conf, grounded, evidence or "Vision semantic area evidence.")
        return ValueField[float](
            value=round(val, 4),
            normalized_value=UnitValue(magnitude=round(val, 4), unit=unit or unit_hint),
            confidence=Confidence(level=level, reason=reason),
            source="VISION_ONLY",
        )

    plot_width = length_field("PLOT_WIDTH")
    plot_depth = length_field("PLOT_DEPTH")
    plot_area = numeric_field(NUMERIC_FIELDS["plot.area"])
    if (
        (plot_area.value is None)
        and plot_width.value is not None
        and plot_depth.value is not None
    ):
        # No directly-stated plot area from Vision: derive from Vision's own
        # width/depth (never mixed with CV geometry).
        val = plot_width.value * plot_depth.value
        plot_area = ValueField[float](
            value=round(val, 4), normalized_value=UnitValue(magnitude=round(val, 4), unit=CanonicalUnit.SQUARE_METRE.value),
            confidence=Confidence(level=ConfidenceLevel.MEDIUM, reason="plot.area = Vision plot.width x plot.depth (no explicit Vision area reading)."),
            source="VISION_ONLY (derived)",
        )

    building_width = length_field("BUILDING_WIDTH")
    building_depth = length_field("BUILDING_DEPTH")
    footprint_area = numeric_field(NUMERIC_FIELDS["building.footprint_area"])
    if (
        (footprint_area.value is None)
        and building_width.value is not None
        and building_depth.value is not None
    ):
        val = building_width.value * building_depth.value
        footprint_area = ValueField[float](
            value=round(val, 4), normalized_value=UnitValue(magnitude=round(val, 4), unit=CanonicalUnit.SQUARE_METRE.value),
            confidence=Confidence(level=ConfidenceLevel.MEDIUM, reason="building.footprint_area = Vision building.width x building.depth (no explicit Vision area reading)."),
            source="VISION_ONLY (derived)",
        )
    if (
        footprint_area.value is not None and plot_area.value is not None and plot_area.value > 0
        and footprint_area.value / plot_area.value > 1.02
    ):
        footprint_area = ValueField[float].conflicting(Conflict(
            description=(
                f"Vision-reported building.footprint_area ({footprint_area.value:.3f} m2) exceeds "
                f"Vision-reported plot.area ({plot_area.value:.3f} m2), which is physically impossible."
            ),
            conflicting_raw_values=[
                UnitValue(magnitude=round(footprint_area.value, 4), unit=CanonicalUnit.SQUARE_METRE.value),
                UnitValue(magnitude=plot_area.value, unit=CanonicalUnit.SQUARE_METRE.value),
            ],
            conflicting_sources=["building.footprint_area (vision)", "plot.area (vision)"],
        ))

    road_width = length_field("ROAD_WIDTH")
    front = length_field("FRONT_SETBACK")
    rear = length_field("REAR_SETBACK")
    left = length_field("LEFT_SETBACK")
    right = length_field("RIGHT_SETBACK")

    floor_count = None
    floor_count_val = None
    for page in vision.pages:
        meta_fc = (page.metadata or {}).get("floor_count")
        if isinstance(meta_fc, (int, float)) and meta_fc > 0:
            floor_count_val = int(meta_fc)
            break
    if floor_count_val is None:
        floor_plan_regions = {r.id for p in vision.pages for r in p.regions if (r.type or "").upper() == "FLOOR_PLAN"}
        if floor_plan_regions:
            floor_count_val = len(floor_plan_regions)
    if floor_count_val is not None:
        floor_count = ValueField[int](
            value=floor_count_val,
            confidence=Confidence(level=ConfidenceLevel.MEDIUM, reason="Floor count from Vision metadata/distinct floor-plan regions."),
            source="VISION_ONLY",
        )

    coverage = coverage_field(footprint_area, plot_area)
    far = far_field(footprint_area, plot_area, floor_count)

    plan = NormalizedPlan(
        plan_id=plan_id,
        source_document_id=document_id,
        plot=PlotSection(width=plot_width, depth=plot_depth, area=plot_area),
        building=BuildingSection(width=building_width, depth=building_depth, footprint_area=footprint_area, floor_count=floor_count),
        road=RoadSection(width=road_width),
        setbacks=SetbackSection(front=front, rear=rear, left=left, right=right),
        coverage=coverage,
        far=far,
        overall_confidence_note=(
            "Built entirely from independent Vision evidence (Vision extraction mode): "
            "CV/native/OpenCV geometry was not consulted at any stage."
        ),
    )
    return plan


__all__ = ["ScoredPlotCandidate", "apply_final_agreement_to_plan", "build_final_agreement", "build_document_verified_fusion", "build_vision_only_plan"]


def _reconcile_area_field(
    direct: Optional[ValueField[float]],
    width_field: ValueField[float],
    depth_field: ValueField[float],
    polygon,
    fallback: ValueField[float],
    label: str,
    trust_direct_over_geometry: bool = False,
) -> ValueField[float]:
    """Pick between a directly-extracted area and width x depth of the FINAL dimensions.

    Mirrors `backend.spatial_reasoning.areas.polygon_area_field`'s intent
    (rectangular -> width*depth, else geometry/direct evidence) but is aware
    that `width_field`/`depth_field` here are the post-CV+Vision-agreement
    FINAL values, which is exactly the information the direct area reading
    (extracted earlier, independently) needs to be checked against.

    `trust_direct_over_geometry` inverts which side wins a strong
    disagreement. The default (False) assumes `direct` is a lingering,
    possibly-stale area computed from pre-agreement candidates -- the
    original case this function was built for (see module docstring's
    "17.59 x 9.21 => 15.87 m2" example), where the freshly-agreed
    width/depth are the more trustworthy side. That assumption inverts for
    a `direct` value read from the sheet's own printed, signed-off
    disclosure (Area Statement "Site Area" / per-floor area) -- that
    number was never derived from width/depth candidates at all, so it
    cannot be "stale" relative to them, and a composite sheet with several
    unrelated drawings makes it entirely plausible that `width_field`/
    `depth_field` themselves are the ones that picked up the wrong
    rectangle. Callers pass True for exactly that provenance (see
    apply_final_agreement_to_plan above).
    """
    width_ok = width_field is not None and width_field.value is not None and width_field.value > 0
    depth_ok = depth_field is not None and depth_field.value is not None and depth_field.value > 0

    geometric_value = None
    if width_ok and depth_ok:
        rectangularity = 1.0
        if polygon is not None:
            try:
                rectangularity = geo.rectangularity(polygon)
            except Exception:
                rectangularity = 1.0
        # Only trust width*depth as an area substitute when the footprint is
        # plausibly rectangular; for irregular footprints the direct/polygon
        # evidence remains authoritative (never blindly width*depth per the
        # "for irregular plots, do NOT blindly use width x depth" rule).
        if rectangularity >= 0.85:
            geometric_value = width_field.value * depth_field.value

    if geometric_value is None:
        # No reliable rectangular geometry to derive from: keep whatever
        # direct evidence exists, else the pre-agreement fallback.
        if direct is not None and direct.value is not None:
            return direct
        return fallback

    # A width*depth area is only as trustworthy as the WEAKER of the two
    # dimensions it's built from -- e.g. PLAN7 (a photograph with no real
    # site plan): its spurious plot.width/plot.depth are correctly capped to
    # LOW by `plot_resolution.plot_confidence`/`pipeline._cap_field_
    # confidence` upstream, but this function used to unconditionally stamp
    # the area computed from them HIGH, discarding that cap a second time --
    # the same "confidence computed correctly, then silently overwritten"
    # bug class `cap_confidence_level` itself already exists to prevent (see
    # its own docstring's PLAN7 example, which is about width/depth, not yet
    # this derived area). Reusing it here, rather than a fresh HIGH literal,
    # closes that gap for the derived value too.
    area_confidence_level = cap_confidence_level(width_field.confidence.level, depth_field.confidence.level)
    cap_note = (
        "" if area_confidence_level == ConfidenceLevel.HIGH
        else f" (confidence capped to {area_confidence_level.value}: derived from a {area_confidence_level.value}-confidence width/depth pair)"
    )
    geometric_field = ValueField[float](
        value=round(geometric_value, 4),
        normalized_value=UnitValue(magnitude=round(geometric_value, 4), unit=CanonicalUnit.SQUARE_METRE.value),
        confidence=Confidence(
            level=area_confidence_level,
            reason=(
                f"{label} = width x depth using the final, independently-agreed "
                f"{width_field.value:.4f} x {depth_field.value:.4f}.{cap_note}"
            ),
        ),
        source=f"{label} = width x depth (final agreed dimensions)",
    )

    if direct is None or direct.value is None or direct.value <= 0:
        return geometric_field

    rel_diff = abs(direct.value - geometric_value) / geometric_value if geometric_value else float("inf")
    if rel_diff <= _AREA_CONFLICT_TOLERANCE:
        # Direct reading is consistent with the final geometry; keep it (it
        # may carry provenance like a printed "SITE AREA" label) but note the
        # cross-check.
        direct.confidence.reason = (
            f"{direct.confidence.reason} (cross-checked against final width x depth = "
            f"{geometric_value:.3f} m2, within {rel_diff:.1%})."
        )
        return direct

    if trust_direct_over_geometry:
        # A printed, signed-off disclosure disagreeing this much with
        # width*depth is far more likely to mean width/depth picked up
        # geometry from an unrelated drawing on the same composite sheet
        # than that the printed figure is wrong. Keep the printed value,
        # but surface the disagreement plainly rather than silently
        # dropping the geometric side's information.
        direct.confidence.reason = (
            f"{direct.confidence.reason} Final width x depth gives {geometric_value:.3f} m2 "
            f"(relative difference {rel_diff:.1%}) -- likely picked up geometry from an "
            "unrelated drawing on the same sheet rather than this printed figure being wrong; "
            "the printed value is kept."
        )
        return direct

    # Strong disagreement, and `direct` is not a printed/authoritative
    # reading: the width and depth backing `geometric_field` were each
    # independently confirmed by the CV+Vision agreement layer, so they are
    # more trustworthy than a lingering/mismatched area reading computed
    # against different (pre-agreement) candidates.
    geometric_field.confidence.reason += (
        f" Overrides a conflicting direct {label} reading of {direct.value:.3f} m2 "
        f"(relative difference {rel_diff:.1%}), which disagreed too strongly with the "
        "final independently-agreed width/depth to be trusted."
    )
    return geometric_field
