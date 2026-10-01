"""
PDF <-> DXF post-hoc cross-check.

This module runs STRICTLY AFTER `PDFHybridExtractor.extract()` and
`DXFHybridExtractor.extract()` (via `pipeline.build_normalized_plan`) have
each independently finished. It never calls either extractor and never
feeds one modality's evidence into the other's extraction algorithm --
that separation is a non-negotiable architectural rule elsewhere in this
project, unaffected by this module.

Why this exists: live testing established the PDF pipeline is reliably
correct while the DXF pipeline has multiple documented, root-caused
failure modes (wrong sub-drawing selected on a multi-drawing sheet,
illegible OCR captions, scale mismatches -- see DXF_FAILURE_TAXONOMY.md).
When a user has both files for the same real building, the PDF's own
independently-resolved values are the single most direct piece of evidence
available for catching a wrong DXF reading.

This is a deliberate, NARROW exception to this project's usual "never
silently pick a winner on a genuine conflict" rule (used everywhere else --
`backend/validation.py`'s CV-vs-Vision comparator, `final_fusion.py`'s own
CV-vs-Vision arbitration -- both refuse to guess and ship CONFLICTING
instead). PDF wins here specifically because it is the documented-reliable
side, not because agreement was reached; the disagreement itself is never
hidden -- it survives on the shipped field's own `ValueField.conflict` and
in the parallel `PdfDxfReconciliationReport` this function also returns.

Confidence-level note, load-bearing: `backend/compliance/engine.py` treats
ANY field with `confidence.level == ConfidenceLevel.CONFLICTING` as an
automatic `CONFLICTING_EVIDENCE` verdict, regardless of whether `.value` is
populated (confirmed by reading that module directly). Setting
`CONFLICTING` on a PDF-wins field would therefore make compliance refuse to
use the very value this module just decided to ship -- so a CONFLICT here
never touches `confidence.level`; only `.conflict` is populated.
"""

from __future__ import annotations

from typing import Any, Optional

from backend.schemas.evidence import Confidence, Conflict, ValueField
from backend.schemas.enums import ConfidenceLevel
from backend.schemas.normalized_plan import NormalizedPlan
from backend.schemas.pdf_dxf_reconciliation import PdfDxfFieldComparison, PdfDxfReconciliationReport
from backend.schemas.units import UnitValue

# Reuses `eval_harness.py`'s own established tolerance regime (a loose,
# uniform absolute-floor-or-relative-percentage tolerance already used
# throughout this project to judge "does this DXF/CV value agree with a
# known-good number") rather than `validation.py`'s tighter CV-vs-Vision
# tolerance -- that one is calibrated for two readings of the SAME PDF
# page; DXF's own reconstruction noise (envelope/wall-union fitting, per
# DXF_FAILURE_TAXONOMY.md items 6/8) is coarser, so a looser comparison is
# the right precedent to reuse here, not a third scheme to invent.
# Provisional, per this project's own "Provisional thresholds" discipline
# -- re-measure as the ground-truthed PDF+DXF corpus grows.
PDF_DXF_ABS_TOL = 0.15
PDF_DXF_REL_TOL = 0.05

# Field path -> canonical unit label (for the report only; both plans
# already store values in canonical units, so no conversion happens here).
_FIELD_UNITS: dict[str, str] = {
    "plot.width": "m", "plot.depth": "m", "plot.area": "m2",
    "building.width": "m", "building.depth": "m", "building.footprint_area": "m2",
    "building.floor_count": "count",
    "road.width": "m",
    "setbacks.front": "m", "setbacks.rear": "m", "setbacks.left": "m", "setbacks.right": "m",
    "coverage": "%", "far": "ratio",
}

# The only fields `DXFHybridExtractor` can ever populate -- it always
# returns empty legacy `plot_candidates`/`building_candidates`/
# `road_candidates` (confirmed by reading `dxf_extractor.py`), so
# PDF-only fields (`building_use`, `development_area`, height fields,
# `floor_areas`, `metadata`) are never compared here at all -- not
# reported as noise, simply out of scope for this cross-check.
COMPARED_FIELDS: tuple[str, ...] = tuple(_FIELD_UNITS.keys())


def _within_tolerance(a: float, b: float, abs_tol: float = PDF_DXF_ABS_TOL, rel_tol: float = PDF_DXF_REL_TOL) -> bool:
    tol = max(abs_tol, abs(b) * rel_tol)
    return abs(a - b) <= tol


def _tolerance_used(a: float, b: float, abs_tol: float = PDF_DXF_ABS_TOL, rel_tol: float = PDF_DXF_REL_TOL) -> float:
    return max(abs_tol, abs(b) * rel_tol)


def _get_field(plan: NormalizedPlan, path: str) -> Optional[ValueField]:
    obj: Any = plan
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    return getattr(obj, parts[-1])


def _set_field(plan: NormalizedPlan, path: str, value_field: ValueField) -> None:
    obj: Any = plan
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value_field)


def _value_and_level(vf: Optional[ValueField]) -> tuple[Optional[float], Optional[str]]:
    if vf is None:
        return None, None
    level = vf.confidence.level.value if vf.confidence is not None else None
    return vf.value, level


def reconcile_pdf_dxf(pdf_plan: NormalizedPlan, dxf_plan: NormalizedPlan) -> tuple[NormalizedPlan, PdfDxfReconciliationReport]:
    """Compare every field `COMPARED_FIELDS` names between two independently
    produced `NormalizedPlan`s, mutate a copy of `pdf_plan` in place to
    reflect the reconciliation (PDF's own structure/geometry/PDF-only
    fields are the base -- this is fundamentally "PDF's plan, corroborated
    or flagged by DXF", never the other way around), and return it
    alongside a full, both-sides-visible `PdfDxfReconciliationReport`.

    Neither input plan is mutated; `pdf_plan` is deep-copied first.
    """
    reconciled = pdf_plan.model_copy(deep=True)
    comparisons: list[PdfDxfFieldComparison] = []
    summary = {"AGREED": 0, "CONFLICT": 0, "PDF_ONLY": 0, "DXF_ONLY": 0, "BOTH_MISSING": 0}

    for field_path in COMPARED_FIELDS:
        unit = _FIELD_UNITS[field_path]
        pdf_vf = _get_field(pdf_plan, field_path)
        dxf_vf = _get_field(dxf_plan, field_path)
        pdf_value, pdf_level = _value_and_level(pdf_vf)
        dxf_value, dxf_level = _value_and_level(dxf_vf)

        if pdf_value is None and dxf_value is None:
            status = "BOTH_MISSING"
            note = "Neither PDF nor DXF resolved this field."
            shipped_value = None
        elif pdf_value is None:
            status = "DXF_ONLY"
            note = (
                f"Only DXF resolved this field ({dxf_value}). NOT backfilled onto the "
                "reconciled plan -- an unverified DXF-only value is a different, "
                "riskier claim than 'PDF corroborates DXF', and is out of this "
                "cross-check's scope by design; recorded here so it is never silently "
                "discarded."
            )
            shipped_value = None
        elif dxf_value is None:
            status = "PDF_ONLY"
            note = "Only PDF resolved this field; DXF found nothing to cross-check against. Shipped unchanged."
            shipped_value = pdf_value
        else:
            agree = _within_tolerance(dxf_value, pdf_value)
            tol = _tolerance_used(dxf_value, pdf_value)
            diff = abs(pdf_value - dxf_value)
            if agree:
                status = "AGREED"
                note = f"PDF and DXF independently agree within tolerance ({diff:.4f} <= {tol:.4f} {unit})."
                boosted = pdf_vf.model_copy(update={
                    "confidence": Confidence(
                        level=ConfidenceLevel.HIGH,
                        score=pdf_vf.confidence.score,
                        reason=(
                            f"Independently corroborated by DXF ({dxf_value:.4f} {unit}, within {tol:.4f}) -- "
                            f"cross-modal agreement. {pdf_vf.confidence.reason or ''}".strip()
                        ),
                    ),
                })
                _set_field(reconciled, field_path, boosted)
                shipped_value = pdf_value
            else:
                status = "CONFLICT"
                note = (
                    f"PDF ({pdf_value:.4f} {unit}) and DXF ({dxf_value:.4f} {unit}) disagree beyond "
                    f"tolerance ({diff:.4f} > {tol:.4f}). Shipping PDF's value as authoritative "
                    "(documented-reliable pipeline) -- flagged via this field's own `.conflict`, "
                    "never silently."
                )
                conflict = Conflict(
                    description=(
                        f"PDF and DXF disagree on {field_path}: PDF={pdf_value:.4f} {unit}, "
                        f"DXF={dxf_value:.4f} {unit} (difference {diff:.4f} exceeds tolerance {tol:.4f}). "
                        "PDF's value shipped as authoritative; see PdfDxfReconciliationReport for detail."
                    ),
                    conflicting_raw_values=[
                        UnitValue(magnitude=pdf_value, unit=unit),
                        UnitValue(magnitude=dxf_value, unit=unit),
                    ],
                    conflicting_sources=["pdf_extractor", "dxf_extractor"],
                )
                flagged = pdf_vf.model_copy(update={"conflict": conflict})
                _set_field(reconciled, field_path, flagged)
                if conflict not in reconciled.conflicts:
                    reconciled.conflicts.append(conflict)
                shipped_value = pdf_value

        summary[status] += 1
        comparisons.append(PdfDxfFieldComparison(
            field=field_path, unit=unit, pdf_value=pdf_value, dxf_value=dxf_value,
            pdf_confidence=pdf_level, dxf_confidence=dxf_level,
            absolute_difference=(abs(pdf_value - dxf_value) if pdf_value is not None and dxf_value is not None else None),
            tolerance=(_tolerance_used(dxf_value, pdf_value) if pdf_value is not None and dxf_value is not None else None),
            status=status, shipped_value=shipped_value, note=note,
        ))

    note_suffix = f" | PDF+DXF cross-check: {summary}"
    reconciled.overall_confidence_note = (reconciled.overall_confidence_note or "") + note_suffix
    reconciled.metadata["pdf_dxf_reconciliation_summary"] = summary

    report = PdfDxfReconciliationReport(
        pdf_document_id=pdf_plan.source_document_id,
        dxf_document_id=dxf_plan.source_document_id,
        fields=comparisons,
        summary=summary,
        has_conflicts=summary["CONFLICT"] > 0,
    )
    return reconciled, report


__all__ = ["reconcile_pdf_dxf", "COMPARED_FIELDS", "PDF_DXF_ABS_TOL", "PDF_DXF_REL_TOL"]
