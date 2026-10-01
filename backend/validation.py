"""
Independent CV <-> Vision validation.

This module is deliberately downstream of BOTH extractors:
- CV is run with Vision disabled.
- Vision is run without native-text grounding.
- Only this module compares the two evidence streams.

It never picks a winner. A disagreement is a first-class CONFLICT.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from backend.cv_extraction.pdf_extractor import PDFHybridExtractor
from backend.cv_extraction.site_plan import extract_independent_cv
from backend.schemas.validation import ExtractionValidationReport, ValidationField
from backend.schemas.vision import VisionDocumentResult, VisionDimension, VisionArea
from backend.schemas.units import LengthUnit, length_to_metres
from backend.schemas.independent_measurements import IndependentMeasurement
from backend.schemas.normalized_plan import NormalizedPlan
from backend.vision_extraction import get_vision_extractor


VISION_FIELD_TYPES = {
    "plot.width": "PLOT_WIDTH",
    "plot.depth": "PLOT_DEPTH",
    "building.width": "BUILDING_WIDTH",
    "building.depth": "BUILDING_DEPTH",
    "road.width": "ROAD_WIDTH",
    "setbacks.front": "FRONT_SETBACK",
    "setbacks.rear": "REAR_SETBACK",
    "setbacks.left": "LEFT_SETBACK",
    "setbacks.right": "RIGHT_SETBACK",
}

VISION_AREA_TYPES = {
    "plot.area": ("PLOT_AREA", "NET_PLOT_AREA"),
    "building.footprint_area": ("BUILDING_FOOTPRINT_AREA", "PROPOSED_COVERAGE_AREA"),
    "coverage": ("COVERAGE_PERCENT",),
    "far.area": ("FAR_AREA",),
    "far": ("FAR_RATIO",),
    "building.gross_built_up_area": ("TOTAL_BUILT_UP_AREA",),
}


@dataclass(frozen=True)
class _VisionEvidence:
    value_m: float
    confidence: float
    evidence: Optional[str]
    page: int


@dataclass(frozen=True)
class _VisionNumericEvidence:
    value: float
    confidence: float
    evidence: Optional[str]
    page: int
    unit: str


def _field_value(plan: NormalizedPlan, path: str) -> tuple[Optional[float], Optional[str]]:
    obj: Any = plan
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj.value, obj.source


def _vision_to_metres(dim: VisionDimension) -> Optional[float]:
    if dim.value is None:
        return None
    unit = (dim.unit or "m").strip().lower()
    # Common model spellings.
    aliases = {
        "meter": "m", "metre": "m", "meters": "m", "metres": "m",
        "millimeter": "mm", "millimetre": "mm", "millimeters": "mm", "millimetres": "mm",
        "centimeter": "cm", "centimetre": "cm", "centimeters": "cm", "centimetres": "cm",
        "feet": "ft", "foot": "ft",
    }
    unit = aliases.get(unit, unit)
    try:
        return length_to_metres(float(dim.value), LengthUnit(unit))
    except (ValueError, TypeError):
        return None


def _vision_area_to_numeric(area: VisionArea) -> tuple[Optional[float], Optional[str]]:
    if area.value is None:
        return None, None
    unit = (area.unit or "m2").strip().lower().replace("²", "2")
    aliases = {
        "sqm": "m2", "sq.m": "m2", "sq.m.": "m2", "sqft": "ft2", "sq.ft": "ft2",
        "sq.ft.": "ft2", "square_metre": "m2", "square_meter": "m2",
        "%": "%", "percent": "%", "percentage": "%", "ratio": "ratio",
    }
    unit = aliases.get(unit, unit)
    if unit == "m2":
        return float(area.value), "m2"
    if unit == "ft2":
        return float(area.value) * 0.09290304, "m2"
    if unit in {"%", "ratio"}:
        return float(area.value), unit
    return None, None


def _best_vision_area(result: VisionDocumentResult, semantic_types: tuple[str, ...]) -> Optional[_VisionNumericEvidence]:
    candidates: list[_VisionNumericEvidence] = []
    wanted = {x.upper() for x in semantic_types}
    for page in result.pages:
        for area in page.areas:
            if area.type.upper() not in wanted:
                continue
            value, unit = _vision_area_to_numeric(area)
            if value is None or unit is None:
                continue
            candidates.append(_VisionNumericEvidence(value=value, confidence=area.confidence, evidence=area.evidence, page=page.page_number, unit=unit))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x.confidence, x.page))
    return candidates[0]


def _cv_numeric_map(cv_result) -> dict[str, IndependentMeasurement]:
    out: dict[str, IndependentMeasurement] = {}
    for measurement in cv_result.measurements:
        value = measurement.value if measurement.value is not None else measurement.value_m
        if value is None:
            continue
        current = out.get(measurement.field)
        if current is None or measurement.confidence > current.confidence:
            out[measurement.field] = measurement
    return out


def _numeric_tolerance(a: float, b: float, unit: str) -> float:
    if unit == "%":
        return max(0.10, 0.01 * max(abs(a), abs(b)))
    if unit == "ratio":
        return max(0.01, 0.01 * max(abs(a), abs(b)))
    return max(0.05, 0.01 * max(abs(a), abs(b)))


def _compare_numeric(field: str, cv: Optional[IndependentMeasurement], vision: Optional[_VisionNumericEvidence]) -> ValidationField:
    cv_value = None if cv is None else (cv.value if cv.value is not None else cv.value_m)
    cv_unit = None if cv is None else cv.unit
    vv = None if vision is None else vision.value
    unit = vision.unit if vision is not None else (cv_unit or "m2")
    if cv_value is None and vv is None:
        return ValidationField(field=field, status="MISSING", unit=unit)
    if cv_value is None:
        return ValidationField(field=field, vision_value=round(vv, 6), status="VISION_ONLY", unit=unit,
                               vision_evidence=vision.evidence, vision_confidence=vision.confidence,
                               note=f"Vision page {vision.page}; CV did not resolve this field.")
    if vv is None:
        return ValidationField(field=field, cv_value=round(cv_value, 6), status="CV_ONLY", unit=unit,
                               cv_source=cv.source, note="CV resolved this field; Vision did not emit a semantic area value.")
    # Convert CV's m2/percentage/ratio units into Vision's canonical unit when possible.
    if cv_unit and cv_unit != unit:
        if cv_unit == "sqm": cv_unit = "m2"
        if cv_unit == "m2" and unit == "ft2":
            cv_value *= 10.7639104167
        elif cv_unit == "ft2" and unit == "m2":
            cv_value *= 0.09290304
    diff = abs(cv_value - vv)
    tol = _numeric_tolerance(cv_value, vv, unit)
    status = "AGREED" if diff <= tol else "CONFLICT"
    return ValidationField(field=field, cv_value=round(cv_value, 6), vision_value=round(vv, 6),
                           absolute_difference=round(diff, 6), tolerance=round(tol, 6), status=status,
                           unit=unit, cv_source=cv.source, vision_evidence=vision.evidence,
                           vision_confidence=vision.confidence,
                           note=(f"Vision page {vision.page}; within tolerance." if status == "AGREED"
                                 else f"Vision page {vision.page}; values differ beyond tolerance."))


def _best_vision_evidence(
    result: VisionDocumentResult, semantic_type: str
) -> Optional[_VisionEvidence]:
    candidates: list[_VisionEvidence] = []
    for page in result.pages:
        for dim in page.dimensions:
            if dim.type != semantic_type:
                continue
            value_m = _vision_to_metres(dim)
            if value_m is None:
                continue
            candidates.append(
                _VisionEvidence(
                    value_m=value_m,
                    confidence=dim.confidence,
                    evidence=dim.evidence,
                    page=page.page_number,
                )
            )
    if not candidates:
        return None
    # Confidence is the primary ranking; stable tie-break keeps the earliest page.
    candidates.sort(key=lambda x: (-x.confidence, x.page))
    return candidates[0]


def _cv_measurement_map(cv_result) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for measurement in cv_result.measurements:
        if measurement.value_m is None:
            continue
        current = out.get(measurement.field)
        if current is None or measurement.confidence > current.confidence:
            out[measurement.field] = measurement
    return out


def _tolerance(value_a: float, value_b: float) -> float:
    # 5 cm minimum is appropriate for drawing/model reading noise; 1% handles
    # larger dimensions without making the tolerance scale-independent.
    return max(0.05, 0.01 * max(abs(value_a), abs(value_b)))


def _compare(
    field: str,
    cv_value: Optional[float],
    cv_source: Optional[str],
    vision: Optional[_VisionEvidence],
) -> ValidationField:
    vv = vision.value_m if vision else None
    if cv_value is None and vv is None:
        return ValidationField(field=field, status="MISSING")
    if cv_value is None:
        return ValidationField(
            field=field,
            vision_value_m=round(vv, 6),
            status="VISION_ONLY",
            vision_evidence=vision.evidence,
            vision_confidence=vision.confidence,
            note=f"Vision page {vision.page}; CV did not resolve this field.",
        )
    if vv is None:
        return ValidationField(
            field=field,
            cv_value_m=round(cv_value, 6),
            status="CV_ONLY",
            cv_source=cv_source,
            note="CV resolved this field; Vision did not emit a semantic value.",
        )
    diff = abs(cv_value - vv)
    tol = _tolerance(cv_value, vv)
    status = "AGREED" if diff <= tol else "CONFLICT"
    return ValidationField(
        field=field,
        cv_value_m=round(cv_value, 6),
        vision_value_m=round(vv, 6),
        absolute_difference_m=round(diff, 6),
        tolerance_m=round(tol, 6),
        status=status,
        cv_source=cv_source,
        vision_evidence=vision.evidence,
        vision_confidence=vision.confidence,
        note=(
            f"Vision page {vision.page}; within tolerance."
            if status == "AGREED"
            else f"Vision page {vision.page}; values differ beyond tolerance."
        ),
    )


def validate_extractions(
    document_path: Path,
    *,
    document_id: Optional[str] = None,
    cv_extraction: Optional[ExtractionResult] = None,
    vision_result: Optional[VisionDocumentResult] = None,
    independent_cv_result=None,
) -> ExtractionValidationReport:
    """Run two independent evidence paths and compare them deterministically.

    CV validation is produced by :func:`extract_independent_cv`, which never
    enables Vision. The legacy PDFHybridExtractor is intentionally excluded
    from this validation path so its historical/global geometry cannot
    contaminate the independent comparison.
    """
    document_id = document_id or document_path.stem

    # Phase 3.3 validation uses ONLY the independent CV/native-text path.
    # The legacy PDFHybridExtractor is deliberately excluded so an old/global
    # geometry result can never contaminate the CV-vs-Vision comparison.
    cv_result = independent_cv_result or extract_independent_cv(document_path, document_id)
    cv_extraction = None

    if vision_result is None:
        vision_result = get_vision_extractor().analyze_pdf(
            document_path, ground_against_native_text=False
        )

    cv_map = _cv_measurement_map(cv_result)
    cv_numeric = _cv_numeric_map(cv_result)
    fields: list[ValidationField] = []
    for field, semantic_type in VISION_FIELD_TYPES.items():
        cv_measurement = cv_map.get(field)
        cv_value = cv_measurement.value_m if cv_measurement else None
        cv_source = cv_measurement.source if cv_measurement else None
        vision = _best_vision_evidence(vision_result, semantic_type)
        fields.append(_compare(field, cv_value, cv_source, vision))
    for field, semantic_types in VISION_AREA_TYPES.items():
        fields.append(_compare_numeric(field, cv_numeric.get(field), _best_vision_area(vision_result, semantic_types)))

    counts = {status: sum(f.status == status for f in fields) for status in
              ("AGREED", "CV_ONLY", "VISION_ONLY", "CONFLICT", "MISSING")}

    from backend.spatial_reasoning.final_fusion import build_final_agreement, build_document_verified_fusion
    final_agreement = build_final_agreement(cv_result, vision_result)
    document_verified_fusion = build_document_verified_fusion(cv_result, vision_result, pdf_path=str(document_path))
    return ExtractionValidationReport(
        document_id=document_id,
        cv_warnings=list(cv_result.warnings),
        vision_warnings=list(vision_result.warnings),
        fields=fields,
        summary=counts,
        cv_result=cv_result.model_dump(mode="json"),
        legacy_cv_plan=None,
        vision_result=vision_result.model_dump(mode="json"),
        final_agreed_values=final_agreement,
        document_verified_fusion=document_verified_fusion,
    )

