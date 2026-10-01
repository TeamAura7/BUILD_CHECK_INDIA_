from __future__ import annotations

from pathlib import Path

from backend.schemas.extraction import ExtractionResult
from backend.schemas.enums import DocumentType
from backend.schemas.vision import VisionDimension, VisionDocumentResult, VisionPageResult
from backend.validation import _compare, _VisionEvidence, validate_extractions


def test_agreement_is_explicit_and_never_fuses_values():
    field = _compare(
        "plot.width",
        12.19,
        "CV geometry/native dimension",
        _VisionEvidence(12.20, 0.91, "12.20", 1),
    )
    assert field.status == "AGREED"
    assert field.cv_value_m == 12.19
    assert field.vision_value_m == 12.20


def test_conflict_is_not_resolved_to_either_source():
    field = _compare(
        "plot.width",
        12.19,
        "CV geometry/native dimension",
        _VisionEvidence(11.20, 0.99, "11.20", 1),
    )
    assert field.status == "CONFLICT"
    assert field.cv_value_m == 12.19
    assert field.vision_value_m == 11.20


def test_cv_only_and_vision_only_are_preserved():
    cv_only = _compare("plot.width", 12.19, "CV", None)
    vision_only = _compare("plot.depth", None, None, _VisionEvidence(9.14, 0.9, "9.14", 1))
    missing = _compare("building.width", None, None, None)

    assert cv_only.status == "CV_ONLY"
    assert cv_only.cv_value_m == 12.19
    assert vision_only.status == "VISION_ONLY"
    assert vision_only.vision_value_m == 9.14
    assert missing.status == "MISSING"


def test_validation_uses_independent_cv_result_without_overwrite(monkeypatch):
    from backend.schemas.independent_measurements import IndependentCVResult, IndependentMeasurement

    cv_result = IndependentCVResult(
        document_id="p",
        pages_analyzed=[1],
        measurements=[
            IndependentMeasurement(
                field="plot.width", value_m=12.19, source="NATIVE_TEXT",
                confidence=0.99, evidence=["12.19"], page=1
            )
        ],
    )
    vision = VisionDocumentResult(
        model_name="test",
        pages=[
            VisionPageResult(
                page_number=1,
                dimensions=[
                    VisionDimension(
                        value=12.19, unit="m", type="PLOT_WIDTH",
                        evidence="12.19", confidence=0.95,
                    )
                ],
            )
        ],
    )

    report = validate_extractions(
        Path("p.pdf"), document_id="p",
        independent_cv_result=cv_result, vision_result=vision,
    )
    width = next(f for f in report.fields if f.field == "plot.width")
    assert width.status == "AGREED"
    assert width.cv_value_m == 12.19
    assert width.vision_value_m == 12.19


def test_cv_extractor_can_be_constructed_with_vision_explicitly_disabled():
    from backend.cv_extraction.pdf_extractor import PDFHybridExtractor
    extractor = PDFHybridExtractor(enable_vision=False)
    assert extractor.enable_vision is False



def test_plan2_expected_area_contract():
    """PLAN2 is a real acceptance fixture for independently validated area fields."""
    import json
    from pathlib import Path
    expected = json.loads(Path("data/test_plans/PLAN2.expected.json").read_text(encoding="utf-8"))
    assert expected["plot.area"] == 222.83
    assert expected["building.footprint_area"] == 174.52
    assert expected["coverage"] == 78.32
    assert expected["far.area"] == 386.55
    assert expected["far"] == 1.73


def test_plan2_independent_cv_extracts_geometry_and_declared_areas_without_legacy_path():
    """Real PLAN2 acceptance test: independent CV must own all Phase-3.3 fields."""
    from backend.cv_extraction.site_plan import extract_independent_cv

    result = extract_independent_cv(Path("data/test_plans/PLAN2.pdf"), "PLAN2")
    values = {}
    for m in result.measurements:
        v = m.value_m if m.value_m is not None else m.value
        if v is not None:
            values[m.field] = v

    expected = {
        "plot.width": 12.19,
        "plot.depth": 18.28,
        "building.width": 10.59,
        "building.depth": 16.48,
        "road.width": 9.20,
        "setbacks.front": 1.00,
        "setbacks.rear": 0.80,
        "setbacks.left": 0.80,
        "setbacks.right": 0.80,
        "plot.area": 222.83,
        "building.footprint_area": 174.52,
        "coverage": 78.32,
        "far.area": 386.55,
        "far": 1.73,
        "building.gross_built_up_area": 579.90,
    }
    for field, target in expected.items():
        assert field in values, field
        assert abs(values[field] - target) <= max(0.03, target * 0.005), (field, values[field], target)


def test_validation_report_does_not_include_legacy_cv_plan():
    """Legacy/global CV is never part of the independent agreement contract."""
    from backend.schemas.independent_measurements import IndependentCVResult, IndependentMeasurement

    cv_result = IndependentCVResult(
        document_id="p", pages_analyzed=[1], measurements=[
            IndependentMeasurement(field="plot.width", value_m=12.19, source="NATIVE_TEXT", confidence=0.99)
        ]
    )
    report = validate_extractions(Path("p.pdf"), document_id="p", independent_cv_result=cv_result,
                                  vision_result=VisionDocumentResult(model_name="test", pages=[]))
    assert report.legacy_cv_plan is None
