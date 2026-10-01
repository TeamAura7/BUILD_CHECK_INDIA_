from __future__ import annotations

from backend.cv_extraction.dimension_candidates import detect_dimension_candidates
from backend.cv_extraction.raw_types import RawLine, RawTextItem, SourceKind
from backend.schemas.geometry import BoundingBox, Line, Point


def _text_item(text: str, page: int = 0) -> RawTextItem:
    return RawTextItem(
        text=text,
        bounding_box=BoundingBox(min_x=0, min_y=0, max_x=20, max_y=10),
        page=page,
        source=SourceKind.PDF_TEXT,
    )


def test_detects_plain_metric_value():
    candidates = detect_dimension_candidates([_text_item("12.50 m")])
    assert len(candidates) == 1
    assert candidates[0].numeric_value == 12.5
    assert candidates[0].unit_hint == "m"


def test_detects_feet_inches_value():
    candidates = detect_dimension_candidates([_text_item("3'-6\"")])
    assert len(candidates) == 1
    assert candidates[0].unit_hint == "ft_in"
    assert candidates[0].numeric_value == 3.5


def test_detects_area_unit():
    candidates = detect_dimension_candidates([_text_item("216.0 sq.ft")])
    assert len(candidates) == 1
    assert candidates[0].unit_hint == "sq_ft"


def test_non_numeric_text_is_not_a_candidate():
    candidates = detect_dimension_candidates([_text_item("SITE LAYOUT PLAN")])
    assert candidates == []


def test_bare_number_without_unit_still_a_low_confidence_candidate():
    candidates = detect_dimension_candidates([_text_item("42")])
    assert len(candidates) == 1
    assert candidates[0].unit_hint is None
    assert candidates[0].confidence < 0.75


def test_nearby_geometry_ids_populated_when_line_close():
    text_item = RawTextItem(
        text="9.0 m",
        bounding_box=BoundingBox(min_x=10, min_y=10, max_x=30, max_y=20),
        page=0,
        source=SourceKind.PDF_TEXT,
    )
    nearby_line = RawLine(
        line=Line(start=Point(x=0, y=15), end=Point(x=100, y=15)),
        page=0,
        source=SourceKind.VECTOR_PDF,
    )
    far_line = RawLine(
        line=Line(start=Point(x=0, y=500), end=Point(x=100, y=500)),
        page=0,
        source=SourceKind.VECTOR_PDF,
    )
    candidates = detect_dimension_candidates([text_item], lines=[nearby_line, far_line])
    assert candidates[0].nearby_geometry_ids == [0]


def test_ocr_confidence_blends_into_candidate_confidence():
    high_conf_item = RawTextItem(
        text="5.0 m",
        bounding_box=BoundingBox(min_x=0, min_y=0, max_x=20, max_y=10),
        page=0,
        source=SourceKind.OCR,
        ocr_confidence=0.95,
    )
    low_conf_item = RawTextItem(
        text="5.0 m",
        bounding_box=BoundingBox(min_x=0, min_y=0, max_x=20, max_y=10),
        page=0,
        source=SourceKind.OCR,
        ocr_confidence=0.2,
    )
    high = detect_dimension_candidates([high_conf_item])[0]
    low = detect_dimension_candidates([low_conf_item])[0]
    assert high.confidence > low.confidence
