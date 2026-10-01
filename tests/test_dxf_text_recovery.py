"""
Tests for `backend.cv_extraction.dxf_text_recovery`.

OCR accuracy itself is pytesseract's problem, not this module's -- these
tests mock `pytesseract.image_to_data` and instead pin the parts this
module is actually responsible for: mapping a recovered pixel bbox back to
world coordinates, associating a recovered value with the nearest
plausible edge (reusing `dimension_candidates`'s already-hardened
prose-vs-dimension gate and distance/orientation scoring), and deriving a
scale factor only when both a unit and a confident association exist.
"""
from __future__ import annotations

from unittest.mock import patch

from backend.cv_extraction.dxf_text_recovery import (
    RecoveredDimension,
    recover_dimension_evidence,
    scale_candidates_from_recovered_dimensions,
)


def _fake_tesseract_data(entries):
    """entries: list of (text, conf, left, top, width, height) in pixels."""
    data = {"text": [], "conf": [], "left": [], "top": [], "width": [], "height": []}
    for text, conf, left, top, width, height in entries:
        data["text"].append(text)
        data["conf"].append(conf)
        data["left"].append(left)
        data["top"].append(top)
        data["width"].append(width)
        data["height"].append(height)
    return data


def test_recovered_text_is_associated_with_nearest_plausible_edge(tmp_path):
    """A recovered numeric annotation near a plot edge, with an explicit
    unit, must be associated with that edge -- not silently ignored, not
    picked as a global final value on its own."""
    # A 20x15 rectangle boundary as edge segments (world/raw drawing units).
    edges = [
        ((0.0, 0.0), (20.0, 0.0)),
        ((20.0, 0.0), (20.0, 15.0)),
        ((20.0, 15.0), (0.0, 15.0)),
        ((0.0, 15.0), (0.0, 0.0)),
    ]
    polygon_points = [[(0.0, 0.0), (20.0, 0.0), (20.0, 15.0), (0.0, 15.0)]]

    with patch("backend.cv_extraction.dxf_text_recovery._import_tesseract") as mock_import:
        mock_pytesseract = mock_import.return_value
        # A label "20.00m" recognized near the bottom edge (y=0) in pixel space.
        mock_pytesseract.image_to_data.return_value = _fake_tesseract_data([
            ("20.00m", 88.0, 100, 5, 60, 20),
        ])
        mock_pytesseract.Output.DICT = "dict"

        recovered = recover_dimension_evidence(polygon_points, [], edges)

    assert len(recovered) == 1
    r = recovered[0]
    assert r.value == 20.0
    assert r.unit_hint == "m"
    assert r.value_metres == 20.0
    assert r.associated_segment_index is not None
    # It should associate with the bottom edge (length 20), not the side edges (length 15).
    assert abs(r.associated_segment_length - 20.0) < 1.0


def test_prose_and_noise_are_not_recovered_as_dimensions(tmp_path):
    """Reuses `dimension_candidates.looks_like_dimension_text`'s gate --
    a numbered clause or a large unitless number (survey/PID-like) must
    not be treated as a dimension, exactly as for native PDF text."""
    edges = [((0.0, 0.0), (20.0, 0.0))]
    polygon_points = [[(0.0, 0.0), (20.0, 0.0), (20.0, 15.0), (0.0, 15.0)]]

    with patch("backend.cv_extraction.dxf_text_recovery._import_tesseract") as mock_import:
        mock_pytesseract = mock_import.return_value
        mock_pytesseract.image_to_data.return_value = _fake_tesseract_data([
            ("46.Due", 80.0, 10, 5, 60, 20),
            ("1234567890", 80.0, 10, 40, 100, 20),
        ])
        mock_pytesseract.Output.DICT = "dict"

        recovered = recover_dimension_evidence(polygon_points, [], edges)

    assert recovered == []


def test_no_association_when_nothing_nearby():
    """A recognized number far from every candidate edge must still be
    returned as evidence (best-effort), but WITHOUT a geometric
    association -- callers must not silently assume proximity when there
    is none."""
    edges = [((0.0, 0.0), (1.0, 0.0))]
    polygon_points = [[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]]

    with patch("backend.cv_extraction.dxf_text_recovery._import_tesseract") as mock_import:
        mock_pytesseract = mock_import.return_value
        # Placed far from the tiny edge (which sits near pixel origin).
        mock_pytesseract.image_to_data.return_value = _fake_tesseract_data([
            ("17.59m", 85.0, 2000, 2000, 60, 20),
        ])
        mock_pytesseract.Output.DICT = "dict"

        recovered = recover_dimension_evidence(polygon_points, [], edges)

    assert len(recovered) == 1
    assert recovered[0].associated_segment_index is None
    assert recovered[0].associated_segment_length is None


def test_scale_candidates_require_both_unit_and_association():
    good = RecoveredDimension(
        value=20.0, unit_hint="m", raw_text="20.00m", confidence=0.9,
        world_bbox=(0, 0, 1, 1), associated_segment_index=0, associated_segment_length=200.0,
    )
    no_unit = RecoveredDimension(
        value=20.0, unit_hint=None, raw_text="20.00", confidence=0.6,
        world_bbox=(0, 0, 1, 1), associated_segment_index=0, associated_segment_length=200.0,
    )
    no_association = RecoveredDimension(
        value=20.0, unit_hint="m", raw_text="20.00m", confidence=0.9,
        world_bbox=(0, 0, 1, 1), associated_segment_index=None, associated_segment_length=None,
    )

    pairs = scale_candidates_from_recovered_dimensions([good, no_unit, no_association])
    assert len(pairs) == 1
    scale, evidence = pairs[0]
    # 20 real metres over a 200-drawing-unit segment -> 0.1 m per drawing unit
    # (e.g. the sheet is actually drawn in decimetres, or at some other scale).
    assert scale == 0.1
    assert evidence is good


def test_empty_geometry_returns_empty_without_error():
    assert recover_dimension_evidence([], [], []) == []


# --- Drawing-type caption matching (DXF_FAILURE_TAXONOMY.md item 0) ---------


def _text_item(text, x, y, w=0.15, h=0.2, conf=0.9):
    from backend.cv_extraction.raw_types import RawTextItem, SourceKind
    from backend.schemas.geometry import BoundingBox

    return RawTextItem(
        text=text, bounding_box=BoundingBox(min_x=x, min_y=y, max_x=x + w, max_y=y + h),
        page=0, source=SourceKind.OCR, ocr_confidence=conf,
    )


def test_isolated_site_plan_caption_is_matched():
    from backend.cv_extraction.dxf_text_recovery import _find_caption_matches

    items = [_text_item("SITE", 10.0, 6.0), _text_item("PLAN", 10.0, 5.5)]
    matches = _find_caption_matches(items, [("SITE", "PLAN")])
    assert len(matches) == 1
    assert matches[0].text == "SITE PLAN"


def test_isolated_caption_with_a_few_nearby_supporting_words_still_matches():
    """A real drawing caption is routinely more than just the two anchor
    keywords -- "SITE PLAN SCALE 1:200" -- the isolation check must not
    reject a genuine title just because it has a couple of nearby
    supporting words."""
    from backend.cv_extraction.dxf_text_recovery import _find_caption_matches

    items = [
        _text_item("SITE", 10.0, 6.0), _text_item("PLAN", 10.0, 5.5),
        _text_item("SCALE", 10.0, 5.0), _text_item("1:200", 10.0, 4.5),
    ]
    matches = _find_caption_matches(items, [("SITE", "PLAN")])
    assert len(matches) == 1


def test_keyword_pair_embedded_in_a_dense_paragraph_is_not_matched_as_a_caption():
    """Refuse rather than guess: 'SITE' and 'PLAN' both appearing as
    ordinary words inside a genuinely dense paragraph (a notes block, not
    just a busy-but-short caption area) must not be mistaken for a
    drawing title. This threshold was measured against PLAN5's own real
    caption (11 nearby labels) and set well above it (see
    MAX_NEARBY_UNRELATED_WORDS_FOR_ISOLATED_CAPTION's own comment) -- this
    test uses a deliberately much denser block to still demonstrate the
    refusal exists, not to pin the exact boundary between the two, which
    isn't validated yet."""
    from backend.cv_extraction.dxf_text_recovery import _find_caption_matches

    paragraph_words = (
        "PLEASE CAREFULLY REFER TO THE ATTACHED SITE PLAN FOR EXACT SETBACK DETAILS "
        "AND ADJOINING OWNER CONSENT LETTER BEFORE ANY SITE COMMENCEMENT OF WORK ONSITE "
        "AS PER THE ZONAL REGULATION DRAWING NUMBER AND THE APPROVED SANCTION PLAN COPY "
        "ISSUED BY THE COMPETENT AUTHORITY ALONG WITH THE KHATA CERTIFICATE"
    ).split()
    items = [_text_item(word, x=i * 0.05, y=0.0) for i, word in enumerate(paragraph_words)]
    matches = _find_caption_matches(items, [("SITE", "PLAN")])
    assert matches == []


def test_no_match_when_a_keyword_is_never_recognized_anywhere():
    from backend.cv_extraction.dxf_text_recovery import _find_caption_matches

    items = [_text_item("SITE", 10.0, 6.0)]  # "PLAN" never appears at all
    assert _find_caption_matches(items, [("SITE", "PLAN")]) == []
