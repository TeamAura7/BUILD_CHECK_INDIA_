"""
Tests for `backend.spatial_reasoning.building_context.extract_plan_context`
-- deterministic text-based extraction of building use, development-area
class, and height-excluding-stilt, none of which had any real extraction
path before (the only wiring was to a `VisionPageResult.metadata` dict
that nothing in the Vision pipeline ever populated), which is why these
fields were unconditionally MISSING in production regardless of what the
source plan printed.
"""

from __future__ import annotations

from backend.spatial_reasoning.building_context import extract_plan_context
from tests.fixtures.geometry_builders import text


def test_labeled_building_use_is_extracted():
    ctx = extract_plan_context([text("NATURE OF WORK : RESIDENTIAL BUILDING")])
    assert ctx.building_use_raw == "residential"


def test_unlabeled_title_building_use_is_extracted():
    ctx = extract_plan_context([text("PROPOSED COMMERCIAL BUILDING")])
    assert ctx.building_use_raw == "commercial"


def test_public_semi_public_use_variant_is_extracted():
    ctx = extract_plan_context([text("USE OF BUILDING: PUBLIC & SEMI PUBLIC")])
    assert ctx.building_use_raw is not None
    assert "public" in ctx.building_use_raw.lower()


def test_no_building_use_text_stays_none():
    ctx = extract_plan_context([text("SITE AREA : 160.77 Sq.m")])
    assert ctx.building_use_raw is None


def test_development_area_class_is_extracted():
    ctx = extract_plan_context([text("DEVELOPMENT AREA : B")])
    assert ctx.development_area == "B"


def test_development_area_requires_the_explicit_label_not_any_nearby_letter():
    ctx = extract_plan_context([text("PLOT NO. A-42, BLOCK B")])
    assert ctx.development_area is None


def test_height_excluding_stilt_is_extracted():
    ctx = extract_plan_context([text("HEIGHT OF BUILDING EXCLUDING STILT FLOOR : 12.50 M")])
    assert ctx.height_excluding_stilt_m == 12.5


def test_height_excluding_stilt_short_label_variant_is_extracted():
    ctx = extract_plan_context([text("HT EXCL. STILT: 9.2")])
    assert ctx.height_excluding_stilt_m == 9.2


def test_ordinary_overall_height_without_stilt_wording_is_not_matched():
    ctx = extract_plan_context([text("HEIGHT OF BUILDING : 12.50 M")])
    assert ctx.height_excluding_stilt_m is None


def test_first_match_wins_across_multiple_text_spans():
    ctx = extract_plan_context([
        text("NATURE OF WORK : RESIDENTIAL BUILDING"),
        text("DEVELOPMENT AREA : A"),
        text("HEIGHT OF BUILDING EXCLUDING STILT : 11.0"),
        text("NATURE OF WORK : COMMERCIAL BUILDING"),  # a second, contradicting span must not override
    ])
    assert ctx.building_use_raw == "residential"
    assert ctx.development_area == "A"
    assert ctx.height_excluding_stilt_m == 11.0


def test_empty_text_evidence_returns_all_none():
    ctx = extract_plan_context([])
    assert ctx.building_use_raw is None
    assert ctx.development_area is None
    assert ctx.height_excluding_stilt_m is None


def test_use_at_the_end_of_a_long_construction_title_is_extracted():
    """No "PROPOSED" directly before the use, as on a real sheet whose title
    ends "...FLOOR AND THIRD FLOOR RCC ROOF RESIDENTIAL BUILDING , IN S.NO ..."."""
    title = (
        "PLAN  SHOWING  THE  PROPOSED  CONSTRUCTION  OF  GROUND FLOOR , FIRST FLOOR , "
        "SECOND FLOOR AND THIRD FLOOR  RCC  ROOF  RESIDENTIAL  BUILDING , IN S.NO : 26 / 2A"
    )
    assert extract_plan_context([text(title)]).building_use_raw == "residential"


def test_the_bare_phrase_in_regulation_prose_is_not_a_use_label():
    """Notes say "residential building" constantly; only a construction-title
    item may supply the use without an explicit label."""
    prose = "22. Every residential building shall provide rainwater harvesting as per the byelaws."
    assert extract_plan_context([text(prose)]).building_use_raw is None
