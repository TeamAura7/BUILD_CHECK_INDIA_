"""Tests for `backend.spatial_reasoning.floor_levels` (floor count from named levels)."""

from __future__ import annotations

from backend.spatial_reasoning.floor_levels import floor_count_from_named_levels

TITLE = "PLAN SHOWING THE PROPOSED CONSTRUCTION OF GROUND FLOOR , FIRST FLOOR , SECOND FLOOR AND THIRD FLOOR RCC ROOF"
CAPTIONS = ["GROUND FLOOR PLAN", "FIRST FLOOR PLAN", "SECOND FLOOR PLAN", "THIRD FLOOR PLAN"]
TABLE = [
    "PROP. PLINTH AREA IN G.F", "PROP. PLINTH AREA IN F.F",
    "PROP. PLINTH AREA IN S.F", "PROP. PLINTH AREA IN T.F", "TOTAL PLINTH AREA",
]


def test_three_independent_forms_that_agree_give_the_count_and_name_all_three():
    result = floor_count_from_named_levels([TITLE, *CAPTIONS, *TABLE])
    assert result is not None
    assert result.count == 4
    assert len(result.forms) == 3


def test_each_form_alone_is_enough_to_count_but_names_only_itself():
    for texts in ([TITLE], CAPTIONS, TABLE):
        result = floor_count_from_named_levels(texts)
        assert result is not None and result.count == 4
        assert len(result.forms) == 1


def test_forms_that_disagree_abstain_rather_than_pick_one():
    assert floor_count_from_named_levels([TITLE, "GROUND FLOOR PLAN", "FIRST FLOOR PLAN"]) is None


def test_levels_must_start_at_ground_and_have_no_gaps():
    assert floor_count_from_named_levels(["FIRST FLOOR PLAN"]) is None
    assert floor_count_from_named_levels(["GROUND FLOOR PLAN", "SECOND FLOOR PLAN"]) is None


def test_a_stilt_level_is_not_counted_as_a_floor():
    """Convention: ground + upper floors. "STILT, GF+2UF" is 3, not 4."""
    stilt = ["STILT FLOOR PLAN", "GROUND FLOOR PLAN", "FIRST FLOOR PLAN", "SECOND FLOOR PLAN"]
    result = floor_count_from_named_levels(stilt)
    assert result is not None and result.count == 3


def test_a_basement_or_podium_level_makes_the_count_ambiguous_so_abstain():
    assert floor_count_from_named_levels(["BASEMENT FLOOR PLAN", *CAPTIONS]) is None
    assert floor_count_from_named_levels([*CAPTIONS, "PODIUM FLOOR PLAN"]) is None


def test_one_caption_may_cover_several_levels_by_word_or_digit_ordinal():
    plan4 = ["GROUND FLOOR PLAN", "1ST, 2ND & 3RD FLOOR PLAN", "FOURTH FLOOR PLAN"]
    assert floor_count_from_named_levels(plan4).count == 5
    plan8 = ["GROUND FLOOR PLAN", "TYPICAL FIRST & SECOND FLOOR PLAN"]
    assert floor_count_from_named_levels(plan8).count == 3


def test_a_sheet_whose_captions_and_table_labels_disagree_abstains():
    """Captions name ground+first only, but the area table also lists a second
    floor: one of them is incomplete and nothing says which."""
    texts = ["GROUND FLOOR PLAN", "FIRST FLOOR PLAN", "SECOND FLOOR", "FIRST FLOOR", "GROUND FLOOR"]
    assert floor_count_from_named_levels(texts) is None


def test_a_table_listing_only_some_floors_is_ignored_not_trusted():
    texts = [*CAPTIONS, "Third Floor", "First Floor", "Terrace"]
    assert floor_count_from_named_levels(texts).count == 4


def test_boilerplate_mentioning_stilt_is_not_a_drawn_stilt_level():
    texts = ["(Excluding Stilt Floor)", *CAPTIONS]
    result = floor_count_from_named_levels(texts)
    assert result is not None and result.count == 4


def test_terrace_and_roof_are_not_floors():
    result = floor_count_from_named_levels([*CAPTIONS, "TERRACE FLOOR PLAN", "RCC ROOF"])
    assert result is not None and result.count == 4


def test_a_street_address_or_single_note_is_not_read_as_a_floor_list():
    assert floor_count_from_named_levels(["#56, 3rd Floor, Kabeer Mutt Road"]) is None
    assert floor_count_from_named_levels(["26.The applicant shall provide a toilet in the ground floor for the use of"]) is None


def test_no_evidence_abstains():
    assert floor_count_from_named_levels([]) is None
    assert floor_count_from_named_levels(["SITE PLAN", "SCALE 1:100"]) is None
