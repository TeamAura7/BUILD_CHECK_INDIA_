"""
Tests for building.floor_count resolution in
`backend.spatial_reasoning.pipeline`.

There was previously no coverage of this at all. Two real bugs were found
live on PLAN5.pdf's Fusion mode, both traced to the same root cause: a
weaker/heuristic source being preferred over a stronger, explicit one.

1. `_FLOOR_COUNT_RE`'s "G+N" pattern required the literal letter "g"
   immediately before "+", so it never matched the sheet's actual wording,
   "(GROUND +2 FLOORS)". The regex's OWN fallback branch (a bare "<N>
   floors") then matched the "2" in that same phrase directly, reporting
   `floor_count=2` -- but "(GROUND +2 FLOORS)" means ground plus two
   floors, i.e. 3 floors total, not 2.
2. Even after (1) was fixed, `build_normalized_plan` tried Vision's
   distinct-floor-plan-region COUNT first and only fell back to the
   explicit text second -- backwards, since counting regions conflates a
   genuine additional floor with a TERRACE/STILT/BASEMENT region (which
   `region_detection`'s OTHER_FLOOR_PLAN type also matches), while an
   explicit printed floor count is authoritative. On PLAN5.pdf this made
   the dashboard show `floor_count=4` (GROUND + FIRST + SECOND + TERRACE
   regions counted) instead of the sheet's own explicit 3 (G+2).
"""

from __future__ import annotations

from pathlib import Path

from backend.schemas.evidence import TextEvidence
from backend.schemas.extraction import ExtractionResult
from backend.schemas.geometry import BoundingBox
from backend.schemas.vision import VisionPageResult, VisionRegion
from backend.spatial_reasoning.pipeline import _detect_floor_count, _FLOOR_COUNT_RE
from backend.spatial_reasoning.vision_semantics import floor_count_from_vision
from tests.fixtures.geometry_builders import standard_rectangular_plan, text


def test_ground_plus_n_floors_phrasing_is_recognized():
    """The exact wording found on the real PLAN5.pdf sheet."""
    m = _FLOOR_COUNT_RE.search("(GROUND +2 FLOORS)")
    assert m is not None
    assert m.group("plus") == "2"
    assert m.group("stories") is None  # must not also/instead match via the bare "<N> floors" fallback


def test_detect_floor_count_computes_ground_plus_n_as_n_plus_one():
    text_evidence = [text("(GROUND +2 FLOORS)", bbox=BoundingBox(min_x=0, min_y=0, max_x=100, max_y=20))]
    result = _detect_floor_count(text_evidence)
    assert result is not None
    assert result.value == 3  # ground + 2 upper floors = 3 floors, not the printed "2"


def test_detect_floor_count_does_not_misread_a_street_address():
    text_evidence = [text("#56, 3rd Floor, Kabeer Mutt Road", bbox=BoundingBox(min_x=0, min_y=0, max_x=100, max_y=20))]
    assert _detect_floor_count(text_evidence) is None


def test_explicit_floor_count_text_wins_over_vision_region_count():
    """
    Reproduces the PLAN5.pdf case: an explicit "(GROUND +2 FLOORS)" label
    (-> 3 floors) coexists with 4 distinct *_FLOOR_PLAN vision regions
    (GROUND/FIRST/SECOND/TERRACE). The explicit text must win.
    """
    er = standard_rectangular_plan()
    er.text_evidence.append(text("(GROUND +2 FLOORS)", bbox=BoundingBox(min_x=0, min_y=0, max_x=100, max_y=20)))
    er.vision_pages = [VisionPageResult(
        page_number=1,
        regions=[
            VisionRegion(id="r1", type="GROUND_FLOOR_PLAN", confidence=0.95),
            VisionRegion(id="r2", type="FIRST_FLOOR_PLAN", confidence=0.95),
            VisionRegion(id="r3", type="SECOND_FLOOR_PLAN", confidence=0.95),
            VisionRegion(id="r4", type="OTHER_FLOOR_PLAN", confidence=0.95, label="TERRACE FLOOR PLAN"),
        ],
    )]

    from backend.spatial_reasoning.pipeline import build_normalized_plan
    plan = build_normalized_plan(er)
    assert plan.building.floor_count is not None
    assert plan.building.floor_count.value == 3
    assert plan.building.floor_count.source != "vision floor-plan region count"


def test_vision_region_count_still_used_when_no_explicit_text_exists():
    er = standard_rectangular_plan()
    er.vision_pages = [VisionPageResult(
        page_number=1,
        regions=[
            VisionRegion(id="r1", type="GROUND_FLOOR_PLAN", confidence=0.95),
            VisionRegion(id="r2", type="FIRST_FLOOR_PLAN", confidence=0.95),
        ],
    )]
    from backend.spatial_reasoning.pipeline import build_normalized_plan
    plan = build_normalized_plan(er)
    assert plan.building.floor_count is not None
    assert plan.building.floor_count.value == 2
    assert plan.building.floor_count.source == "vision floor-plan region count"


def test_floor_count_from_vision_counts_terrace_as_a_distinct_region():
    """
    Documents the known limitation being worked around above: this
    function itself has no way to distinguish a genuine additional floor
    from a terrace/stilt/basement region -- that's exactly why explicit
    text must be preferred over it, not a bug in this function per se.
    """
    er = ExtractionResult(document_id="d", document_type="VECTOR_PDF", page_count=1)
    er.vision_pages = [VisionPageResult(
        page_number=1,
        regions=[
            VisionRegion(id="r1", type="GROUND_FLOOR_PLAN", confidence=0.95),
            VisionRegion(id="r2", type="OTHER_FLOOR_PLAN", confidence=0.95, label="TERRACE FLOOR PLAN"),
        ],
    )]
    assert floor_count_from_vision(er, page=0) == 2


def test_a_floor_plan_caption_is_not_read_as_a_floor_count():
    """"2 FLOOR PLAN" is the caption of the second floor's drawing. Reading it
    as "2 floors" shipped a MEDIUM-confidence wrong count on a real sheet (true
    count 3)."""
    assert _FLOOR_COUNT_RE.search("2 FLOOR PLAN") is None
    assert _detect_floor_count([text("2 FLOOR PLAN", bbox=BoundingBox(min_x=0, min_y=0, max_x=100, max_y=20))]) is None
    assert _FLOOR_COUNT_RE.search("3 floors").group("stories") == "3"


def test_gf_plus_n_uf_notation_counts_ground_plus_n_upper_floors_and_ignores_stilt():
    m = _FLOOR_COUNT_RE.search("Consisting of STILT, GF+2UF")
    assert m is not None and m.group("plus") == "2"
    result = _detect_floor_count([text("Consisting of STILT, GF+2UF", bbox=BoundingBox(min_x=0, min_y=0, max_x=100, max_y=20))])
    assert result.value == 3
    # "STILT+2UF" with no "GF+" is not the ground-plus-N form; nothing is guessed.
    assert _FLOOR_COUNT_RE.search("GF, STILT+2UF") is None
