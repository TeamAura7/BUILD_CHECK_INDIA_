"""
Tests for building-height resolution in
`backend.spatial_reasoning.pipeline.build_normalized_plan`: overall height
must be resolved separately from floor height and from regulatory-maximum-
height text, and a regulatory height figure must never become the plan's
`building_height_estimated` nor any other Python threshold.

There was no existing coverage of this at all (grep for
`building_height_estimated`/`BUILDING_HEIGHT` across `tests/` found
nothing) despite `pipeline.py` already implementing the separation -- this
closes that gap and pins the new `REGULATORY_TEXT` dimension type
(introduced for the ELEVATION/SECTION height focus pass, see
`vision_extraction/prompts.py::HEIGHT_FOCUS_PROMPT`) as something the
deterministic layer must ignore.
"""

from __future__ import annotations

from backend.schemas.enums import ConfidenceLevel
from backend.schemas.vision import VisionDimension, VisionPageResult
from backend.spatial_reasoning.pipeline import build_normalized_plan
from tests.fixtures.geometry_builders import standard_rectangular_plan


def _with_vision_dimensions(dimensions: list[VisionDimension]):
    er = standard_rectangular_plan()
    er.vision_pages = [VisionPageResult(page_number=1, dimensions=dimensions)]
    return er


def test_explicit_building_height_is_used_directly():
    er = _with_vision_dimensions([
        VisionDimension(value=9.6, unit="m", type="BUILDING_HEIGHT", evidence="9.60", confidence=0.9),
    ])
    plan = build_normalized_plan(er)
    assert plan.building_height_estimated is not None
    assert plan.building_height_estimated.value == 9.6


def test_floor_height_alone_without_floor_count_does_not_produce_a_height():
    """Deriving height needs BOTH floor_to_floor height AND a resolved floor
    count -- neither may be assumed, so with no floor count resolved the
    field must stay MISSING rather than guessing a floor count of 1."""
    er = _with_vision_dimensions([
        VisionDimension(value=3.0, unit="m", type="FLOOR_HEIGHT", evidence="3.00", confidence=0.9),
    ])
    plan = build_normalized_plan(er)
    assert plan.building_height_estimated is None or plan.building_height_estimated.value is None


def test_floor_height_times_resolved_floor_count_derives_height_only_when_no_explicit_height():
    er = _with_vision_dimensions([
        VisionDimension(value=3.0, unit="m", type="FLOOR_HEIGHT", evidence="3.00", confidence=0.9),
    ])
    # Floor count independently resolved via a text label, same as production evidence.
    from tests.fixtures.geometry_builders import text
    from backend.schemas.geometry import BoundingBox

    er.text_evidence.append(text("No. of floors: 2", bbox=BoundingBox(min_x=0, min_y=0, max_x=50, max_y=10)))
    plan = build_normalized_plan(er)
    assert plan.building.floor_count is not None and plan.building.floor_count.value == 2
    assert plan.building_height_estimated is not None
    assert plan.building_height_estimated.value == 6.0  # 2 floors x 3.0 m -- no assumed generic storey height
    assert plan.building_height_estimated.confidence.level in (ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW)


def test_explicit_building_height_wins_over_a_derivable_floor_height_stack():
    er = _with_vision_dimensions([
        VisionDimension(value=3.0, unit="m", type="FLOOR_HEIGHT", evidence="3.00", confidence=0.9),
        VisionDimension(value=12.5, unit="m", type="BUILDING_HEIGHT", evidence="12.50", confidence=0.9),
    ])
    from tests.fixtures.geometry_builders import text
    from backend.schemas.geometry import BoundingBox

    er.text_evidence.append(text("No. of floors: 2", bbox=BoundingBox(min_x=0, min_y=0, max_x=50, max_y=10)))
    plan = build_normalized_plan(er)
    # The explicit printed overall height (12.5) is strictly stronger
    # evidence than a derived 2 x 3.0 = 6.0 -- never silently overridden by
    # the derivation, and never averaged with it.
    assert plan.building_height_estimated.value == 12.5


def test_regulatory_max_height_text_never_becomes_the_proposed_building_height():
    """
    `HEIGHT_FOCUS_PROMPT` asks Vision to tag a printed "max permitted
    height" figure as REGULATORY_TEXT specifically so it can never be
    confused with this building's actual proposed height. `pipeline.py`'s
    height loop only recognizes dimension types containing "HEIGHT" via an
    explicit elif chain (BUILDING_HEIGHT/FLOOR_HEIGHT/PLINTH_HEIGHT/
    PARAPET_HEIGHT) -- REGULATORY_TEXT must fall through untouched, not
    silently match a substring check.
    """
    er = _with_vision_dimensions([
        VisionDimension(value=15.0, unit="m", type="REGULATORY_TEXT", evidence="MAX HEIGHT PERMISSIBLE: 15.0 M", confidence=0.9),
    ])
    plan = build_normalized_plan(er)
    assert plan.building_height_estimated is None or plan.building_height_estimated.value is None
    assert plan.building_height_estimated is None or plan.building_height_estimated.value != 15.0


def test_regulatory_text_does_not_contaminate_an_explicit_real_height():
    er = _with_vision_dimensions([
        VisionDimension(value=15.0, unit="m", type="REGULATORY_TEXT", evidence="MAX HEIGHT PERMISSIBLE: 15.0 M", confidence=0.9),
        VisionDimension(value=9.6, unit="m", type="BUILDING_HEIGHT", evidence="9.60", confidence=0.9),
    ])
    plan = build_normalized_plan(er)
    assert plan.building_height_estimated is not None
    assert plan.building_height_estimated.value == 9.6


def test_proposed_building_height_is_treated_the_same_as_building_height():
    """PROPOSED_BUILDING_HEIGHT is the same real-world quantity as
    BUILDING_HEIGHT under a different sheet label -- must populate
    building_height_estimated directly, same as BUILDING_HEIGHT does."""
    er = _with_vision_dimensions([
        VisionDimension(value=9.6, unit="m", type="PROPOSED_BUILDING_HEIGHT", evidence="PROPOSED HEIGHT: 9.60 M", confidence=0.9),
    ])
    plan = build_normalized_plan(er)
    assert plan.building_height_estimated is not None
    assert plan.building_height_estimated.value == 9.6


def test_regulatory_max_height_never_becomes_the_proposed_building_height():
    """REGULATORY_MAX_HEIGHT (the new, more specific regulatory-ceiling
    type) must be ignored by the height-resolution elif chain exactly like
    REGULATORY_TEXT -- a legal ceiling is never this building's actual
    height no matter which of the two regulatory types Vision used."""
    er = _with_vision_dimensions([
        VisionDimension(value=15.0, unit="m", type="REGULATORY_MAX_HEIGHT", evidence="MAX HEIGHT PERMISSIBLE AS PER RULE: 15.0 M", confidence=0.9),
    ])
    plan = build_normalized_plan(er)
    assert plan.building_height_estimated is None or plan.building_height_estimated.value is None
    assert plan.overall_confidence_note is not None
    assert "Regulatory height text" in plan.overall_confidence_note


def test_stilt_height_adds_as_its_own_component_in_a_derived_estimate():
    er = _with_vision_dimensions([
        VisionDimension(value=3.0, unit="m", type="FLOOR_HEIGHT", evidence="3.00", confidence=0.9),
        VisionDimension(value=2.4, unit="m", type="STILT_HEIGHT", evidence="STILT: 2.40", confidence=0.9),
    ])
    from tests.fixtures.geometry_builders import text
    from backend.schemas.geometry import BoundingBox

    er.text_evidence.append(text("No. of floors: 2", bbox=BoundingBox(min_x=0, min_y=0, max_x=50, max_y=10)))
    plan = build_normalized_plan(er)
    assert plan.building_height_estimated is not None
    # 2 floors x 3.0 m + 2.4 m stilt = 8.4 m -- stilt is a distinct labeled
    # component, not folded into (or omitted from) the floor-height stack.
    assert plan.building_height_estimated.value == 8.4


def test_building_use_development_area_and_height_excluding_stilt_derived_from_plan_text():
    """
    Integration: these three fields previously reached `build_normalized_plan`
    only via a `VisionPageResult.metadata` dict that nothing in the Vision
    pipeline ever populated, so they were unconditionally MISSING in
    production. `building_context.extract_plan_context` now reads them
    directly from the plan's own printed text (works for a PDF's native/OCR
    text and a DXF's TEXT/MTEXT entities alike, since both populate
    `extraction.text_evidence`), with no Vision involvement at all.
    """
    from backend.schemas.geometry import BoundingBox
    from tests.fixtures.geometry_builders import standard_rectangular_plan, text

    er = standard_rectangular_plan()
    er.text_evidence.extend([
        text("NATURE OF WORK : RESIDENTIAL BUILDING", bbox=BoundingBox(min_x=0, min_y=0, max_x=80, max_y=10)),
        text("DEVELOPMENT AREA : B", bbox=BoundingBox(min_x=0, min_y=12, max_x=80, max_y=22)),
        text(
            "HEIGHT OF BUILDING EXCLUDING STILT FLOOR : 12.50 M",
            bbox=BoundingBox(min_x=0, min_y=24, max_x=100, max_y=34),
        ),
    ])
    plan = build_normalized_plan(er)

    assert plan.building_use is not None and plan.building_use.value == "residential"
    assert plan.development_area is not None and plan.development_area.value == "B"
    assert plan.building_height_excluding_stilt is not None
    assert plan.building_height_excluding_stilt.value == 12.5


def test_no_matching_text_leaves_these_fields_missing_not_guessed():
    plan = build_normalized_plan(standard_rectangular_plan())
    assert plan.building_use is None or plan.building_use.value is None
    assert plan.development_area is None or plan.development_area.value is None
    assert plan.building_height_excluding_stilt is None or plan.building_height_excluding_stilt.value is None


def test_height_excluding_stilt_is_a_distinct_field_never_merged_into_building_height():
    er = _with_vision_dimensions([
        VisionDimension(value=9.6, unit="m", type="BUILDING_HEIGHT", evidence="9.60", confidence=0.9),
        VisionDimension(value=7.2, unit="m", type="HEIGHT_EXCLUDING_STILT", evidence="HEIGHT EXCL. STILT: 7.20 M", confidence=0.9),
    ])
    plan = build_normalized_plan(er)
    assert plan.building_height_estimated is not None
    assert plan.building_height_estimated.value == 9.6
    assert plan.building_height_excluding_stilt is not None
    assert plan.building_height_excluding_stilt.value == 7.2
    # The two must genuinely stay distinct, not silently collapse to one value.
    assert plan.building_height_estimated.value != plan.building_height_excluding_stilt.value


def test_unset_vision_confidence_reports_low_not_medium():
    """A VisionDimension whose confidence was never populated defaults to
    0.0 (see backend/schemas/vision.py) -- ambiguous between "measured as
    zero" and "never set". Either way it must never be reported as MEDIUM:
    a binary `HIGH if conf >= .85 else MEDIUM` mapping has no LOW branch and
    would silently bypass the compliance engine's LOW -> REQUIRES_REVIEW
    safety gate (see backend/schemas/enums.py::confidence_level_from_score)."""
    er = _with_vision_dimensions([
        VisionDimension(value=9.6, unit="m", type="BUILDING_HEIGHT", evidence="9.60"),  # confidence omitted -> 0.0
    ])
    plan = build_normalized_plan(er)
    assert plan.building_height_estimated is not None
    assert plan.building_height_estimated.value == 9.6
    assert plan.building_height_estimated.confidence.level == ConfidenceLevel.LOW


def test_high_vision_confidence_still_reports_high():
    """Regression guard: the LOW fix must not have lowered the HIGH threshold."""
    er = _with_vision_dimensions([
        VisionDimension(value=9.6, unit="m", type="BUILDING_HEIGHT", evidence="9.60", confidence=0.9),
    ])
    plan = build_normalized_plan(er)
    assert plan.building_height_estimated.confidence.level == ConfidenceLevel.HIGH
